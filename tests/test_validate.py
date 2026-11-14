"""The validate mode: the check framework, the mode verdicts and the command line (validate.py, main.py)."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import constants
import main
import validate
from validate import (Check, Context, Fail, NotNeeded, Result, Skip, Status, Warn, mode_verdict, run_check,
                      run_checks)


def ok(ctx):
    return "fine"


def fails(ctx):
    raise Fail("broken", "fix it")


def checks_of(*specs) -> list[Check]:
    return [Check(key, key, fn, **kwargs) for key, fn, kwargs in specs]


# ------------------------------------------------- run_check -------------------------------------------------

class TestRunCheck:
    def test_pass_with_the_message(self):
        result = run_check(Check('a', 'A', ok), Context())
        assert (result.status, result.message) == (Status.PASS, 'fine')

    @pytest.mark.parametrize('exc, status', [(Fail, Status.FAIL), (Warn, Status.WARN), (Skip, Status.SKIP)])
    def test_outcomes_keep_message_and_hint(self, exc, status):
        def fn(ctx):
            raise exc("message", "hint")
        result = run_check(Check('a', 'A', fn), Context())
        assert (result.status, result.message, result.hint) == (status, 'message', 'hint')
        assert not result.not_needed

    def test_not_needed_is_a_skip_that_is_marked(self):
        def fn(ctx):
            raise NotNeeded("not used")
        result = run_check(Check('a', 'A', fn), Context())
        assert result.status == Status.SKIP and result.not_needed

    def test_an_exception_is_a_fail(self):
        def fn(ctx):
            raise KeyError('x')
        result = run_check(Check('a', 'A', fn), Context())
        assert result.status == Status.FAIL and 'KeyError' in result.message

    def test_a_check_that_hangs_times_out(self):
        result = run_check(Check('a', 'A', lambda ctx: time.sleep(5), timeout=0.1), Context())
        assert result.status == Status.FAIL and 'did not finish' in result.message

    def test_a_check_with_a_timeout_gives_its_result(self):
        result = run_check(Check('a', 'A', fails, timeout=5), Context())
        assert (result.status, result.message) == (Status.FAIL, 'broken')


# ------------------------------------------------- run_checks ------------------------------------------------

class TestRunChecks:
    def test_dependents_of_a_failed_check_are_skipped(self):
        called = []
        checks = checks_of(('a', fails, {}), ('b', lambda ctx: called.append('b'), {'requires': ('a',)}))
        results = run_checks(checks, Context(), (), quick=False)
        assert results['b'].status == Status.SKIP and 'needs: a' in results['b'].message
        assert not called

    def test_dependents_of_a_warning_run(self):
        def warns(ctx):
            raise Warn("hmm")
        checks = checks_of(('a', warns, {}), ('b', ok, {'requires': ('a',)}))
        assert run_checks(checks, Context(), (), quick=False)['b'].status == Status.PASS

    def test_a_requirement_that_did_not_run_skips(self):
        checks = checks_of(('a', ok, {'mode': 'http'}), ('b', ok, {'requires': ('a',)}))
        results = run_checks(checks, Context(), (), quick=False)
        assert 'a' not in results and results['b'].status == Status.SKIP

    def test_not_needed_carries_over_to_dependents(self):
        def not_needed(ctx):
            raise NotNeeded("unused")
        checks = checks_of(('a', not_needed, {}), ('b', ok, {'requires': ('a',)}))
        assert run_checks(checks, Context(), (), quick=False)['b'].not_needed

    def test_only_the_given_modes_run(self):
        checks = checks_of(('core', ok, {}), ('f', ok, {'mode': 'file'}), ('h', ok, {'mode': 'http'}))
        assert set(run_checks(checks, Context(), ('file',), quick=False)) == {'core', 'f'}

    def test_quick_skips_slow_checks(self):
        called = []
        checks = checks_of(('slow', lambda ctx: called.append(1), {'slow': True}))
        result = run_checks(checks, Context(), (), quick=True)['slow']
        assert result.status == Status.SKIP and '--quick' in result.message and not called


# ------------------------------------------------- verdicts --------------------------------------------------

class TestModeVerdict:
    checks = checks_of(('a', ok, {}), ('b', ok, {}), ('c', ok, {}))

    def verdict(self, **statuses):
        results = {key: (r if isinstance(r, Result) else Result(r, f"{key} says")) for key, r in statuses.items()}
        return mode_verdict(('a', 'b', 'c'), results, self.checks)

    def test_all_pass(self):
        assert self.verdict(a=Status.PASS, b=Status.PASS, c=Status.PASS) == (Status.PASS, 'should work')

    def test_the_first_failure_is_named(self):
        status, text = self.verdict(a=Status.PASS, b=Status.FAIL, c=Status.FAIL)
        assert status == Status.FAIL and 'NOT WORKING: b: b says' in text

    def test_warnings_are_listed(self):
        status, text = self.verdict(a=Status.WARN, b=Status.PASS, c=Status.WARN)
        assert status == Status.WARN and '(a, c)' in text

    def test_a_skip_means_not_tested(self):
        status, text = self.verdict(a=Status.PASS, b=Status.SKIP)
        assert status == Status.SKIP and 'not tested' in text

    def test_a_not_needed_skip_does_not_count(self):
        status, _ = self.verdict(a=Status.PASS, b=Result(Status.SKIP, 'unused', not_needed=True))
        assert status == Status.PASS

    def test_a_failure_wins_over_a_skip(self):
        assert self.verdict(a=Status.SKIP, b=Status.FAIL)[0] == Status.FAIL


def test_report_ignores_failed_optional_checks():
    checks = checks_of(('a', ok, {}), ('extra', ok, {'optional': True}))
    results = {'a': Result(Status.PASS, 'ok'), 'extra': Result(Status.FAIL, 'nope')}
    assert validate.report(results, checks, (), 'p')
    results['a'] = Result(Status.FAIL, 'broken')
    assert not validate.report(results, checks, (), 'p')


def test_every_mode_verdict_uses_existing_checks():
    keys = {c.key for c in validate.CHECKS}
    for name, needed in validate.MODE_CHECKS.items():
        assert set(needed) <= keys, name
        assert validate.MODE_OF_VERDICT[name] in validate.MODES
    assert set(validate.EXTRAS) <= keys


def test_requirements_come_before_the_checks_that_need_them():
    seen = set()
    for check in validate.CHECKS:
        assert set(check.requires) <= seen, check.key
        seen.add(check.key)


def test_validate_install_rejects_unknown_modes():
    with pytest.raises(ValueError, match='unknown mode'):
        validate.validate_install(modes=('nope',))


# ------------------------------------------------- helpers ---------------------------------------------------

def test_requested_devices():
    spec = {'device': 'cuda', 'models': [{'name': 'a'}, {'name': 'b', 'device': 'CPU'}],
            'landmarks': {'path': 'x.onnx'}}
    assert validate.requested_devices(spec) == {'a': 'cuda', 'b': 'cpu', 'landmarks': 'cuda'}
    assert validate.requested_devices({'models': [{'name': 'a'}]}) == {'a': 'cpu'}


@pytest.mark.parametrize('device, gpu', [('cpu', False), ('cuda', True), ('cuda:1', True), ('gpu', True)])
def test_is_gpu_device(device, gpu):
    assert validate.is_gpu_device(device) == gpu


def test_capture_fd_output_catches_writes_to_the_file_descriptor(capfd):
    with capfd.disabled():
        with validate.capture_fd_output() as captured:
            os.write(sys.stdout.fileno(), b"from C\n")
            print("from python", flush=True)
    assert captured == ["from C\nfrom python\n"]


def test_test_image_is_deterministic():
    a, b = validate.test_image(), validate.test_image()
    assert a.shape == (240, 320, 3) and a.dtype.name == 'uint8' and (a == b).all() and a.std() > 10


def test_check_https_cert_not_needed_without_ssl():
    with pytest.raises(NotNeeded):
        validate.check_https_cert(Context(use_ssl=False))


def test_check_https_cert_missing_files(tmp_path):
    with pytest.raises(Fail, match='not found'):
        validate.check_https_cert(Context(use_ssl=True, ssl_cert=str(tmp_path / 'c.pem'),
                                          ssl_key=str(tmp_path / 'k.pem')))


def test_check_http_port_in_use():
    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        sock.listen()
        with pytest.raises(Fail, match="can't be used"):
            validate.check_http_port(Context(host='127.0.0.1', port=sock.getsockname()[1]))


# ------------------------------------------------- command line ----------------------------------------------

class TestModesArg:
    def test_comma_separated(self):
        assert main.modes_arg('file, http,file') == ('file', 'http')

    @pytest.mark.parametrize('value', ['nope', 'file,nope', ','])
    def test_unknown(self, value):
        with pytest.raises(argparse.ArgumentTypeError):
            main.modes_arg(value)


@pytest.mark.parametrize('argv, error', [
    (['file', '--quick'], 'only be used with validate'),
    (['http', '--only', 'file'], 'only be used with validate'),
    (['validate', '--all-profiles', '--profile', 'x'], 'either --profile or --all-profiles'),
])
def test_main_rejects_validate_options(monkeypatch, capsys, argv, error):
    monkeypatch.setattr(sys, 'argv', ['main.py', *argv])
    with pytest.raises(SystemExit):
        main.main()
    assert error in capsys.readouterr().err


# ------------------------------------------------- end to end ------------------------------------------------

def _privacy_models_present() -> bool:
    from app.config import load_pipeline_spec
    spec_path = constants.install_dir / 'configs' / 'privacy' / 'pipeline.yml'
    if not spec_path.is_file():
        return False
    return all(Path(m['path']).is_file() for m in load_pipeline_spec(spec_path).get('models', []))


@pytest.mark.skipif(not _privacy_models_present(), reason="the models of the privacy profile are not installed")
def test_validate_privacy_profile_end_to_end():
    """Run the real checks (profile, pipeline, images) in a separate process, as a user would."""
    proc = subprocess.run([sys.executable, str(constants.install_dir / 'main.py'), 'validate', '--profile', 'privacy',
                           '--only', 'file', '--quick'], cwd=constants.install_dir, capture_output=True, text=True,
                          timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert 'File: images   should work' in proc.stderr


# ------------------------------------------------- colors ----------------------------------------------------

def _log_result_output(colorize: bool) -> str:
    from loguru import logger
    out = []
    handler = logger.add(out.append, format="{message}", colorize=colorize)
    try:
        validate._log_result(Check('a', 'A', ok), Result(Status.FAIL, "needs <b> x"))
    finally:
        logger.remove(handler)
    return ''.join(out)


def test_log_result_without_colors_is_plain_text():
    assert _log_result_output(False) == "[FAIL] A: needs <b> x\n"


def test_log_result_colors_the_status_red():
    text = _log_result_output(True)
    assert "\x1b[31m" in text and "FAIL" in text and "needs <b> x" in text
