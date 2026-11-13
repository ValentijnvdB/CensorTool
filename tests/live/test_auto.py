"""Auto mode of overlay mode: rules, saving programs to config.yml and which windows the watcher reports."""
import os

import pytest
import yaml

from app import config
from app.live import auto
from app.live.auto import AutoRule
from app.live.x11 import WindowInfo


class TestParseRules:

    def test_programs_and_titles(self):
        rules = auto.parse_rules(['celluloid', {'program': 'firefox', 'title': 'Private'}, {'title': 'secret'}])
        assert rules == [AutoRule('celluloid'), AutoRule('firefox', 'Private'), AutoRule(None, 'secret')]

    def test_empty(self):
        assert auto.parse_rules([]) == [] and auto.parse_rules(None) == []

    @pytest.mark.parametrize('value', ['celluloid', [''], [3], [{'program': 'x', 'size': 2}], [{}],
                                       [{'title': '('}], [{'program': ''}]])
    def test_invalid(self, value):
        with pytest.raises(ValueError):
            auto.parse_rules(value)


class TestAutoRule:

    def test_program(self):
        rule = AutoRule('celluloid')
        assert rule.matches(('io.github.celluloid_player.Celluloid', 'Io.github.celluloid_player.Celluloid'), '')
        assert not rule.matches(('Navigator', 'firefox'), 'Celluloid fan page')

    def test_title_is_a_case_insensitive_pattern(self):
        rule = AutoRule('firefox', r'private browsing$')
        assert rule.matches(('Navigator', 'firefox'), 'News — Mozilla Firefox Private Browsing')
        assert not rule.matches(('Navigator', 'firefox'), 'Private Browsing help — Mozilla Firefox')
        assert not rule.matches(('celluloid', 'Celluloid'), 'Private Browsing')

    def test_title_only(self):
        assert AutoRule(None, 'secret').matches(('anything', 'Anything'), 'my SECRET file')


@pytest.mark.parametrize('wm_class, name', [
    (('io.github.celluloid_player.Celluloid', 'Io.github.celluloid_player.Celluloid'), 'celluloid'),
    (('Navigator', 'firefox'), 'firefox'),
    (('gl', 'mpv'), 'mpv'),
    ((), None),
])
def test_program_name_of(wm_class, name):
    assert auto.program_name_of(wm_class) == name


class TestAddOverlayAutoProgram:

    @pytest.fixture
    def config_file(self, tmp_path, monkeypatch):
        path = tmp_path / 'config.yml'
        monkeypatch.setattr(config, '_config_path', path)
        return path

    def test_flow_list_is_extended(self, config_file):
        config_file.write_text("a: 1\n# about auto mode\noverlay_auto_programs: [mpv]  # note\nb: 2\n")
        config.add_overlay_auto_program('celluloid')
        text = config_file.read_text()
        assert yaml.safe_load(text) == {'a': 1, 'overlay_auto_programs': ['mpv', 'celluloid'], 'b': 2}
        assert '# about auto mode' in text

    def test_block_list_is_replaced(self, config_file):
        config_file.write_text("overlay_auto_programs:\n  - mpv\n  - {program: firefox, title: Private}\nb: 2\n")
        config.add_overlay_auto_program('celluloid')
        assert yaml.safe_load(config_file.read_text()) == {
            'overlay_auto_programs': ['mpv', {'program': 'firefox', 'title': 'Private'}, 'celluloid'], 'b': 2}

    def test_missing_key_is_added(self, config_file):
        config_file.write_text("a: 1\n")
        config.add_overlay_auto_program('celluloid')
        assert yaml.safe_load(config_file.read_text()) == {'a': 1, 'overlay_auto_programs': ['celluloid']}

    def test_listed_already(self, config_file):
        config_file.write_text("overlay_auto_programs: [Celluloid]\n")
        config.add_overlay_auto_program('celluloid')
        assert config_file.read_text() == "overlay_auto_programs: [Celluloid]\n"

    def test_the_tracked_profiles_accept_it(self, tmp_path, monkeypatch):
        for profile in ('nudity', 'privacy'):
            path = tmp_path / f'{profile}.yml'
            path.write_text((config.constants.profiles_path / profile / 'config.yml').read_text())
            monkeypatch.setattr(config, '_config_path', path)
            config.add_overlay_auto_program('celluloid')
            assert config.load_config(path).overlay_auto_programs == ['celluloid']


class FakeConnection:
    """Windows as (WindowInfo, title, managed, type)."""

    def __init__(self, windows):
        self.windows = {w[0].window: w for w in windows}

    def cardinals(self, win, name, n):
        assert name == 'WM_STATE'
        return [1] if self.windows[win][2] else None

    def window_info(self, win):
        return self.windows[win][0]

    def title(self, win):
        return self.windows[win][1]

    def window_type(self, win):
        return self.windows[win][3]


def window(win, wm_class=('celluloid', 'Celluloid'), title='', managed=True, type_='NORMAL', pid=500):
    return WindowInfo(window=win, pid=pid, wm_class=wm_class, normal=type_ == 'NORMAL'), title, managed, type_


class TestAutoWatcherCheck:

    def check(self, windows, rules):
        found = []
        watcher = auto.AutoWatcher(lambda: rules, found.append)
        conn = FakeConnection(windows)
        for win in conn.windows:
            watcher._check(conn, win, rules)
        return found, watcher, conn

    def test_matching_windows_are_reported(self):
        found, _, _ = self.check([window(1), window(2, ('Navigator', 'firefox')), window(3, type_='DIALOG')],
                                 [AutoRule('celluloid')])
        assert found == [1, 3]

    def test_each_window_once(self):
        found, watcher, conn = self.check([window(1)], [AutoRule('celluloid')])
        watcher._check(conn, 1, [AutoRule('celluloid')])
        assert found == [1]

    def test_unmanaged_panels_and_own_windows_are_skipped(self):
        found, _, _ = self.check([window(1, managed=False), window(2, type_='DOCK'), window(3, pid=os.getpid())],
                                 [AutoRule('celluloid')])
        assert found == []

    def test_title_rule(self):
        windows = [window(1, ('Navigator', 'firefox'), 'Page — Private Browsing'),
                   window(2, ('Navigator', 'firefox'), 'Page')]
        found, _, _ = self.check(windows, [AutoRule('firefox', 'private browsing')])
        assert found == [1]

    def test_no_rules(self):
        assert self.check([window(1)], [])[0] == []
