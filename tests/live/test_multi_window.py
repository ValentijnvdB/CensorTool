"""Censoring several sources (overlay mode windows) in one live loop, and finding a started program's new windows."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import CONFIG
from app.live import launch, quick
from app.live.input_devices.generic import InputDevice
from app.live.output_devices.generic import OutputDevice
from app.live.windowing import WindowInfo


class TestTurnScheduler:

    def run(self, keys, priority, turns, changed=None):
        """Let sources take turns; changed(key) says whether a source has a new frame."""
        scheduler = quick.TurnScheduler()
        taken = []
        for _ in range(turns):
            for key in scheduler.order(keys, priority):
                if changed is None or changed(key):
                    scheduler.took_turn(key, priority)
                    taken.append(key)
                    break
        return taken

    def test_sources_take_turns(self):
        taken = self.run(['a', 'b', 'c'], None, 9)
        assert {k: taken.count(k) for k in 'abc'} == {'a': 3, 'b': 3, 'c': 3}

    def test_priority_source_gets_more_turns(self):
        taken = self.run(['a', 'b'], 'b', 40)
        assert taken.count('b') == pytest.approx(3 * taken.count('a'), abs=2)
        assert 'a' in taken[:5]  # but the others are not starved

    def test_priority_source_goes_first(self):
        assert quick.TurnScheduler().order(['a', 'b', 'c'], 'c')[0] == 'c'

    def test_unchanged_source_does_not_save_up_turns(self):
        changing = {'a', 'b'}
        scheduler = quick.TurnScheduler()
        for _ in range(20):  # c has nothing to censor for a while
            key = next(k for k in scheduler.order(['a', 'b', 'c'], None) if k in changing)
            scheduler.took_turn(key, None)
        taken = []
        for _ in range(9):  # then it changes as well: it gets its share, not a burst
            key = scheduler.order(['a', 'b', 'c'], None)[0]
            scheduler.took_turn(key, None)
            taken.append(key)
        assert taken.count('c') == 3

    def test_new_source_joins_the_rotation(self):
        scheduler = quick.TurnScheduler()
        for _ in range(10):
            scheduler.took_turn('a', None)
        assert scheduler.order(['a', 'b'], None)[0] == 'b'


class CountingDevice(InputDevice):
    """Frames whose first channel is value (tells the source) and second a counter (every frame differs)."""

    def __init__(self, value: int):
        self.value = value
        self.count = 0
        self.closed_threads = 0

    def get_frame(self):
        time.sleep(0.002)
        self.count += 1
        frame = np.zeros((2, 2, 3), np.uint8)
        frame[..., 0], frame[..., 1] = self.value, self.count % 256
        return time.monotonic(), frame

    def close_thread(self):
        self.closed_threads += 1

    def close(self):
        pass


class DictSources(quick.Sources):

    def __init__(self, devices):
        self.lock = threading.Lock()
        self.devs = dict(devices)

    def devices(self):
        with self.lock:
            return dict(self.devs)


class RecordingOutput(OutputDevice):

    def __init__(self):
        self.frames = []  # (source, frame)

    def push_frame(self, frame, timestamp=None, source=None):
        self.frames.append((source, frame))

    def close(self):
        pass


class FakeHandle:

    def __init__(self, job):
        self.job = job

    def done(self):
        return True

    def wait(self):
        return self.job


class FakePipeline:

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def submit(self, job):
        job.success, job.result = True, {'image': job.image}
        return FakeHandle(job)


@pytest.fixture
def fake_censoring(monkeypatch):
    monkeypatch.setattr(quick, 'cc', SimpleNamespace(Job=lambda **kw: SimpleNamespace(**kw),
                                                     wait_any=lambda handles: None))
    monkeypatch.setattr(quick, 'make_image_pipeline', lambda mode: FakePipeline())
    monkeypatch.setattr(quick, 'read_censor_config_from_file_w_hash', lambda *a, **kw: (True, object(), 'h'))
    monkeypatch.setattr(quick, 'config_file_hash', lambda: 'h')
    monkeypatch.setattr(quick, 'reload_config_if_changed', lambda h: (False, 'h'))
    monkeypatch.setattr(CONFIG.live, 'frames_in_flight', 2)
    monkeypatch.setattr(CONFIG.live, 'show_fps', False)


def run_loop(sources, output, until, timeout=5.0):
    stop = threading.Event()
    thread = threading.Thread(target=quick.live_censor_loop, args=(stop, sources, output, None), daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while not until() and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    thread.join(timeout=5)
    assert not thread.is_alive()


class TestLoopWithSeveralSources:

    def test_frames_are_pushed_with_their_source(self, fake_censoring):
        devices = {'a': CountingDevice(0), 'b': CountingDevice(100)}
        output = RecordingOutput()
        run_loop(DictSources(devices), output,
                 until=lambda: {s for s, _ in output.frames} == {'a', 'b'} and len(output.frames) > 20)
        assert {s for s, _ in output.frames} == {'a', 'b'}
        for source, frame in output.frames:  # every frame was pushed for the source it came from
            assert frame[0, 0, 0] == devices[source].value

    def test_removed_source_is_no_longer_captured(self, fake_censoring):
        devices = {'a': CountingDevice(0), 'b': CountingDevice(100)}
        sources = DictSources(devices)
        output = RecordingOutput()
        stop = threading.Event()
        thread = threading.Thread(target=quick.live_censor_loop, args=(stop, sources, output, None), daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while {s for s, _ in output.frames} != {'a', 'b'} and time.monotonic() < deadline:
                time.sleep(0.01)
            with sources.lock:
                del sources.devs['b']
            deadline = time.monotonic() + 5
            while devices['b'].closed_threads == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert devices['b'].closed_threads == 1  # its capture thread stopped
            n = len(output.frames)
            time.sleep(0.1)
            assert all(s == 'a' for s, _ in output.frames[n + 2:])  # (up to 2 were still in flight)
        finally:
            stop.set()
            thread.join(timeout=5)

    def test_no_sources_waits(self, fake_censoring):
        output = RecordingOutput()
        run_loop(DictSources({}), output, until=lambda: False, timeout=0.2)
        assert output.frames == []

    def test_single_device_uses_the_single_source_key(self, fake_censoring):
        output = RecordingOutput()
        run_loop(CountingDevice(0), output, until=lambda: len(output.frames) > 3)
        assert {s for s, _ in output.frames} == {quick.SingleSource.KEY}


class FakeDisplay:

    def __init__(self, infos):
        self.infos = {i.window: i for i in infos}

    def window_list(self, stacking=False):
        return list(self.infos)

    def window_info(self, win):
        return self.infos[win]


class TestProgramWindowWatcher:

    def watcher(self, how='pid', known=(), first_window=1):
        found = []
        proc = SimpleNamespace(pid=100)
        watcher = launch.ProgramWindowWatcher(proc, ['celluloid'], how, first_window, set(known), found.append)
        return watcher, found

    def check(self, watcher, infos, monkeypatch, parents=None):
        monkeypatch.setattr(launch, 'parent_map', lambda: parents or {100: 1, 101: 100})
        watcher._check(FakeDisplay(infos), {100}, {}, time.monotonic())

    def info(self, window, pid=None, app_ids=()):
        return WindowInfo(window=window, pid=pid, app_ids=app_ids, normal=True)

    def test_new_window_of_the_process_tree(self, monkeypatch):
        watcher, found = self.watcher(known={5})
        self.check(watcher, [self.info(1, 100), self.info(5, 101), self.info(7, 101), self.info(8, 500)],
                   monkeypatch)
        assert found == [7]  # 1 is the first window, 5 existed before, 8 is someone else's

    def test_each_window_is_reported_once(self, monkeypatch):
        watcher, found = self.watcher()
        infos = [self.info(7, 100)]
        self.check(watcher, infos, monkeypatch)
        self.check(watcher, infos, monkeypatch)
        assert found == [7]

    def test_class_match_only_when_the_first_window_was_found_by_class(self, monkeypatch):
        other = self.info(7, 500, ('celluloid', 'Celluloid'))
        by_pid, found_by_pid = self.watcher(how='pid')
        self.check(by_pid, [other], monkeypatch)
        by_class, found_by_class = self.watcher(how='class-existing')
        self.check(by_class, [other], monkeypatch)
        assert found_by_pid == [] and found_by_class == [7]
