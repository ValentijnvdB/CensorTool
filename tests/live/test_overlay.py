"""Overlay mode logic that runs without an X server: geometry, window picking and the capture device."""
import argparse
import threading

import numpy as np
import pytest

import main
from app.live import x11
from app.live.input_devices import window_capture_device
from app.live.input_devices.window_capture_device import WindowCaptureDevice


class TestVisibleArea:

    def test_without_shadow_margins(self):
        assert x11.visible_area(10, 20, 300, 200, None) == ((10, 20, 300, 200), (0, 0))

    def test_shadow_margins_are_removed(self):
        # _GTK_FRAME_EXTENTS is left, right, top, bottom
        assert x11.visible_area(10, 20, 300, 200, [5, 7, 3, 9]) == ((15, 23, 288, 188), (5, 3))

    def test_negative_margins_are_ignored(self):
        assert x11.visible_area(0, 0, 100, 100, [-4, 0, 0, 0]) == ((0, 0, 100, 100), (0, 0))

    def test_margins_larger_than_the_window_use_the_whole_window(self):
        assert x11.visible_area(0, 0, 100, 100, [60, 60, 0, 0]) == ((0, 0, 100, 100), (0, 0))


class TestWindowAt:
    STACK = [(1, (0, 0, 1000, 1000)),  # bottom
             (2, (100, 100, 200, 200)),
             (3, (150, 150, 200, 200))]  # top

    def test_topmost_window_wins(self):
        assert x11.window_at(self.STACK, 160, 160, exclude=set()) == 3

    def test_excluded_window_is_skipped(self):
        assert x11.window_at(self.STACK, 160, 160, exclude={3}) == 2

    def test_right_and_bottom_edges_are_outside(self):
        assert x11.window_at(self.STACK, 299, 120, exclude=set()) == 2
        assert x11.window_at(self.STACK, 300, 120, exclude=set()) == 1
        assert x11.window_at(self.STACK, 349, 349, exclude=set()) == 3
        assert x11.window_at(self.STACK, 350, 349, exclude=set()) == 1

    def test_nothing_there(self):
        assert x11.window_at(self.STACK, 2000, 10, exclude=set()) is None


class TestSessionProblem:

    def test_wayland_is_refused(self, monkeypatch):
        monkeypatch.setattr(x11.sys, 'platform', 'linux')
        monkeypatch.setenv('XDG_SESSION_TYPE', 'wayland')
        assert 'Wayland' in x11.session_problem()

    def test_other_platforms_are_refused(self, monkeypatch):
        monkeypatch.setattr(x11.sys, 'platform', 'win32')
        assert 'X11' in x11.session_problem()

    def test_no_display(self, monkeypatch):
        monkeypatch.setattr(x11.sys, 'platform', 'linux')
        monkeypatch.delenv('XDG_SESSION_TYPE', raising=False)
        monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
        monkeypatch.delenv('DISPLAY', raising=False)
        assert 'DISPLAY' in x11.session_problem()


class FakeGrabber:
    """Stands in for x11.WindowGrabber. Frames are filled with the target id, so tests can tell them apart."""
    instances: list['FakeGrabber'] = []
    visible = True
    fail = False

    def __init__(self, client: int):
        self.client = client
        self.area = (0, 0, 4, 3)
        self.failures = 0
        self.closed = False
        FakeGrabber.instances.append(self)

    def refresh(self):
        self.visible = FakeGrabber.visible
        return True

    def grab(self):
        if not FakeGrabber.visible or FakeGrabber.fail:  # like the real one, capturing a hidden window fails
            self.failures += 1
            return None
        return np.full((3, 4, 4), self.client % 256, np.uint8)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_grabber(monkeypatch):
    FakeGrabber.instances = []
    FakeGrabber.visible = True
    FakeGrabber.fail = False
    monkeypatch.setattr(x11, 'WindowGrabber', FakeGrabber)
    monkeypatch.setattr(WindowCaptureDevice, 'HIDDEN_POLL_INTERVAL', 0)
    monkeypatch.setattr(window_capture_device.time, 'sleep', lambda _s: None)
    return FakeGrabber


class TestWindowCaptureDevice:

    def test_frames_are_bgr(self, fake_grabber):
        device = WindowCaptureDevice(7)
        _, frame = device.get_frame()
        assert frame.shape == (3, 4, 3)
        assert (frame == 7).all()

    def test_hidden_window_repeats_the_last_frame(self, fake_grabber):
        device = WindowCaptureDevice(7)
        _, first = device.get_frame()
        fake_grabber.visible = False
        _, again = device.get_frame()
        assert again is first

    def test_hidden_from_the_start_gives_a_black_frame(self, fake_grabber):
        fake_grabber.visible = False
        device = WindowCaptureDevice(7)
        _, frame = device.get_frame()
        assert frame.shape == (3, 4, 3)
        assert not frame.any()

    def test_capture_that_keeps_failing_raises(self, fake_grabber):
        fake_grabber.fail = True
        device = WindowCaptureDevice(7)
        with pytest.raises(x11.X11Error):
            device.get_frame()

    def test_each_thread_has_its_own_grabber(self, fake_grabber):
        device = WindowCaptureDevice(7)
        device.get_frame()
        thread = threading.Thread(target=device.get_frame)
        thread.start()
        thread.join()
        assert len(fake_grabber.instances) == 2
        device.close()
        assert all(g.closed for g in fake_grabber.instances)


class TestWindowIdArgument:

    @pytest.mark.parametrize('value, expected', [('0x3a00007', 0x3a00007), ('12345', 12345)])
    def test_hex_and_decimal(self, value, expected):
        assert main.window_id_arg(value) == expected

    def test_invalid(self):
        with pytest.raises(argparse.ArgumentTypeError):
            main.window_id_arg('firefox')
