"""Overlay mode logic that runs without an X server: geometry, window picking and the capture device."""
import argparse
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import main
from app.live import windowing
from app.live.windowing import x11
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
        monkeypatch.setattr(windowing.sys, 'platform', 'darwin')
        assert 'darwin' in windowing.session_problem()
        monkeypatch.setattr(windowing, '_system', None)
        with pytest.raises(windowing.WindowSystemError, match='darwin'):
            windowing.get()

    def test_no_display(self, monkeypatch):
        monkeypatch.setattr(x11.sys, 'platform', 'linux')
        monkeypatch.delenv('XDG_SESSION_TYPE', raising=False)
        monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
        monkeypatch.delenv('DISPLAY', raising=False)
        assert 'DISPLAY' in x11.session_problem()


class FakeGrabber:
    """Stands in for a Grabber. Frames are filled with the target id, so tests can tell them apart."""
    instances: list['FakeGrabber'] = []
    visible = True
    fail = False

    def __init__(self, client: int):
        self.client = self.window = client
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
    monkeypatch.setattr(windowing, 'get', lambda: SimpleNamespace(grabber=FakeGrabber))
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
        with pytest.raises(windowing.WindowSystemError):
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


class TestDelayArgument:

    @pytest.mark.parametrize('value, expected', [('0', 0.0), ('2.5', 2.5)])
    def test_valid(self, value, expected):
        assert main.delay_arg(value) == expected

    @pytest.mark.parametrize('value', ['-1', 'soon', 'nan', 'inf'])
    def test_invalid(self, value):
        with pytest.raises(argparse.ArgumentTypeError):
            main.delay_arg(value)


class FakeDisplay:
    def __init__(self, active, app_window=True, panel=False):
        self.active, self.app_window, self.panel = active, app_window, panel

    def active_window(self):
        return self.active

    def application_window(self, win):
        return win

    def is_app_window(self, win):
        return self.app_window

    def window_info(self, win):
        return SimpleNamespace(panel=self.panel)

    def title(self, win):
        return 'Video'

    def close(self):
        pass


class TestFocusedTarget:

    @pytest.fixture
    def use_display(self, monkeypatch):
        from app.live import overlay

        def use(display):
            monkeypatch.setattr(windowing, 'get', lambda: SimpleNamespace(open_display=lambda: display))
            return overlay
        return use

    def test_focused_window(self, use_display):
        assert use_display(FakeDisplay(0x3a00007))._focused_target(0) == 0x3a00007

    def test_waits_first(self, use_display, monkeypatch):
        overlay = use_display(FakeDisplay(0x3a00007))
        slept = []
        monkeypatch.setattr(overlay.time, 'sleep', slept.append)
        overlay._focused_target(1.5)
        assert slept == [1.5]

    @pytest.mark.parametrize('display', [FakeDisplay(None), FakeDisplay(0x10, panel=True),
                                         FakeDisplay(0x10, app_window=False)])
    def test_no_application_window_focused(self, use_display, display):
        with pytest.raises(ValueError):
            use_display(display)._focused_target(0)

    def test_not_with_window_ids_or_command(self, use_display):
        overlay = use_display(FakeDisplay(0x10))
        with pytest.raises(ValueError):
            overlay.start_overlay_censor(window_ids=[0x10], focused=0)
        with pytest.raises(ValueError):
            overlay.start_overlay_censor(command=['mpv'], focused=0)


class TestFocusedCommandLine:

    @pytest.mark.parametrize('argv', [
        ['file', '--focused'],
        ['overlay', '--focused', '--window', '0x10'],
        ['overlay', '--focused', '--', 'mpv'],
    ])
    def test_refused(self, argv, monkeypatch):
        monkeypatch.setattr('sys.argv', ['main.py', *argv])
        with pytest.raises(SystemExit) as e:
            main.main()
        assert e.value.code == 2
