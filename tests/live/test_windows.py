"""The Windows window system: the parts that can be tested on any platform (it has not run on Windows yet)."""
import subprocess

import numpy as np
import pytest

from app.live import launch, windowing
from app.live.windowing import windows


class TestIsAppWindowStyle:

    def test_normal_window(self):
        assert windows.is_app_window_style(0, owned=False, cloaked=0, width=800, height=600)

    def test_dialog(self):
        assert windows.is_app_window_style(0, owned=True, cloaked=0, width=400, height=300)

    def test_tool_window_unless_it_wants_a_taskbar_button(self):
        assert not windows.is_app_window_style(windows.WS_EX_TOOLWINDOW, False, 0, 800, 600)
        assert windows.is_app_window_style(windows.WS_EX_TOOLWINDOW | windows.WS_EX_APPWINDOW, False, 0, 800, 600)

    def test_cloaked_by_the_app_is_hidden_but_another_desktop_is_not(self):
        assert not windows.is_app_window_style(0, False, cloaked=0x1, width=800, height=600)
        assert windows.is_app_window_style(0, False, cloaked=windows.DWM_CLOAKED_SHELL, width=800, height=600)

    def test_empty_window(self):
        assert not windows.is_app_window_style(0, False, 0, width=0, height=0)


@pytest.mark.parametrize('path, stem', [
    (r'C:\Program Files\VideoLAN\VLC\vlc.exe', 'vlc'),
    (r'C:\Program Files\Mozilla Firefox\Firefox.EXE', 'firefox'),
    ('C:/Tools/mpv.com', 'mpv'),
])
def test_exe_stem(path, stem):
    assert windows.exe_stem(path) == stem


class TestNamesOfWindowsPrograms:

    def test_command_with_exe_matches_the_process_name(self):
        names = launch.program_names([r'C:\Program Files\VideoLAN\VLC\vlc.exe', 'clip.mp4'])
        assert launch.class_matches(names, ('vlc',))

    def test_rule_with_or_without_exe(self):
        assert launch.class_matches({'vlc.exe'}, ('vlc',))
        assert launch.class_matches({'vlc'}, ('vlc',))
        assert not launch.class_matches({'vlc.exe'}, ('notepad',))


def test_crop_is_a_clipped_copy():
    image = np.arange(6 * 8 * 4, dtype=np.uint8).reshape(6, 8, 4)
    part = windows.crop(image, (2, 1), (4, 3))
    assert part.shape == (3, 4, 4)
    assert (part == image[1:4, 2:6]).all()
    assert part.base is None or part.base is not image
    assert windows.crop(image, (-3, 0), (100, 100)).shape == (6, 8, 4)


def test_session_problem_off_windows():
    assert 'Windows' in windows.session_problem()


def test_get_selects_the_windows_window_system(monkeypatch):
    monkeypatch.setattr(windowing.sys, 'platform', 'win32')
    monkeypatch.setattr(windows, 'session_problem', lambda: None)
    monkeypatch.setattr(windowing, '_system', None)
    assert isinstance(windowing.get(), windows.WindowsWindowSystem)


class FakeGrabber:
    def __init__(self, hwnd, fail=False):
        self.window = hwnd
        self.area, self.visible, self.failures = (0, 0, 4, 3), True, 0
        self.closed = False

    def refresh(self):
        return True

    def close(self):
        self.closed = True


class TestWindowsGrabberFallback:

    @pytest.fixture(autouse=True)
    def reset(self, monkeypatch):
        monkeypatch.setattr(windows.WindowsGrabber, '_wgc_unavailable', None)

    def test_print_window_when_wgc_is_unavailable(self, monkeypatch):
        def no_wgc(_hwnd):
            raise windows.WgcUnavailable("not installed")
        monkeypatch.setattr(windows, 'WgcGrabber', no_wgc)
        monkeypatch.setattr(windows, 'PrintWindowGrabber', FakeGrabber)
        grabber = windows.WindowsGrabber(5)
        assert isinstance(grabber._inner, FakeGrabber)
        assert windows.WindowsGrabber._wgc_unavailable == "not installed"

    def test_switches_to_print_window_when_wgc_keeps_failing(self, monkeypatch):
        class FailingWgc(FakeGrabber):
            def grab(self):
                self.failures += 1
                return None

        class Printing(FakeGrabber):
            def grab(self):
                return np.zeros((3, 4, 4), np.uint8)

        monkeypatch.setattr(windows, 'WgcGrabber', FailingWgc)
        monkeypatch.setattr(windows, 'PrintWindowGrabber', Printing)
        grabber = windows.WindowsGrabber(5)
        first = grabber._inner
        for _ in range(windows.WindowsGrabber.MAX_WGC_FAILURES):
            assert grabber.grab() is None
        assert first.closed and isinstance(grabber._inner, Printing)
        assert grabber.grab() is not None


def test_started_program_detaches_on_this_platform(monkeypatch):
    calls = {}

    def fake_popen(command, **kwargs):
        calls.update(kwargs)
        raise FileNotFoundError

    monkeypatch.setattr(launch.subprocess, 'Popen', fake_popen)
    monkeypatch.setattr(launch.sys, 'platform', 'win32')
    monkeypatch.setattr(launch.subprocess, 'CREATE_NEW_PROCESS_GROUP', 0x200, raising=False)
    with pytest.raises(launch.LaunchError):
        launch.start_program(['vlc.exe'])
    assert calls.get('creationflags') == 0x200 and 'start_new_session' not in calls

    def test_a_window_wgc_cant_capture_does_not_turn_it_off(self, monkeypatch):
        def cant(_hwnd):
            raise windows.WinError("this window can't be captured")
        monkeypatch.setattr(windows, 'WgcGrabber', cant)
        monkeypatch.setattr(windows, 'PrintWindowGrabber', FakeGrabber)
        assert isinstance(windows.WindowsGrabber(5)._inner, FakeGrabber)
        assert windows.WindowsGrabber._wgc_unavailable is None


class FakeWindowsCapture:
    """Stands in for windows_capture.WindowsCapture: delivers one frame when started."""
    instances = []
    refuse_border = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.handlers = {}
        FakeWindowsCapture.instances.append(self)

    def event(self, handler):
        self.handlers[handler.__name__] = handler
        return handler

    def start_free_threaded(self):
        if self.refuse_border and self.kwargs['draw_border'] is False:
            raise RuntimeError("border can't be hidden")
        buffer = np.full((4, 6, 4), 7, np.uint8)
        frame = type('Frame', (), {'frame_buffer': buffer})()
        self.handlers['on_frame_arrived'](frame, None)
        buffer[:] = 0  # the real buffer is reused after the call: the grabber must have copied it
        return type('Control', (), {'stop': lambda self: None})()


class TestWgcGrabber:

    @pytest.fixture
    def wgc(self, monkeypatch):
        import sys
        import types
        module = types.ModuleType('windows_capture')
        module.WindowsCapture = FakeWindowsCapture
        monkeypatch.setitem(sys.modules, 'windows_capture', module)
        monkeypatch.setattr(windows, 'api', lambda: None)
        FakeWindowsCapture.instances = []
        FakeWindowsCapture.refuse_border = False
        return FakeWindowsCapture

    def test_captures_by_handle_without_cursor_or_border(self, wgc):
        grabber = windows.WgcGrabber(0x1234)
        assert wgc.instances[0].kwargs == {'cursor_capture': False, 'draw_border': False, 'window_hwnd': 0x1234}
        grabber.visible, grabber.area = True, (0, 0, 5, 3)
        frame = grabber.grab()
        assert frame.shape == (3, 5, 4) and (frame == 7).all()

    def test_border_that_cant_be_hidden_is_kept(self, wgc):
        wgc.refuse_border = True
        windows.WgcGrabber(1)
        assert [c.kwargs['draw_border'] for c in wgc.instances] == [False, None]

    def test_package_without_window_hwnd(self, wgc, monkeypatch):
        import sys

        class OldCapture:  # versions before window_hwnd
            def __init__(self, cursor_capture=None, draw_border=None, monitor_index=None, window_name=None):
                pass

        monkeypatch.setattr(sys.modules['windows_capture'], 'WindowsCapture', OldCapture)
        with pytest.raises(windows.WgcUnavailable, match='too old'):
            windows.WgcGrabber(1)
