"""
Check the window system overlay mode uses, step by step, on this computer:

    python -m app.live.windowing.selfcheck            # checks the focused window (the terminal)
    python -m app.live.windowing.selfcheck --pick     # click the window to check

It lists the windows, captures one (and saves the capture as a PNG), follows it while you move it, watches for new
windows, and shows a test overlay above it. Meant for a platform overlay mode has not been tried on yet (Windows):
paste the output when something fails.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path


def step(title: str):
    print(f"\n=== {title}")


def run(name: str, fn) -> bool:
    try:
        fn()
        return True
    except Exception:
        print(f"FAILED: {name}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pick', action='store_true', help='click the window to check (default: the focused one)')
    parser.add_argument('--seconds', type=float, default=5, help='how long to follow / watch (default 5)')
    parser.add_argument('--no-overlay', action='store_true', help='skip the test overlay (needs PySide6)')
    args = parser.parse_args()

    from app.live import windowing

    step("session")
    print(f"platform: {sys.platform}, python {sys.version.split()[0]}")
    problem = windowing.session_problem()
    if problem:
        print(f"Overlay mode can't run here: {problem}")
        return 1
    ws = windowing.get()
    print(f"window system: {ws.name}")

    qt = None
    if not args.no_overlay:
        try:
            import os
            from PySide6.QtWidgets import QApplication
            os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '0'
            ws.prepare_qt_environment(os.environ)
            qt = QApplication.instance() or QApplication(['CensorTool selfcheck'])
            ws.qt_started()  # (also makes the process DPI aware on Windows, as overlay mode does)
        except ImportError:
            print("PySide6 is not installed: skipping the test overlay.")

    display = ws.open_display()
    step("windows (bottom to top, last 15)")
    for win in display.window_list(stacking=True)[-15:]:
        info = display.window_info(win)
        print(f"{win:#x} pid={info.pid} app_ids={info.app_ids} normal={info.normal} panel={info.panel} "
              f"area={display.visible_area(win)} title={display.title(win)[:50]!r}")

    step("the window to check")
    if args.pick:
        print("Click a window (right click or Escape cancels)...")
        target = ws.pick_window()
        if target is None:
            print("No window picked.")
            return 1
    else:
        focus = ws.focus_tracker()
        time.sleep(0.5)
        target = focus.active
        focus.stop()
        if target is None:
            print("No focused window found; use --pick.")
            return 1
    print(f"{target:#x}: {display.title(target)!r}, {display.window_info(target)}, "
          f"area {display.visible_area(target)}")

    def capture():
        grabber = ws.grabber(target)
        try:
            print(f"grabber: {type(grabber).__name__}" + (f" using {type(grabber._inner).__name__}"
                                                          if hasattr(grabber, '_inner') else ''))
            frame, times = None, []
            for _ in range(30):
                start = time.perf_counter()
                grabber.refresh()
                image = grabber.grab()
                times.append(time.perf_counter() - start)
                frame = image if image is not None else frame
            print(f"visible={grabber.visible} area={grabber.area} failures={grabber.failures}")
            if frame is None:
                print("FAILED: no frame captured")
                return
            print(f"frame {frame.shape} {frame.dtype}, {sum(times) / len(times) * 1000:.1f} ms per refresh+grab, "
                  f"mean brightness {frame[..., :3].mean():.0f} (0 = black: capturing doesn't work for it)")
            import cv2
            path = Path(tempfile.gettempdir()) / 'censortool-selfcheck.png'
            cv2.imwrite(str(path), frame[..., :3])
            print(f"saved the last capture to {path}")
        finally:
            grabber.close()

    step("capturing")
    run("capturing", capture)

    def follow():
        changes = []
        holder = []
        tracker = ws.tracker(target, on_change=lambda: changes.append(holder[0].state if holder else None))
        holder.append(tracker)
        changes.clear()  # the first state, from before it started
        print(f"state: {tracker.state}")
        print(f"Move, resize or minimise the window within {args.seconds:.0f} s...")
        time.sleep(args.seconds)
        tracker.stop()
        print(f"{len(changes)} changes" + (f", last: {changes[-1]}" if changes else
                                          " (none: following doesn't work, or you didn't move it)"))

    step("following the window")
    run("following", follow)

    def watch():
        monitor = ws.monitor(follow_titles=True)
        try:
            before = set(monitor.display.window_list())
            print(f"Open a new window (or change a title) within {args.seconds:.0f} s...")
            deadline = time.monotonic() + args.seconds
            events = 0
            while time.monotonic() < deadline:
                changes = monitor.wait(0.1)
                if changes.list_changed or changes.windows:
                    events += 1
                    print(f"  list changed={changes.list_changed} windows={[hex(w) for w in changes.windows]}")
            new = set(monitor.display.window_list()) - before
            print(f"{events} change(s); new windows: {[hex(w) for w in new]}")
        finally:
            monitor.close()

    step("watching for new windows")
    run("watching", watch)

    if qt is not None:
        def overlay():
            from PySide6.QtCore import Qt, QTimer
            from PySide6.QtGui import QColor, QPainter
            from PySide6.QtWidgets import QWidget

            class Box(QWidget):
                def paintEvent(self, _event):
                    painter = QPainter(self)
                    painter.fillRect(self.rect(), QColor(168, 93, 253))
                    painter.drawText(self.rect(), Qt.AlignCenter, 'CensorTool selfcheck overlay')

            area = display.visible_area(target)
            if area is None:
                print("The window is not shown: skipping the test overlay.")
                return
            box = Box(None, Qt.FramelessWindowHint | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus
                      | Qt.Tool)
            box.setAttribute(Qt.WA_ShowWithoutActivating)
            box.winId()
            keep = ws.place_above(box.windowHandle(), target)  # noqa: F841 (kept while the overlay exists)
            box.setGeometry(*area)
            box.show()
            box.setGeometry(*area)
            ws.overlay_shown(box.windowHandle(), target)
            print(f"A purple box should cover the window exactly for {args.seconds:.0f} s, and stay below windows "
                  f"you move over it. Clicks should go through it to the window.")
            QTimer.singleShot(int(args.seconds * 1000), qt.quit)
            qt.exec()
            box.hide()

        step("test overlay")
        run("overlay", overlay)

    display.close()
    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
