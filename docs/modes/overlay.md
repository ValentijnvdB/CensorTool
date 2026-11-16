# Overlay: censoring windows in place (`overlay`)

```
pip install -r requirements-overlay.txt   # once: installs PySide6
python main.py overlay                    # then click the window to censor
python main.py overlay --window 0x3a00007 --window 0x4200003  # or give ids (from xwininfo or wmctrl -l)
python main.py overlay -- celluloid --new-window ~/Videos/clip.mp4  # or start a program and censor its window
python main.py overlay --auto celluloid mpv  # or censor every window of these programs (auto mode)
python main.py overlay --focused          # or censor the focused window (--focused 3: after 3 seconds)
```

`overlay` captures a window and shows it censored in an overlay that lies exactly over it, so you don't need a
second display. The overlay lets clicks, scrolling and typing through to the window below, so you can use it as
usual. It follows the window when it is moved or resized and hides with it when it is minimised. Other windows,
menus and panels stay on top of it.

The overlay is shown before the first frame is censored, so the window is never visible uncensored. Like live mode, the
censored image lags slightly behind the window (the time it takes to censor a frame), which you notice when
scrolling. Menus, dropdowns and tooltips of the censored window are separate windows and are not censored.

Overlay mode works on Linux with an X11 session. It has been tested on Cinnamon (Linux Mint); Wayland and macOS are
not supported yet. For Windows, see [below](#windows).

## Starting a program

Everything after `--` is a program to start; its window is censored, and so are the windows it opens later (dialogs,
more windows). The program keeps running when overlay mode stops. Its window can be visible for a moment before the
overlay covers it; add `--curtain` to cover your screens while the program starts. Programs that run only once, like
Celluloid and Firefox, hand the file to the instance that is already running and exit. Overlay mode covers the
program's open windows before starting it, so the new file isn't shown uncensored there, then censors that instance's
window with a warning. Use the program's own option for a separate window, such as `celluloid --new-window`.

## The focused window

`--focused` censors the window that has the focus when overlay mode starts. It is meant for starting overlay mode
from another program or a keyboard shortcut, while the window to censor is focused. Started from a terminal, the
focused window is the terminal itself, so give a delay in seconds (`--focused 3`) and switch to the window in the
meantime. Overlay mode stops with an error when the desktop or a panel has the focus. Each start is a separate
overlay mode with its own tray icon. `--focused` can be combined with `--auto`, but not with `--window` or `--`.

## Tray menu

A tray icon has *Add window...*, *Stop censoring* (a list of the censored windows), *Hidden windows* (see
[below](#several-windows)), *Auto mode*, *Always censor a program...*, *Reload configs* and *Quit*; without a system
tray, stop it with Ctrl+C. Started with `--window`, `--focused` or `--` (and without `--auto`), overlay mode stops when the last
censored window is closed. Otherwise it keeps running in the tray so you can add a window, also when you cancel the
first click or close the last censored window. A window you stop censoring isn't censored automatically again until it
is closed.

## Auto mode

Auto mode (`--auto`, or *Auto mode* in the tray) censors every window of the programs listed in
`overlay_auto_programs` in `config.yml`, including windows that are open already and windows they open later.
Programs given with `--auto` are added to that list for this run; *Always censor a program...* asks you to click a
window and adds its program to the list in `config.yml`, so it applies from then on. An entry can also match the
window title, with a case-insensitive regular expression:

```yaml
overlay_auto_programs: [celluloid, mpv, {program: firefox, title: 'Private Browsing'}]
```

Programs are matched by their window class (`xprop WM_CLASS` shows it), ignoring case and a reverse-domain prefix,
so `celluloid` matches `io.github.celluloid_player.Celluloid`. A window is censored as soon as its title matches,
and stays censored when the title changes back. Auto mode notices a new window about when it appears, but its overlay
can follow a frame or two later (about 20-35 ms in tests), so a new window can be visible uncensored very briefly.
`--auto` can be combined with `--window` and with `-- <command>`.

## Several windows

All windows share the censoring: windows whose content changes take turns, and the focused window gets three times
as many turns as the others. A window that doesn't change (a paused video, an idle page) costs almost nothing. At
most `live_max_windows` windows (`config.yml`, 4 by default) are censored at the same time; a faster PC handles
more. A window found automatically (auto mode, or a window a started program opens) beyond that limit is hidden
behind a dark placeholder instead; *Hidden windows* in the tray can censor it once you stopped censoring another,
or show it uncensored. Overlay mode also uses the live mode settings `live_input_size`, `live_frames_in_flight` and
`live_show_fps` (see [config.md](../config.md#live-webcam-and-overlay)); the capture region `live_cap_*` is not used.

## Windows

**Windows (10 and 11) support is written but has not been run on Windows yet.** It captures windows with Windows
Graphics Capture when the `windows-capture` package is installed (part of `requirements-overlay.txt`), and falls back
to the slower PrintWindow, with which some programs come out black. On Windows 10 a yellow border shows around
captured windows. Programs are matched by the name of their executable without `.exe` (`vlc`, `firefox`), and
`--window` takes a window handle (HWND). To check what works on a computer, run
`python -m app.live.windowing.selfcheck --pick`: it captures, follows and covers a window you click, step by step,
and reports what fails.

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--window` | click | X11 id of a window to censor, in hex (`0x3a00007`) or decimal. Repeat it for more windows. |
| `-- <command>` | - | Start this program and censor its window. Must come last. |
| `--curtain` | off | With `-- <command>`: cover the screens while the program starts. |
| `--auto [PROGRAM ...]` | off | Auto mode: censor every window of the programs in `overlay_auto_programs` (`config.yml`) and of the programs given here. |

The [general arguments](../../README.md#usage) (`--profile`, `--debug`, ...) work as well.
