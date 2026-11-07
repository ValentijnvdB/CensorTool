"""Minimal interactive selection menu for the terminal (arrow keys + enter), without extra dependencies."""
import os
import sys

UP, DOWN, ENTER = 'up', 'down', 'enter'


def is_interactive() -> bool:
    """Whether stdin and stdout are a terminal, so a menu can be shown."""
    return sys.stdin.isatty() and sys.stdout.isatty()


if os.name == 'nt':
    import msvcrt

    def _enable_ansi():
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING

    def _read_keys():
        _enable_ansi()
        while True:
            ch = msvcrt.getwch()
            if ch in ('\x00', '\xe0'):  # special key, the next char is the key code
                code = msvcrt.getwch()
                if code == 'H':
                    yield UP
                elif code == 'P':
                    yield DOWN
            elif ch in ('\r', '\n'):
                yield ENTER
            elif ch == '\x03':
                raise KeyboardInterrupt
            elif ch in ('k', 'w'):
                yield UP
            elif ch in ('j', 's'):
                yield DOWN

else:
    import termios
    import tty

    def _read_keys():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)  # keystrokes without enter; Ctrl+C still raises KeyboardInterrupt
            while True:
                ch = os.read(fd, 1)
                if ch == b'\x1b':
                    seq = os.read(fd, 2)
                    if seq in (b'[A', b'OA'):
                        yield UP
                    elif seq in (b'[B', b'OB'):
                        yield DOWN
                elif ch in (b'\r', b'\n'):
                    yield ENTER
                elif ch in (b'k', b'w'):
                    yield UP
                elif ch in (b'j', b's'):
                    yield DOWN
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def select(title: str, options: list[str], index: int = 0) -> str:
    """
    Show the options in the terminal and let the user pick one with the arrow keys and enter.

    :raises KeyboardInterrupt: if the user presses Ctrl+C.
    """
    if not options:
        raise ValueError("No options to select from.")

    out = sys.stdout

    def render(first: bool):
        if not first:
            out.write(f"\x1b[{len(options)}F")  # back to the first option line
        for i, option in enumerate(options):
            line = f"\x1b[1;36m> {option}\x1b[0m" if i == index else f"  {option}"
            out.write(f"\x1b[2K{line}\n")
        out.flush()

    out.write(f"{title} \x1b[2m(arrow keys to move, enter to select)\x1b[0m\n\x1b[?25l")  # hide the cursor
    try:
        render(first=True)
        for key in _read_keys():
            if key == UP:
                index = (index - 1) % len(options)
            elif key == DOWN:
                index = (index + 1) % len(options)
            elif key == ENTER:
                break
            render(first=False)
    finally:
        out.write("\x1b[?25h")  # show the cursor again
        out.flush()

    return options[index]
