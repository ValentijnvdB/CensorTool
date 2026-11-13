"""Finding the window of a program started by overlay mode (python main.py overlay -- <command>)."""
import pytest

import main
from app.live import launch
from app.live.windowing import processes
from app.live.windowing import WindowInfo


class TestProgramNames:

    @pytest.mark.parametrize('command, names', [
        (['celluloid', '--new-window', '/home/me/clip.mp4'], {'celluloid'}),
        (['/usr/bin/celluloid'], {'celluloid'}),
        (['flatpak', 'run', 'io.github.celluloid_player.Celluloid', 'clip.mp4'],
         {'io.github.celluloid_player.celluloid'}),
        (['env', 'GDK_BACKEND=x11', 'mpv', 'clip.mp4'], {'mpv'}),
    ])
    def test_names(self, command, names):
        assert launch.program_names(command) == names


class TestClassMatches:

    @pytest.mark.parametrize('names, app_ids', [
        ({'celluloid'}, ('celluloid', 'io.github.celluloid_player.Celluloid')),
        ({'io.github.celluloid_player.celluloid'}, ('celluloid', 'io.github.celluloid_player.Celluloid')),
        ({'firefox'}, ('Navigator', 'firefox')),
        ({'mpv'}, ('gl', 'mpv')),
    ])
    def test_matches(self, names, app_ids):
        assert launch.class_matches(names, app_ids)

    @pytest.mark.parametrize('names, app_ids', [
        ({'celluloid'}, ('Navigator', 'firefox')),
        ({'celluloid'}, ()),
        (set(), ('celluloid', 'Celluloid')),
    ])
    def test_no_match(self, names, app_ids):
        assert not launch.class_matches(names, app_ids)


def test_process_tree():
    parents = {1: 0, 10: 1, 11: 10, 12: 11, 20: 1, 13: 10}
    assert processes.process_tree(10, parents) == {10, 11, 12, 13}
    assert processes.process_tree(99, parents) == {99}


def test_parent_map_has_this_process():
    import os
    assert processes.parent_map()[os.getpid()] == os.getppid()


def info(window, pid=None, app_ids=(), normal=True):
    return WindowInfo(window=window, pid=pid, app_ids=app_ids, normal=normal)


class TestChooseWindow:
    NAMES = {'celluloid'}

    def test_window_of_the_process_tree(self):
        new = [info(1, pid=500, app_ids=('other', 'Other')), info(2, pid=101)]
        assert launch.choose_window(new, {100, 101}, self.NAMES) == (2, 'pid')

    def test_normal_window_before_a_dialog(self):
        new = [info(1, pid=100, normal=False), info(2, pid=100)]
        assert launch.choose_window(new, {100}, self.NAMES) == (2, 'pid')

    def test_dialog_when_there_is_nothing_else(self):
        assert launch.choose_window([info(1, pid=100, normal=False)], {100}, self.NAMES) == (1, 'pid')

    def test_process_match_before_class_match(self):
        new = [info(1, pid=500, app_ids=('celluloid', 'Celluloid')), info(2, pid=100)]
        assert launch.choose_window(new, {100}, self.NAMES) == (2, 'pid')

    def test_class_match_for_other_processes(self):
        """Flatpak/Snap apps report another pid."""
        new = [info(1, pid=3, app_ids=('celluloid', 'Celluloid'))]
        assert launch.choose_window(new, {100}, self.NAMES) == (1, 'class-new')

    def test_nothing_yet(self):
        assert launch.choose_window([info(1, pid=500, app_ids=('x', 'X'))], {100}, self.NAMES) is None


class TestChooseExistingWindow:

    def test_topmost_normal_window_of_the_program(self):
        stacking = [info(1, app_ids=('celluloid', 'Celluloid')),  # bottom
                    info(2, app_ids=('celluloid', 'Celluloid')),
                    info(3, app_ids=('celluloid', 'Celluloid'), normal=False),
                    info(4, app_ids=('Navigator', 'firefox'))]  # top
        assert launch.choose_existing_window(stacking, {'celluloid'}) == 2

    def test_none(self):
        assert launch.choose_existing_window([info(1, app_ids=('Navigator', 'firefox'))], {'celluloid'}) is None


class TestStartProgram:

    def test_missing_program(self):
        with pytest.raises(launch.LaunchError, match='not found'):
            launch.start_program(['surely-not-a-program-censortool'])

    def test_environment_is_passed(self):
        proc = launch.start_program(['sh', '-c', 'exit "$CENSORTOOL_TEST"'],
                                    env={'PATH': '/usr/bin:/bin', 'CENSORTOOL_TEST': '3'})
        assert proc.wait(timeout=5) == 3


class TestSplitCommand:

    def test_without_command(self):
        assert main.split_command(['overlay', '-p', 'x']) == (['overlay', '-p', 'x'], None)

    def test_command_keeps_its_own_options_and_dashes(self):
        argv = ['overlay', '--curtain', '--', 'celluloid', '--new-window', '--', '-file.mp4']
        assert main.split_command(argv) == (['overlay', '--curtain'],
                                            ['celluloid', '--new-window', '--', '-file.mp4'])

    def test_empty_command(self):
        assert main.split_command(['overlay', '--']) == (['overlay'], [])
