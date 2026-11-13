from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

Command = Literal['quit', 'reload']


class OutputDevice(ABC):

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def push_frame(self, frame: np.ndarray, timestamp: float | None = None, source=None):
        """
        Show a censored frame.

        :param timestamp: the capture timestamp of the frame (time.monotonic(), see InputDevice.get_frame).
        :param source: the key of the input device the frame came from (see quick.Sources).
        """
        pass

    def poll_command(self) -> Command | None:
        """A command the user gave through this device since the last call (e.g. a key press), or None."""
        return None
