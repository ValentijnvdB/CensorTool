from abc import ABC, abstractmethod

import numpy as np


class InputDevice(ABC):

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def get_frame(self) -> tuple[float, np.ndarray]:
        pass

    def screen_region(self) -> tuple[int, int, int, int] | None:
        """
        The (left, top, width, height) region of the screen the last frame was captured from,
        or None if the frames are not a capture of the screen (the mouse cursor is then not drawn).
        """
        return None
