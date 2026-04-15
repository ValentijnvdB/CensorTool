from abc import ABC, abstractmethod

import numpy as np


class InputDevice(ABC):

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def get_frame(self) -> tuple[float, np.ndarray]:
        pass


