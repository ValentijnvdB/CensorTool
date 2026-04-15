from abc import ABC, abstractmethod

import numpy as np


class OutputDevice(ABC):

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def push_frame(self, frame: np.ndarray):
        pass