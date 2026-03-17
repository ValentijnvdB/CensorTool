import cv2
import numpy as np


def bytes_to_np(image_bytes: bytes) -> np.ndarray:
    np_data = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)

def np_to_bytes(np_image: np.ndarray, ext: str) -> bytes:
    return bytes(cv2.imencode(ext=ext, img=np_image)[1])