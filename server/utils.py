import cv2
import numpy as np


def bytes_to_np(image_bytes: bytes) -> np.ndarray:
    np_data = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)

def np_to_bytes(np_image: np.ndarray, ext: str, quality: int | None = None) -> bytes:
    """
    Encode an image.

    :param quality: 1-100 for '.jpg'/'.jpeg' and '.webp'; None for OpenCV's default (for WebP that is lossless,
        which is slow: about 200 ms for a 1080p frame).
    """
    params = []
    if quality is not None:
        if ext.lower() in ('.jpg', '.jpeg'):
            params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
        elif ext.lower() == '.webp':
            params = [cv2.IMWRITE_WEBP_QUALITY, int(quality)]
    return bytes(cv2.imencode(ext, np_image, params)[1])