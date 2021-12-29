import numpy as np
import cv2
from typing import Tuple, Union, List

__all__ = [
    'detect_lines'
]

THETA = {
    'convex' : 90,
    'linear' : 180,
}

def detect_lines(image:np.ndarray,
                 kernel:Tuple[int],
                 transducer:str)->List[Union[None, int]]:
        """
        Detect left and right most lines from an ultrasound image.
        """

        # Canny edge detection
        cimg = cv2.Canny(cv2.GaussianBlur(image,
                                          kernel,
                                          0), 0, 255)

        # Hough line detection
        lines = cv2.HoughLines(cimg, 1,
                               np.pi / 180,
                               THETA[transducer], None, 0, 0)

        # Identify left & right most lines
        if lines is not None:
            cut_off_points = (lines.ravel()
                                   .reshape(-1,2)
                                   .T[0]
                                   .astype('int32'))
            left, right = cut_off_points.min(), cut_off_points.max()
            return [left if left >= 0 else None,
                    right if right >= 0 else None]

        # None if no lines detected
        return [None]
