import numpy as np
import cv2
from pathlib import Path
from copy import deepcopy
from PIL import Image
from typing import List, Tuple, Union


HSV_FILTERS = {
    "blue" : (np.array([110,50,50]), np.array([130,255,255])),
    "green" : (np.array([50, 50, 120]), np.array([70, 255, 255])),
    "yellow" : (np.array([22, 93, 0]), np.array([45, 255, 255]))
}

CROPPING_RATIOS = (.256, .095, .953, .945)


def cropImage(img:np.ndarray, cropping:Tuple[float],
              keep_array=True)->Union[np.ndarray, Image.Image]:
    """
    Crop `img` w.r.t. the defined `cropping` ratio.
    """
    assert (isinstance(cropping, tuple) and
            len(cropping) == 4 and
            [type(element)
             for element in cropping] == [float] * len(cropping)), \
        "Cropping ratios must be in (float, float, float, float) format."

    img = Image.fromarray(img)
    w, h = img.size
    left_ratio, upper_ratio, right_ratio, lower_ratio = CROPPING_RATIOS
    img = img.crop((w * left_ratio, h * upper_ratio,
                    w * right_ratio, h * lower_ratio))
    if keep_array:
        return np.array(img)
    return img


def filterImage(img:np.ndarray,
                hsv_filters:List[Tuple[Tuple[float]]])->np.ndarray:
    """
    Filter specific color range from `img` w.r.t. `hsv_filters`.
    Return immediately if one of the colors identified.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color = np.zeros_like(img, np.uint8)
    blackimg = deepcopy(color)

    for hsv_filter in hsv_filters:
        if (color != blackimg).all():
            return color
        hsv_min, hsv_max = hsv_filter
        mask = cv2.inRange(hsv, hsv_min, hsv_max)
        imask = mask > 0
        color[imask] = img[imask]

    if (color == blackimg).all(): print("No segmentation detected.")

    return color


def extractSegment(img:np.ndarray,
                   correction:Tuple[int]=(2,2),
                   iterations:int=10,
                   smoothing:Tuple[int]=(7,7),
                   binary_threshold:int=127)->np.ndarray:

    """
    Extract filtered segments and convert to an binary image.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones(correction, 'uint8')
    dilated = cv2.dilate(gray, kernel, iterations=iterations)
    corrected = cv2.erode(dilated, kernel, iterations=iterations)
    corrected = cv2.GaussianBlur(corrected, smoothing, 0)
    ret, thresh = cv2.threshold(corrected, binary_threshold,
                                255, cv2.THRESH_BINARY)

    return thresh


def fillContours(img:np.ndarray)->Image.Image:

    """
    Fill shape defined by a closed contour.
    """
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    result = np.zeros_like(img)
    for contour in contours:
        cv2.drawContours(result, [contour], 0, (255, 255, 255), cv2.FILLED)

    return Image.fromarray(result)



def getSegmentationMask(img_path: str, cropping=True,
                        hsv_filters=list(HSV_FILTERS.values()),
                        correction=(2,2), morph_iter=10, smoothing=(7,7),
                        binary_threshold=80, show=False, save=False):
    img = cv2.imread(img_path)


    """
    Get a segmentation mask (in binary image form) from edge-defined
    segmentation images.
    """

    # cropping
    if cropping:
        img = cropImage(img, CROPPING_RATIOS)

#     width, height, _ = img.shape
    color = filterImage(img, hsv_filters)

    thresh = extractSegment(color, correction=correction,
                             iterations=morph_iter,
                             smoothing=smoothing,
                             binary_threshold=binary_threshold)


    segmented_img = fillContours(thresh)

    Image.fromarray(img).show()
    segmented_img.show()


    if save:
        output_dir = kwargs.get('output_dir')
        output_dir = Path(output_dir) if output_dir else Path('output')
        if not output_dir.is_dir():
            output_dir.mkdir()
        segmented_img.save(output_dir / f'{Path(img_path).stem}_segmented.png')
