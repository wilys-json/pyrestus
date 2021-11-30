import numpy as np
import cv2
from pathlib import Path
from copy import deepcopy
from PIL import Image
from typing import List, Tuple, Union


HSV_FILTERS = {
    "blue" : ((110,50,50),
              (130,255,255)),
    "green" : ((30, 0, 0),
               (80, 255, 255)),
    "yellow" : ((22, 93, 0),
                (45, 255, 255))
}

CROPPING_RATIOS = (.256, .095, .953, .945)

MORPHOLOGICAL_CORRECTION = (2,2)

MORPHOLOGICAL_ITERATIONS = 10

GAUSSIAN_SMOOTHING = (7,7)

BINARY_THRESHOLD = 127

OUTPUT_DIR = 'segmented'

def crop_image(img:np.ndarray, cropping:Tuple[float],
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
    left_ratio, upper_ratio, right_ratio, lower_ratio = cropping
    img = img.crop((w * left_ratio, h * upper_ratio,
                    w * right_ratio, h * lower_ratio))
    if keep_array:
        return np.array(img)

    return img


def filter_image(img:np.ndarray,
                id:str,
                hsv_filters:List[Tuple[Tuple[float]]])->np.ndarray:
    """
    Filter specific color range from `img` w.r.t. `hsv_filters`.
    Return immediately if one of the colors identified.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color = np.zeros_like(img, np.uint8)
    blackimg = deepcopy(color)

    for hsv_filter in hsv_filters:
        # if not (color == blackimg).all():
        #     print(hsv_filter)
        #     return color
        hsv_min, hsv_max = hsv_filter
        mask = cv2.inRange(hsv, hsv_min, hsv_max)
        imask = mask > 0
        color[imask] = img[imask]

    if (color == blackimg).all():
        print(f"Warning: No segmentation detected in {id}.")

    return color


def extract_segment(img:np.ndarray,
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


def fill_contours(img:np.ndarray)->Image.Image:

    """
    Fill shape defined by a closed contour.
    """
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    result = np.zeros_like(img)
    for contour in contours:
        cv2.drawContours(result, [contour], 0, (255, 255, 255), cv2.FILLED)

    return Image.fromarray(result)



def get_segmentation_mask(img_path: str, cropping:Tuple[float]=CROPPING_RATIOS,
                        hsv_filters:List=list(HSV_FILTERS.values()),
                        correction:Tuple[int]=MORPHOLOGICAL_CORRECTION,
                        morph_iter:int=MORPHOLOGICAL_ITERATIONS,
                        smoothing:Tuple[int]=GAUSSIAN_SMOOTHING,
                        binary_threshold:int=BINARY_THRESHOLD,
                        save:bool=True, **kwargs):



    """
    Get a segmentation mask (in binary image form) from edge-defined
    segmentation images.
    """

    img = cv2.imread(img_path)

    if cropping:
        img = crop_image(img=img, cropping=CROPPING_RATIOS)

    color = filter_image(img=img, id=img_path, hsv_filters=hsv_filters)

    # if kwargs.get('show_binary') == True:
    #     Image.fromarray(color).show()

    thresh = extract_segment(img=color,
                            correction=correction,
                            iterations=morph_iter,
                            smoothing=smoothing,
                            binary_threshold=binary_threshold)

    if kwargs.get('show_original') == True:
        Image.fromarray(img).show()

    if kwargs.get('show_binary') == True:
        Image.fromarray(thresh).show()

    if kwargs.get('lines_only') == True:
        segmented_img = Image.fromarray(thresh)
    else:
        segmented_img = fill_contours(img=thresh)

    if kwargs.get('show_mask') == True:
        segmented_img.show()


    if save:
        output_dir = kwargs.get('output_dir')
        timestamp = kwargs.get('timestamp')
        output_dir = ((Path(output_dir) if output_dir
                     else Path(OUTPUT_DIR)))

        if not output_dir.is_dir():
            output_dir.mkdir()
        output_dir = output_dir / f'{Path(img_path).parent.stem}-{timestamp}'
        if not output_dir.is_dir():
            output_dir.mkdir()
        segmented_img.save(output_dir / f'{Path(img_path).stem}_segmented.png')
