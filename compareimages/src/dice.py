import numpy as np
import pandas as pd
import cv2
from typing import Tuple
from tqdm import tqdm
from PIL import Image
from itertools import combinations
from .utils import (inconsistent_name, create_overlapping_images,
                    read_binary, make_hyperlink)

def dice_coefficient(img1: np.ndarray, img2: np.ndarray, **kwargs) -> float:
    """
    Naive Implementation of Dice score calculation.
    """
    if kwargs.get('ignore_error'):
        if img1.shape != img2.shape and not kwargs.get('shape_only'):
            return 0.0

    else:
        assert img1.shape == img2.shape, \
        f"""Unequal image size at {kwargs.get('index')}:\n
        `{(kwargs.get('raters')[0]
           if kwargs.get('raters')
           else 'The first')} image has shape {img1.shape}\n
        while `{(kwargs.get('raters')[1]
                 if kwargs.get('raters')
                 else 'The second')}` image has shape {img2.shape}\n"""

    if kwargs.get('shape_only'):
        img1 = _get_shape1d(img1)
        img2 = _get_shape1d(img2)


    intersection, union = ((np.intersect1d(img1, img2).shape[0] / 2,
                           np.union1d(img1, img2).shape[0])
                           if kwargs.get('shape_only') else
                           (((img1.ravel() == img2.ravel()) * 1).sum(),
                             (img1.size + img2.size)))

    # Dice Cofficient: 2 * (A n B) / A U B
    return (2 * intersection) / union


def shape_intersection_union(img1: np.ndarray,
                             img2: np.ndarray) -> Tuple[int, int]:
    """
    Return the intersection and union of two shapes from binary images.
    """
    img1_dtype = img1.dtype
    img2_dtype = img2.dtype
    img1 = np.argwhere(img1 == 255)
    img2 = np.argwhere(img2 == 255)
    img1_view = img1.view([('',img1_dtype)]*img1.shape[1])
    img2_view = img2.view([('',img2_dtype)]*img2.shape[1])

    return (np.intersect1d(img1_view, img2_view).shape[0] / 2,
            np.union1d(img1_view, img2_view).shape[0])


def _get_shape1d(img: np.ndarray) -> np.ndarray:
    """
    Return the view of a 2d shape defined in a binary image.
    """
    img_dtype = img.dtype
    img = np.argwhere(img == 255)

    return img.view([('',img_dtype)]*img.shape[1])


def dice_scores(df:pd.DataFrame, **kwargs)->Tuple[pd.DataFrame, pd.DataFrame]:

    """
    Calculate image-wise and average (inter-/intra) rater Dice scores.
    """

    assert len(df.columns) > 1, \
    "DataFrame must contain more than 1 rater."

    if kwargs.get('ignore_inconsistent_name') == False:
        assert not inconsistent_name(df), \
        "DataFrame contains inconsistent names. Please check file names again."


    dice_df = pd.DataFrame()
    hyperlink_df = None
    temp_dfs = []
    tqdm.pandas()

    if kwargs.get('repeated_image'):
        indices = df.index.values
    else:
        indices = np.vectorize(lambda x: x.name)(df[df.columns[0]].values)
    # df = df.applymap(lambda x : cv2.imread(str(x), cv2.IMREAD_GRAYSCALE))

    # if kwargs.get('shape_only'):
    #     df = df.applymap(lambda x : _get_shape1d(x))

    for comb in combinations(df.columns, 2):
        rater_a, rater_b = comb
        dice_df[f"Dice-{rater_a}-{rater_b}"] = df.progress_apply(
        lambda x : dice_coefficient(read_binary(x[rater_a]),
                                    read_binary(x[rater_b]),
                                   index=x.name,
                                   raters=(rater_a, rater_b),
                                   ignore_error=kwargs.get('ignore_error'),
                                   shape_only=kwargs.get('shape_only')),
                                   axis = 1)

        if kwargs.get('create_overlapping_image'):
            if not kwargs.get('output_dir'): continue
            paths = create_overlapping_images(df[rater_a],
                                              df[rater_b],
                                              output_dir=kwargs.get('output_dir'),
                                              indices=indices,
                                              raters=(rater_a, rater_b),
                                              scale_down=kwargs.get('scale_down'))

            temp_df = pd.DataFrame(dice_df[f"Dice-{rater_a}-{rater_b}"]
                                    .astype('string') + "#" + paths)

            temp_df = temp_df.applymap(lambda x : make_hyperlink(x))
            temp_dfs.append(temp_df)

    if kwargs.get('create_overlapping_image'):
        hyperlink_df = pd.concat(temp_dfs, axis=1)
        hyperlink_df = hyperlink_df.set_index(indices)

    return (dice_df.set_index(indices),
            pd.DataFrame(dice_df.mean(axis=0), columns=['Dice Scores']),
            hyperlink_df)
