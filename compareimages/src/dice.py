import numpy as np
import pandas as pd
from typing import Tuple
from tqdm import tqdm
from PIL import Image
from itertools import combinations
from .utils import inconsistent_name

def dice_coefficient(img1: Image, img2: Image, **kwargs) -> float:
    """
    Naive Implementation of Dice score calculation.
    """
    if kwargs.get('ignore_error') == True:
        if img1.size != img2.size:
            return 0.0

    else:
        assert img1.size == img2.size, \
        f"""Unequal image size at {kwargs.get('index')}:\n
        ``{kwargs.get('raters')[0]}`'s image has size {img1.size}\n
        while `{kwargs.get('raters')[1]}`'s image has size {img2.size}\n"""

    img1_arr = np.array(img1.resize((1,img1.size[0]*img1.size[1])))
    img2_arr = np.array(img2.resize((1,img2.size[0]*img2.size[1])))

    intersection = ((img1_arr == img2_arr) * 1).sum()

    # Dice Cofficient: 2 * (A n B) / A U B
    return (2 * intersection) / (img1_arr.size + img2_arr.size)



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
    tqdm.pandas()
    indices = np.vectorize(lambda x: x.name)(df[df.columns[0]].values)
    for comb in combinations(df.columns, 2):
        rater_a, rater_b = comb
        dice_df[f"Dice-{rater_a}-{rater_b}"] = df.progress_apply(
        lambda x : dice_coefficient(Image.open(x[rater_a]),
                                   Image.open(x[rater_b]),
                                   index=x.name,
                                   raters=(rater_a, rater_b),
                                   ignore_error=kwargs.get('ignore_error')),
                                   axis = 1)

    return (dice_df.set_index(indices),
            pd.DataFrame(dice_df.mean(axis=0), columns=['Dice Scores']))
