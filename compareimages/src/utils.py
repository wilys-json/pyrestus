import pandas as pd
import numpy as np
import warnings
import sys
import cv2
from pathlib import Path
from typing import Tuple
from secrets import token_hex

IMAGE_FORMATS = ['.jpg', '.png', '.jpeg', '.bmp']

def contains_dir(folder:str)->bool:
    """
    Check if a directory contains any subdirectories.
    """
    return all([f.is_dir() for f in Path(folder).iterdir() if f.name[0] != '.'])


def make_rater_dataframe(folder:str)->pd.DataFrame:
    """
    Make a single column pd.DataFrame from a directory.
    """
    folder = Path(folder)
    assert folder.is_dir(), f"Cannot find folder {folder}."
    return pd.DataFrame((file for file in sorted(folder.iterdir())
                         if (not file.is_dir())
                          and (file.suffix in IMAGE_FORMATS)
                          and (file.name[0] != '.')),
                        columns=[folder.name])


def make_raters_dataframe(folder:str, ignore_null:bool=True,
                          warning:bool=True)->pd.DataFrame:
    """
    Make pd.DataFrame from a directory containing raw segmentations
    or segmentation masks from multiple Raters.
    """

    dfs = []
    for subdir in Path(folder).iterdir():
        if subdir.is_dir():
            dfs.append(make_rater_dataframe(subdir))


    main_df = pd.concat(dfs, axis=1)
    is_null = sum(main_df.isnull().values.any(axis=1))

    if ignore_null:
        main_df = main_df.dropna()

    if warning:
        if is_null > 0:
            warnings.warn(f"""{is_null} column(s) contain(s) empty values due to
                            unequal file numbers.""")
            if ignore_null:
                warnings.warn(f"{is_null} columns(s) have/has been dropped")


    return main_df


def inconsistent_name(df:pd.DataFrame)->bool:
    """
    Check if every row contains the same file name.
    """
    assert not df.isnull().values.any(), \
    "DataFrame contains empty values. Please use .dropna() before name checking"

    return df.apply(lambda files : len(set([file.name for file in files])) > 1,
                    axis=1).values.any()


def unify_image_size(): pass


def is_black_image(img:np.ndarray)->bool:

    """
    Check if Image is all black.
    """
    return (img == 0).all()


def _color_values(val:float, flag_val:float,
                  operator:str, color:str='green')->str:
    """
    Color a Pandas cell.
    """
    color = '' if eval(f'{str(val)}{operator}{str(flag_val)}') else color
    return 'background-color: %s' % color


def flag_red(val:float, flag_val:float, operater:str)->str:
    """
    Color a Pandas cell red.
    """
    return _color_values(val, flag_val, operator, 'red')


def flag_green(val:float, flag_val:float, operator:str)->str:
    """
    Color a Pandas cell green.
    """
    return _color_values(val, flag_val, operator, 'green')


def create_overlapping_image(img1: np.ndarray,
                             img2: np.ndarray,
                             alpha=0.5,
                             beta=0.5,
                             **kwargs)->np.ndarray:

    """
    Create an overlapping image from `img1` and `img2`.
    Set transparency of `img1` with `alpha`.
    Set transparency of `img2` with `beta`.
    Add optional keyword argument `channels` : 'rgb' to covert to RGB.
    """

    if kwargs.get('ignore_error'):
        r, c = min(img1.shape, img2.shape)
        img1 = img1[:r, :c]
        img2 = img2[:r, :c]

    else:
        assert img1.shape == img2.shape, \
        "Images have different sizes."

    output = cv2.addWeighted(img1, alpha, img2, beta, 0)

    if kwargs.get('channels') == 'rgb':
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

    output_dir = kwargs.get('output_dir')

    if output_dir:
        raters = 'A-B' if not kwargs.get('raters') else kwargs.get('raters')
        output_dir = output_dir / 'overlapped_images' / '-'.join(raters)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = kwargs.get('filename')
        filename = str((output_dir / f'{filename}') if filename
                    else (output_dir / f'{token_hex(4).upper()}'))
        cv2.imwrite(filename, output)
        return output, filename

    return output

def scale_down(img:np.ndarray, scaling_factor:float=0.2)->np.ndarray:
    """
    Scale down an image w.r.t. `scaling_factor`.
    """
    height, width = img.shape[0], img.shape[1]
    dim = (int(width * scaling_factor), int(height * scaling_factor))

    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


def draw_hausdorff_lines(overlapped_img:np.ndarray,
                         directed_hausdorff_AtoB:Tuple[float, Tuple[int, int]],
                         directed_hausdorff_BtoA:Tuple[float, Tuple[int, int]],
                         thickness:int=1):

    """
    Draw lines indicating directed Hausdorff Distance.
    Green line indicates the minimum of directed Hausdorff Distance,
    while Red line indicates the maxmum.
    """

    red = (255, 0, 0)
    green = (0, 255, 0)

    if len(overlapped_img.shape) < 3:
        overlapped_img = cv2.cvtColor(overlapped_img, cv2.COLOR_GRAY2RGB)

    min_directed_hd, max_directed_hd = sorted([directed_hausdorff_AtoB,
                                               directed_hausdorff_BtoA])

    min_starting_point, min_ending_point = min_directed_hd[1:]
    cv2.line(overlapped_img, min_starting_point,
             min_ending_point, green, thickness)

    max_starting_point, max_ending_point = max_directed_hd[1:]
    cv2.line(overlapped_img, max_starting_point,
            max_ending_point, red, thickness)


def create_overlapping_images(img_col1: pd.Series, img_col2: pd.Series,
                              output_dir:Path, **kwargs):
    """
    Create a list of overlapping images from `img_col1` and `img_col2`
    """
    image_pairs = pd.concat([img_col1, img_col2], axis=1)

    image_pairs.set_index(kwargs.get('indices'), inplace=True)
    filepaths = []
    for idx, img1, img2 in image_pairs.itertuples():
        _, filepath=create_overlapping_image(read_binary(img1),
                                             read_binary(img2),
                                             channels=kwargs.get('channels'),
                                             output_dir=output_dir,
                                             raters=kwargs.get('raters'),
                                             filename=idx, ignore_error=True)
        filepaths += [filepath]

    return filepaths


def make_hyperlink(val:str):
    """
    Styling function for making hyperlink from `#`-seperated string.
    """
    value, link = val.split('#')
    return '<a href="{}">{}</a>'.format(link, value)

def read_binary(file:Path):

    return cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
