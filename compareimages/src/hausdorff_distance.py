import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff
from itertools import combinations
from .utils import (is_black_image, create_overlapping_image,
                    scale_down, draw_hausdorff_lines, read_binary,
                    make_hyperlink, create_overlapping_images)

def get_canny_edge(img:np.ndarray, threshold1=0, threshold2=255,
                **kwargs)->np.ndarray:
    """
    Extract line by Canny Edge Method.
    """
    cnt = cv.Canny(img, threshold1, threshold2)
    return np.argwhere(cnt>0)


def get_contour(img:np.ndarray, retrieve_mode:int=cv2.RETR_TREE,
               approximation:int=cv2.CHAIN_APPROX_SIMPLE, **kwargs)->np.ndarray:

    """
    Extract line by cv2.findContours().
    """
    cnt, _ = cv2.findContours(img1, retrieve_mode, approximation)
    r, _, c = cnt[0].shape
    cnt_line = cnt[0].reshape(r,c)

    return cnt_line


def get_raw_points(img:np.ndarray, **kwargs)->np.ndarray:
    """
    Extract every point from binary image.
    """
    return np.argwhere(img>0)


def get_thin_line(img:np.ndarray, **kwargs)->np.ndarray:
    """
    Extract line by thinning method.
    """
    return np.argwhere(cv2.ximgproc.thinning(img) >0)


# def hausdorff_distance(img1:np.ndarray, img2:np.ndarray,
#                       hausdorff_extractor:cv2.HausdorffDistanceExtractor=cv2.createHausdorffDistanceExtractor(),
#                       extraction_method:str='thin',
#                       retrieve_mode:int=cv2.RETR_EXTERNAL,
#                       approximation:int=cv2.CHAIN_APPROX_TC89_KCOS,
#                       threshold1=0,
#                       threshold2=255,
#                       **kwargs)->float:
#
#     """
#     Compute Hausdorff Distance of `img1` and `img2` processed with
#     `extraction_method`.
#     """
#
#
#     extraction_methods = {
#         "thin" : get_thin_line,
#         "canny" : get_canny_edge,
#         "contour" : get_contour,
#         "raw_points" : get_raw_points
#     }
#
#
#     if kwargs.get('ignore_error'):
#         if (img1.shape != img2.shape
#         or (is_black_image(img1))
#         or (is_black_image(img2))):
#             return -1.0
#
#         else:
#             assert img1.shape == img2.shape, \
#             f"""Unequal image size at {kwargs.get('index')}:\n
#             ``{kwargs.get('raters')[0]}`'s image has size {img1.shape}\n
#             while `{kwargs.get('raters')[1]}`'s image has size {img2.shape}\n"""
#
#
#     line1 = extraction_methods[extraction_method](img1,
#                                                   retrieve_mode=retrieve_mode,
#                                                   approximation=approximation,
#                                                   threshold1=threshold1,
#                                                   threshold2=threshold2)
#     line2 = extraction_methods[extraction_method](img2,
#                                                   retrieve_mode=retrieve_mode,
#                                                   approximation=approximation,
#                                                   threshold1=threshold1,
#                                                   threshold2=threshold2)
#
#     if kwargs.get('ignore_error'):
#         if not (line1).all() or not (line2).all():
#             return -1.0
#
#     point_threshold = kwargs.get('point_threshold')
#
#     if point_threshold:
#         if (len(line1.ravel()) < point_threshold
#          or len(line2.ravel()) < point_threshold):
#             return -1.0
#
#
#     return hausdorff_extractor.computeDistance(line1, line2)



def hausdorff_distance(img1:np.ndarray, img2: np.ndarray,
                      extraction_method:str='thin',
                      retrieve_mode:int=cv2.RETR_EXTERNAL,
                      approximation:int=cv2.CHAIN_APPROX_TC89_KCOS,
                      threshold1=0,
                      threshold2=255,
                      **kwargs):

    extraction_methods = {
        "thin" : get_thin_line,
        "canny" : get_canny_edge,
        "contour" : get_contour,
        "raw_points" : get_raw_points
    }

    na = (-1.0, (-1.0,(0,0),(0,0)), (-1.0, (0,0), (0,0)))

    if kwargs.get('ignore_error'):
        if (img1.shape != img2.shape
        or (is_black_image(img1))
        or (is_black_image(img2))):
            return na

    else:
        assert img1.shape == img2.shape, \
        f"""Unequal image size at {kwargs.get('index')}:\n
        ``{kwargs.get('raters')[0]}`'s image has size {img1.shape}\n
        while `{kwargs.get('raters')[1]}`'s image has size {img2.shape}\n"""


    line1 = extraction_methods[extraction_method](img1,
                                                  retrieve_mode=retrieve_mode,
                                                  approximation=approximation,
                                                  threshold1=threshold1,
                                                  threshold2=threshold2)
    line2 = extraction_methods[extraction_method](img2,
                                                  retrieve_mode=retrieve_mode,
                                                  approximation=approximation,
                                                  threshold1=threshold1,
                                                  threshold2=threshold2)

    if kwargs.get('ignore_error'):
        if not (line1).all() or not (line2).all():
            return na

    point_threshold = kwargs.get('point_threshold')

    if point_threshold:
        if (len(line1.ravel()) < point_threshold
         or len(line2.ravel()) < point_threshold):
            return na

    hAB, hBA = directed_hausdorff(line1,line2), directed_hausdorff(line2,line1)
    hd, AtoB, BtoA = max(hAB[0], hBA[0]), hAB[1:], hBA[1:]


    return (hd,
            (hAB[0], tuple(line1[AtoB[0]][::-1]), tuple(line2[AtoB[1]][::-1])),
            (hBA[0], tuple(line2[BtoA[0]][::-1]), tuple(line1[BtoA[1]][::-1])))
    # Deprecated
    # return hd


def hausdorff_distances(df:pd.DataFrame, **kwargs)->pd.DataFrame:

    """
    Get Hausdorff Distance from all images in a DataFrame with image path.
    """

    assert len(df.columns) > 1, \
    "DataFrame must contain more than 1 rater."

    if kwargs.get('ignore_inconsistent_name') == False:
        assert not inconsistent_name(df), \
        "DataFrame contains inconsistent names. Please check file names again."


    hd_df = pd.DataFrame()
    temp_dfs = []
    tqdm.pandas()
    hyperlink_df = None

    if kwargs.get('repeated_image'):
        indices = df.index.values
    else:
        indices = np.vectorize(lambda x: x.name)(df[df.columns[0]].values)
    # extractor = cv2.createHausdorffDistanceExtractor()
    # df = df.applymap(lambda x : read_binary(x))

    for comb in combinations(df.columns, 2):
        carrier_df = pd.DataFrame()
        rater_a, rater_b = comb
        print(f'Comparing {rater_a} and {rater_b} ...')
        temp = df.progress_apply(
        lambda x : hausdorff_distance(read_binary(x[rater_a]),
                                      read_binary(x[rater_b]),
                                      index=x.name,
                                      raters=(rater_a, rater_b),
                                      ignore_error=kwargs.get('ignore_error'),
                                      point_threshold=kwargs.get('point_threshold')),
                                      axis = 1).apply(pd.Series)
        carrier_df[['hd', 'hAB', 'hBA']] = temp
        hd_df[f"Hausdorff_Distance-{rater_a}-{rater_b}"] = carrier_df['hd']

        if kwargs.get('create_overlapping_image'):
            if not kwargs.get('output_dir'): continue
            print('creating overlapped images....')
            distances = carrier_df[['hAB', 'hBA']].apply(tuple, axis=1)
            paths = create_overlapping_images(df[rater_a],
                                              df[rater_b],
                                              output_dir=kwargs.get('output_dir'),
                                              indices=indices,
                                              raters=(rater_a, rater_b),
                                              distances=distances,
                                              channels='rgb')

            temp_df = pd.DataFrame(hd_df[f"Hausdorff_Distance-{rater_a}-{rater_b}"]
                                    .astype('string') + "#" + paths)
            temp_df.fillna('-1.0#0', inplace=True)

            temp_df = temp_df.applymap(lambda x : make_hyperlink(x))
            temp_dfs.append(temp_df)

    if kwargs.get('create_overlapping_image'):
        hyperlink_df = pd.concat(temp_dfs, axis=1)
        hyperlink_df = hyperlink_df.set_index(indices)

    return hd_df.set_index(indices), hyperlink_df
