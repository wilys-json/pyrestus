import pandas as pd
import numpy as np
import warnings
import sys
from pathlib import Path

IMAGE_FORMATS = ['.jpg', '.png', '.jpeg', '.bmp']

def containsDir(folder:str)->bool:
    """
    Check if a directory contains any subdirectories.
    """
    return all([f.is_dir() for f in Path(folder).iterdir()])

def makeRaterDataFrame(folder:str)->pd.DataFrame:
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


def makeRatersDataFrame(folder:str, ignore_null:bool=True,
                       warning:bool=True)->pd.DataFrame:
    """
    Make pd.DataFrame from a directory containing raw segmentations
    or segmentation masks from multiple Raters.
    """

    dfs = []
    for subdir in Path(folder).iterdir():
        if subdir.is_dir():
            dfs.append(makeRaterDataFrame(subdir))
        else:
            print("Master folder can only contain multiple rater folders.")
            sys.exit(0)

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


def inconsistentName(df:pd.DataFrame)->bool:
    """
    Check if every row contains the same file name.
    """
    assert not df.isnull().values.any(), \
    "DataFrame contains empty values. Please use .dropna() before name checking"

    return df.apply(lambda files : len(set([file.name for file in files])) > 1,
                    axis=1).values.any()

# TODO: modify this function
def color_values(val:float)->str:
    color = '' if val < 1.0 else 'green'
    return 'background-color: %s' % color
