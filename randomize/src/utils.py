import os
from sys import platform
from glob import glob
from .randomize import OUTPUT_DIR, HIDDEN_DATA


def hide_files():
    for csv in glob(os.path.join(OUTPUT_DIR, "*")):
        if (csv.startswith(os.path.join(OUTPUT_DIR, HIDDEN_DATA[1:]))
            and csv.endswith('.csv')):
            filename = csv.split(os.path.sep)[-1]
            if platform == 'darwin':
                os.rename(csv, os.path.join(OUTPUT_DIR,"."+filename))
            elif platform == 'win32':
                os.popen('attrib +h ' + csv)

def unhide_files():

    if platform == 'darwin':
        for csv in glob(os.path.join(OUTPUT_DIR, ".*")):
            if (csv.startswith(os.path.join(OUTPUT_DIR, HIDDEN_DATA))
                and csv.endswith('.csv')):
                filename = csv.split(os.path.sep)[-1]
                os.rename(csv, os.path.join(OUTPUT_DIR, filename[1:]))

    elif platform == 'win32':
        for csv in glob(os.path.join(OUTPUT_DIR, "*")):
            if (csv.startswith(os.path.join(OUTPUT_DIR, HIDDEN_DATA[1:]))
                and csv.endswith('.csv')):
                os.popen('attrib -h ' + csv)
