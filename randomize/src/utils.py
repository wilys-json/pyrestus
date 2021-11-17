import os
from sys import platform
from glob import glob
from .randomize import OUTPUT_DIR, HIDDEN_DATA


def hide_files():
    for csv in os.listdir(OUTPUT_DIR):
                if platform == 'darwin':
                    if (csv.startswith(os.path.join(OUTPUT_DIR, HIDDEN_DATA[1:]))
                        and csv.endswith('.csv')):
                            filename = csv.split(os.path.sep)[-1]
                            os.rename(csv, os.path.join(OUTPUT_DIR,"."+filename))
                elif platform == 'win32':
                    if csv.endswith('.csv'): os.popen('attrib +h ' + os.path.join(OUTPUT_DIR, csv))

def unhide_files():
    for csv in os.listdir(OUTPUT_DIR):
            if platform == 'darwin':
                    if (csv.startswith(os.path.join(OUTPUT_DIR, HIDDEN_DATA))
                        and csv.endswith('.csv')):
                            filename = csv.split(os.path.sep)[-1]
                            os.rename(csv, os.path.join(OUTPUT_DIR, filename[1:]))

            elif platform == 'win32':
                    if csv.endswith('.csv'): os.popen('attrib -h ' + os.path.join(OUTPUT_DIR, csv))
