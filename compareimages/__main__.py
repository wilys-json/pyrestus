import argparse
import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from src import (containsDir, makeRaterDataFrame,
                 makeRatersDataFrame, getSegmentationMask)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputFolder', type=str, nargs='?', default='')
    parser.add_argument('--generate-mask', action='store_true', default=False)
    parser.add_argument('-o', '--show-original', action='store_true',
                        default=False)
    parser.add_argument('-b', '--show-binary', action='store_true',
                        default=False)
    parser.add_argument('-m', '--show-mask', action='store_true',
                        default=False)
    parser.add_argument('--calculate-dice-score', action='store_true',
                        default=False)

    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")



    if not args.inputFolder:
        sys.exit(0)


    if containsDir(args.inputFolder):
        df = makeRatersDataFrame(args.inputFolder)
    else:
        df = makeRaterDataFrame(args.inputFolder)

    tqdm.pandas()

    if args.generate_mask:
        print(f"Generating segmentation mask from {args.inputFolder}...")
        df.progress_applymap(
            lambda x: getSegmentationMask(str(x), save=True,
                                          show_original=args.show_original,
                                          show_binary=args.show_binary,
                                          show_mask=args.show_mask,
                                          timestamp=timestamp))
        print(f"{len(df)} segmentation masks generated in {args.inputFolder}.")





if __name__ == '__main__':
    main()
