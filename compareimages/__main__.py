import argparse
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from src import (containsDir, makeRaterDataFrame,
                 makeRatersDataFrame, getSegmentationMask, DiceScores)

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
    parser.add_argument('-l', '--lines-only', action='store_true',
                        default=False)
    parser.add_argument('--ignore-error', action='store_true',
                        default=False)
    parser.add_argument('--ignore-inconsistent-name', action='store_true',
                        default=False)
    parser.add_argument('--calculate-dice-score', action='store_true',
                        default=False)
    parser.add_argument('--repeated-image', type=str, default='')
    parser.add_argument('--no-csv-header', action='version', version=None,
                        default=0)
    parser.add_argument('--debug', action='store_true', default=False)

    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    if not args.inputFolder:
        sys.exit(0)

    input_dir = Path(args.inputFolder)
    if args.repeated_image:
        args.ignore_inconsistent_name = True
        match_file_type = Path(args.repeated_image).suffix
        assert match_file_type == '.csv', \
            f"""Must provide .csv file for repeated image Dice.\n
                {match_file_type} file was provided."""

        df = pd.read_csv(args.repeated_image, header=args.no_csv_header)
        df = df.applymap(lambda x: (input_dir / x))

    else:
        if containsDir(args.inputFolder):
            df = makeRatersDataFrame(args.inputFolder)
        else:
            df = makeRaterDataFrame(args.inputFolder)

    tqdm.pandas()

    if args.debug:
        debug_log_dir = Path('log')
        if not debug_log_dir.is_dir():
            debug_log_dir.mkdir()
        df.to_html((debug_log_dir / f'DataFrame-{timestamp}.html'))

    if args.generate_mask:
        print(f"Generating segmentation mask from {args.inputFolder}...")
        df.progress_applymap(
            lambda x: getSegmentationMask(str(x), save=True,
                                          show_original=args.show_original,
                                          show_binary=args.show_binary,
                                          show_mask=args.show_mask,
                                          lines_only=args.lines_only,
                                          timestamp=timestamp))
        print(f"{df.size} segmentation masks generated in {args.inputFolder}.")
        sys.exit(0)


    if args.calculate_dice_score:
        output_dir = Path('outputs')
        image_wise_dice, average_dice = DiceScores(df,
                    ignore_error=args.ignore_error,
                    ignore_inconsistent_name=args.ignore_inconsistent_name)

        if not output_dir.is_dir():
            output_dir.mkdir()
        image_wise_dice.to_html(output_dir / f"Image-Wise-Dice-{timestamp}.html")
        average_dice.to_html(output_dir / f"Average-Dice-{timestamp}.html")
        sys.exit(0)


if __name__ == '__main__':
    main()
