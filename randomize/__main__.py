import argparse
import sys
import os
from glob import glob
from datetime import datetime
from src import randomize, HIDDEN_DATA, OUTPUT_DIR

def main():
    parser = argparse.ArgumentParser(prefix_chars='-+')
    parser.add_argument('inputFolder', type=str, nargs='?', default='')
    parser.add_argument('-hide', action='store_false', default=True)
    parser.add_argument('+hide', action='store_true', default=False)
    parser.add_argument('--output-folder',
                        type=str, metavar='O', default='randomized')
    parser.add_argument('--duplicate-factor', type=float,
                        metavar='Dup-Factor', default=0.3)

    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)


    if args.hide:
        for csv in glob(os.path.join(OUTPUT_DIR, "*")):
            if (csv.startswith(os.path.join(OUTPUT_DIR, HIDDEN_DATA[1:]))
                and csv.endswith('.csv')):
                filename = csv.split(os.path.sep)[-1]
                os.rename(csv, os.path.join(OUTPUT_DIR,"."+filename))
    else:
        for csv in glob(os.path.join(OUTPUT_DIR, ".*")):
            if (csv.startswith(os.path.join(OUTPUT_DIR, HIDDEN_DATA))
                and csv.endswith('.csv')):
                filename = csv.split(os.path.sep)[-1]
                os.rename(csv, os.path.join(OUTPUT_DIR, filename[1:]))

    if not args.inputFolder:
        sys.exit(0)

    print('Randomizing files in {}'.format(args.inputFolder))
    output_folder = os.path.join(OUTPUT_DIR, '{}_{}'.format(args.output_folder,
                                                            timestamp))

    randomize(timestamp, args.inputFolder, args.duplicate_factor, output_folder)


if __name__ == '__main__':
    main()
