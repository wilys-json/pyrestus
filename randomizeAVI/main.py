import argparse
import sys
import os
from randomize import randomize, HIDDEN_DATA


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Program to randomize .avi files.', prefix_chars='-+')
    parser.add_argument('inputFolder', type=str, nargs='?', default='')
    parser.add_argument('-hide', action='store_false', default=True)
    parser.add_argument('+hide', action='store_true', default=False)
    parser.add_argument('--output-folder', type=str, metavar='O', default='output')
    parser.add_argument('--duplicate-factor', type=float, metavar='Dup-Factor', default=0.3)

    args = parser.parse_args()

    if args.hide:
        if os.path.exists(HIDDEN_DATA[1:]): os.rename(HIDDEN_DATA[1:], HIDDEN_DATA)
    else:
        if os.path.exists(HIDDEN_DATA): os.rename(HIDDEN_DATA, HIDDEN_DATA[1:])

    if not args.inputFolder:
        sys.exit(0)

    print('Randomizing files in {}'.format(args.inputFolder))
    randomize(args.inputFolder, args.duplicate_factor, args.output_folder)
