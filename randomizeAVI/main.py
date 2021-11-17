import argparse
import sys
import os
from datetime import datetime
from randomize import randomize, HIDDEN_DATA


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Program to randomize .avi files.', prefix_chars='-+')
    parser.add_argument('inputFolder', type=str, nargs='?', default='')
    parser.add_argument('-hide', action='store_false', default=True)
    parser.add_argument('+hide', action='store_true', default=False)
    parser.add_argument('--output-folder', type=str, metavar='O', default='')
    parser.add_argument('--duplicate-factor', type=float, metavar='Dup-Factor', default=0.3)

    args = parser.parse_args()

    if args.hide:
        if os.path.exists(HIDDEN_DATA[1:]): os.rename(HIDDEN_DATA[1:], HIDDEN_DATA)
    else:
        if os.path.exists(HIDDEN_DATA): os.rename(HIDDEN_DATA, HIDDEN_DATA[1:])

    if not args.inputFolder:
        sys.exit(0)

    print('Randomizing files in {}'.format(args.inputFolder))
    output_folder = ('output_{}'.format(datetime.now()
                                                .strftime("%Y-%m-%d_%H%M%S"))
                      if not args.output_folder
                      else args.output_folder)

    randomize(args.inputFolder, args.duplicate_factor, output_folder)
