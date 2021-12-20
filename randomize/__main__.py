#################################################################################
# MIT License                                                                   #
#                                                                               #
# Copyright (c) 2021 Wilson Lam                                                 #
#                                                                               #
# Permission is hereby granted, free of charge, to any person obtaining a copy  #
# of this software and associated documentation files (the "Software"), to deal`#
# in the Software without restriction, including without limitation the rights  #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell     #
# copies of the Software, and to permit persons to whom the Software is         #
# furnished to do so, subject to the following conditions:                      #
#                                                                               #
# The above copyright notice and this permission notice shall be included in all#
# copies or substantial portions of the Software.                               #
#                                                                               #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE #
# SOFTWARE.                                                                     #
#################################################################################

import argparse
import sys
import os
from glob import glob
from datetime import datetime
from src import *

def main():
    parser = argparse.ArgumentParser(prefix_chars='-+')
    parser.add_argument('inputFolder', type=str, nargs='?', default='')
    parser.add_argument('-hide', action='store_false', default=True)
    parser.add_argument('+hide', action='store_true', default=False)
    parser.add_argument('--duplicate-factor', type=float,
                        metavar='Dup-Factor', default=0.3)

    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    if args.hide: hide_files()
    else: unhide_files()

    if not args.inputFolder:
        sys.exit(0)

    print('Randomizing files in {}'.format(args.inputFolder))
    output_folder = os.path.join(OUTPUT_DIR,
                                '{}_{}'.format(HIDDEN_DATA[1:], timestamp))

    randomize(timestamp, args.inputFolder, args.duplicate_factor, output_folder)
    hide_files()

if __name__ == '__main__':
    main()
