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

import os
from sys import platform
from glob import glob
from .randomize import OUTPUT_DIR, HIDDEN_DATA


def hide_files():

    for csv in os.listdir(OUTPUT_DIR):
                if platform == 'darwin':
                    if (csv.startswith(HIDDEN_DATA[1:])
                        and csv.endswith('.csv')):
                            filename = csv.split(os.path.sep)[-1]
                            os.rename(os.path.join(OUTPUT_DIR, csv),
                                      os.path.join(OUTPUT_DIR,"."+filename))
                elif platform == 'win32':
                    if csv.endswith('.csv'):
                        os.popen('attrib +h ' + os.path.join(OUTPUT_DIR, csv))


def unhide_files():

    for csv in os.listdir(OUTPUT_DIR):
            if platform == 'darwin':
                    if (csv.startswith(HIDDEN_DATA) and csv.endswith('.csv')):
                            filename = csv.split(os.path.sep)[-1]
                            os.rename(os.path.join(OUTPUT_DIR, csv),
                                      os.path.join(OUTPUT_DIR, filename[1:]))

            elif platform == 'win32':
                    if csv.endswith('.csv'):
                        os.popen('attrib -h ' + os.path.join(OUTPUT_DIR, csv))
