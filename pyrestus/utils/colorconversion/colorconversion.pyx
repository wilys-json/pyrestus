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

import numpy as np
import cv2
from numpy import uint8
from numpy cimport uint8_t
cimport numpy as np
cimport cython

__all__ = ['convert_color']

COLOR_CONVERSION = {
    'RGB' : (lambda x : x),
    'YBR_FULL_422' : (
        lambda x: cv2.cvtColor(x, cv2.COLOR_YUV2RGB)
    )
}

@cython.boundscheck(False)
def convert_color(np.ndarray[int, ndim=4, mode="c"] img,
                  tuple coordinates,
                  str mode):

  cdef int frame = img.shape[0]
  cdef channels = img.shape[3]
  cdef int X0 = coordinates[0]
  cdef int Y0 = coordinates[1]
  cdef int X1 = coordinates[2]
  cdef int Y1 = coordinates[3]
  cdef int h = Y1-Y0
  cdef int w = X1-X0

  data = np.empty(shape=(frame, h, w, channels), dtype=uint8)

  for i in range(frame):
    data[i] = COLOR_CONVERSION[mode](img[i][Y0:Y1, X0:X1])

  return data
