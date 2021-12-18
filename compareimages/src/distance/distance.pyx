import numpy as np
from numpy import float64
from numpy cimport float64_t
cimport numpy as np
cimport cython

__all__ = ['get_distance_matrix']

@cython.boundscheck(False)
def get_distance_matrix(int[:,::1] arr1, int[:,::1] arr2):

  cdef double d = 0;
  cdef int N1 = arr1.shape[0]
  cdef int N2 = arr2.shape[0]
  cdef int data_dims = arr1.shape[1]
  cdef int i, j, k

  empty_array = np.empty(shape=(N1, N2), dtype=float64)
  cdef double[:, :] distance_matrix = empty_array


  for i in range(N1):
    for j in range(N2):
      d = 0
      for k in range(data_dims):
        d += (arr1[i,k] - arr2[j,k]) ** 2
      distance_matrix[i,j] = d

  return np.sqrt(np.asarray(distance_matrix, dtype=float64))
