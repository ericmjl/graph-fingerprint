import numpy as np
cimport numpy as np
from cython import boundscheck, nonecheck, wraparound
from autograd.core import primitive
from binary_matrix_utils import sparse_binary_transpose


@primitive
def csr_binary_dot_left(inputs, rows, cols):
    """
    The binary matrix is on the left of the dot product.
    """
    out = np.zeros_like(inputs)
    _csr_binary_dot_left(rows, cols, inputs, out)
    return out


@nonecheck(False)
@wraparound(False)
@boundscheck(False)
cdef inline void _csr_binary_dot_left(int[::1] rows,
                                      int[::1] cols,
                                      double[:,::1] inputs,
                                      double[:,::1] out):
    cdef int idx, i, j, k
    for idx in range(rows.shape[0]):
        i = rows[idx]
        k = cols[idx]
        for j in range(inputs.shape[1]):
            out[i, j] += inputs[k, j]


@primitive
def csr_binary_dot_right(inputs, rows, cols):
    """
    The binary matrix is on the right of the dot product.
    """
    out = np.zeros_like(inputs)
    _csr_binary_dot_right(rows, cols, inputs, out)
    return out


@nonecheck(False)
@wraparound(False)
@boundscheck(False)
cdef inline void _csr_binary_dot_right(int[:] rows,
                                       int[:] cols,
                                       double[:,::1] inputs,
                                       double[:,::1] out):

    cdef int i, j, k, idx
    for idx in range(cols.shape[0]):
        j = cols[idx]
        i = rows[idx]
        for k in range(inputs.shape[0]):
            out[k, j] += inputs[k, i]


def make_grad_csr_binary_dot_left(ans, inputs, rows, cols):
    """
    Makes the gradient of csr_binary_dot_left.
    """
    rowsT, colsT = sparse_binary_transpose(rows, cols)
    def gradient_product(g):
        return csr_binary_dot_right(g, rowsT, colsT)
    return gradient_product

csr_binary_dot_left.defgrad(make_grad_csr_binary_dot_left, argnum=0)


def make_grad_csr_binary_dot_right(ans, inputs, rows, cols):
    """
    Makes the gradient of csr_binary_dot_right.
    """
    def gradient_product(g):
        return csr_binary_dot_right(g, rows, cols)
    return gradient_product

csr_binary_dot_right.defgrad(make_grad_csr_binary_dot_right, argnum=0)
