import numpy
import pyximport
pyximport.install(setup_args={"include_dirs": numpy.get_include()},
                  reload_support=True)

from binary_dot import csr_binary_dot_left, csr_binary_dot_right
import binary_matrix_utils as bmu
import autograd.numpy as np
from autograd.util import check_grads

binmat = np.array([[1, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0]],
                  dtype=int)

rows, cols = bmu.binary_matrix_to_sparse_rows(binmat)

feats = np.array([[1., 2., 3.],
                  [4., 5., 6.],
                  [7., 8., 9.]])


def test_csr_binary_dot_left():
    dotleft_cython = csr_binary_dot_left(feats, rows, cols)
    dotleft_numpy = np.dot(binmat, feats)

    assert np.allclose(dotleft_cython, dotleft_numpy)


def test_csr_binary_dot_right():
    dotright_cython = csr_binary_dot_right(feats, rows, cols)
    dotright_numpy = np.dot(feats, binmat)

    assert np.allclose(dotright_cython, dotright_numpy)


def test_gradient_csr_binary_dot_left():
    """
    Checks that the gradient is computed correctly.
    """
    def sum_csr_binary_dot_left(feats):
        result = csr_binary_dot_left(feats, rows, cols)
        return np.sum(result)

    # gradfunc = grad(sum_csr_binary_dot_left)
    check_grads(sum_csr_binary_dot_left, feats)


def test_gradient_csr_binary_dot_right():
    def sum_csr_binary_dot_right(feats):
        result = csr_binary_dot_right(feats, rows, cols)
        return np.sum(result)

    check_grads(sum_csr_binary_dot_right, feats)
