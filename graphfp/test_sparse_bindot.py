from scipy.sparse import csr_matrix
from .sparse_bindot import bindot_left, bindot_right
import autograd.numpy as np
from autograd.util import check_grads

binmat = np.array([[1, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0]],
                  dtype=int)

sparse_binmat = csr_matrix(binmat)

feats = np.array([[1., 2., 3.],
                  [4., 5., 6.],
                  [7., 8., 9.]])


def test_gradient_bindot_left():
    """
    Checks that the gradient is computed correctly.
    """
    def sum_bindot_left(feats):
        result = bindot_left(feats, sparse_binmat)
        return np.sum(result)

    check_grads(sum_bindot_left, feats)


def test_gradient_bindot_right():
    """
    Checks that the gradient is computed correctly.
    """
    def sum_bindot_right(feats):
        result = bindot_right(feats, sparse_binmat)
        return np.sum(result)

    check_grads(sum_bindot_right, feats)
