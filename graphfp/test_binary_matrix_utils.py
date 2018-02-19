import numpy as np

from .binary_matrix_utils import (binary_matrix_to_sparse_rows,
                                  sparse_binary_transpose,
                                  sparse_rows_to_binary_matrix,
                                  to_sparse_format)

dct = {0: [0, 1, 2],
       1: [0, 1, 3],
       2: [0, 3],
       3: [1, 2]}

matrix = np.matrix([[1, 1, 1, 0],
                    [1, 1, 0, 1],
                    [1, 0, 0, 1],
                    [0, 1, 1, 0]])

"""
The functions are called outside because the results are shared. The functions
are intended to be internally consistent with one another.
"""

rows_dct, cols_dct = to_sparse_format(dct)
rows_mat, cols_mat = binary_matrix_to_sparse_rows(matrix)
binmat = sparse_rows_to_binary_matrix(rows_mat, cols_mat, matrix.shape)
rows_T, cols_T = sparse_binary_transpose(rows_mat, cols_mat)


def test_to_sparse_format():
    assert np.allclose(rows_dct, np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, ]))
    assert np.allclose(cols_dct, np.array([0, 1, 2, 0, 1, 3, 0, 3, 1, 2, ]))


def test_binary_matrix_to_sparse_rows():
    assert np.allclose(rows_dct, rows_mat)
    assert np.allclose(cols_dct, cols_mat)


def test_sparse_rows_to_binary_matrix():
    assert np.allclose(binmat, matrix)


def test_sparse_binary_transpose():
    assert np.allclose(binmat.T,
                       sparse_rows_to_binary_matrix(rows_T,
                                                    cols_T,
                                                    matrix.shape))
