import operator as op
import numpy as np


def to_sparse_format(dct):
    rows, cols = zip(*sorted(dct.items(), key=op.itemgetter(0)))
    rows = np.repeat(rows, list(map(len, cols)))
    cols = np.concatenate(cols)
    return rows.astype('int32'), cols.astype('int32')


def sparse_binary_transpose(rows, cols):
    """
    Transposes the sparse binary matrix.

    Parameters:
    ===========
    - rows, cols: `numpy` arrays of dimension (1, n)
    """
    new_cols = rows[rows.argsort()]
    new_rows = cols[rows.argsort()]
    return new_rows.astype('int32'), new_cols.astype('int32')


def binary_matrix_to_sparse_rows(matrix):
    """
    Converts a binary sparse matrix into rows and columns. Automatically casts
    as int32, as there is no need for storing them as anything more complex.
    """
    n_rows, n_cols = matrix.shape

    rows = []
    cols = []

    for row in range(n_rows):
        for col in range(n_cols):
            if matrix[row, col] == 1:
                rows.append(row)
                cols.append(col)
    return np.array(rows).astype('int32'), np.array(cols).astype('int32')


def sparse_rows_to_binary_matrix(rows, cols, shape):
    binmat = np.zeros(shape=shape)
    for row, col in zip(rows, cols):
        binmat[row, col] = 1
    return binmat.astype('int32')
