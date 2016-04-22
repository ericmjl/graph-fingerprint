import autograd.numpy as np
from autograd.scipy.misc import logsumexp


def relu(x):
    """
    Rectified Linear Unit
    """
    return x * (x > 0)


def softmax(x, axis=0):
    """
    The softmax function normalizes everything to between 0 and 1.
    """
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


def tanh(x):
    """
    tanh non-linearity
    """
    return np.tanh(x)


def logistic(x):
    """
    logistic nonlinearity
    """
    return np.log(1 / (1 + np.exp(x)))
