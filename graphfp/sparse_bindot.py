from autograd.core import primitive


@primitive
def bindot_left(inputs, binmat):
    """
    The binary matrix is on the left of the dot product.
    """
    out = binmat @ inputs
    return out


@primitive
def bindot_right(inputs, binmat):
    """
    The binary matrix is on the right of the dot product.
    """
    out = inputs @ binmat
    return out


def make_grad_bindot_left(ans, inputs, binmat):
    """
    Makes the gradient of bindot_left.
    """
    def gradient_product(g):
        return bindot_right(g, binmat.T)
    return gradient_product

bindot_left.defgrad(make_grad_bindot_left, argnum=0)


def make_grad_bindot_right(ans, inputs, binmat):
    """
    Makes the gradient of bindot_right.
    """
    def gradient_product(g):
        return bindot_right(g, binmat)
    return gradient_product


bindot_right.defgrad(make_grad_bindot_right, argnum=0)
