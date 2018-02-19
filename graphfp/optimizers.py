import autograd.numpy as np
from pyflatten import flatten


def sgd(gradfunc, wb, callback=None, num_iters=200,
        step_size=0.1, mass=0.9, adaptive=False):
    """
    Batch stochastic gradient descent with momentum.

    Todo:
    - Refactor to make this follow the SGD signature and code in the autograd
      examples.
    """
    wb_vect, wb_unflattener = flatten(wb)
    velocity = np.zeros(len(wb_vect))

    for i in range(num_iters):
        g = gradfunc(wb_vect, wb_unflattener)
        velocity = mass * velocity - (1.0 - mass) * g
        wb_vect += step_size * velocity

        if adaptive:
            step_size = step_size * (1 - step_size)

        wb = wb_unflattener(wb_vect)
        if callback:
            callback(wb, i)

    return wb_vect, wb_unflattener


def adam(grad, wb, callback=None, num_iters=100,
         step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""

    wb_vect, wb_unflattener = flatten(wb)

    m = np.zeros(len(wb_vect))
    v = np.zeros(len(wb_vect))

    for i in range(num_iters):
        g = grad(wb_vect, wb_unflattener)
        if callback:
            callback(wb_unflattener(wb_vect), i)
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        wb_vect -= step_size*mhat/(np.sqrt(vhat) + eps)

    return wb_vect, wb_unflattener
