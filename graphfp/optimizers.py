from .flatten import flatten
import autograd.numpy as np


def sgd(gradfunc, wb, layers, graphs, callback=None, num_iters=200,
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
