import jax.numpy as np
import numpy as onp
from jax import jit, device_put

from jax.debug import print


@jit
def test():
    a = np.ones((2, 3)) * 3
    b = np.arange(6).reshape((2, 3))
    c = np.maximum(3, b)
    print("a {c}", c=c)

    # a = device_put(a)
    # print(a)

    # a = onp.ones((1, 3))
    # b = np.ones_like(a)
    # print(a)


test()
