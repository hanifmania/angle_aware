import numpy as np

a = np.array([1, 2])
b = np.arange(6).reshape(-1, 2)
dist = np.linalg.norm(a - b, axis=1)
ret = dist < 2
print(b)
print(dist)
print(ret)
print(b[ret])
