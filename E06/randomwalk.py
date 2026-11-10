import numpy as np


def random_walk_2d(s, N):
  
  phi = np.random.uniform(0, 2 * np.pi, N)

  dx = s * np.cos(phi)
  dy = s * np.sin(phi)

  x = np.cumsum(np.insert(dx, 0, 0))
  y = np.cumsum(np.insert(dy, 0, 0))

  return x, y