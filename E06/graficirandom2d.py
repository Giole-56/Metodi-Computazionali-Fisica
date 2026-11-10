import matplotlib.pyplot as plt
import numpy as np
from randomwalk import random_walk_2d

s = 1.0  
N = 100  
num_walkers = 5  

walkers_x = []
walkers_y = []
r_squared_list = []

for i in range(num_walkers):
  x, y = random_walk_2d(s, N)
  walkers_x.append(x)
  walkers_y.append(y)

  r_squared = x**2 + y**2
  r_squared_list.append(r_squared)

passi = np.arange(N + 1)  

# Grafico 2D delle posizioni
plt.figure(figsize=(8, 8))

for i in range(num_walkers):
  plt.plot(
      walkers_x[i],
      walkers_y[i],
      label=f"Walker {i+1}",
      alpha=0.8,
      linewidth=1.2,
  )

  plt.plot(walkers_x[i][-1], walkers_y[i][-1], "o")

plt.plot(0, 0, "ko", markersize=8, label="Inizio (0,0)")

plt.title("Random Walk 2D per 5 Walker ", fontsize=14)
plt.xlabel("Posizione X", fontsize=12)
plt.ylabel("Posizione Y", fontsize=12)
plt.axis("equal")  
plt.legend()
plt.show()

# Grafico a due pannelli (Subplot)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for i in range(num_walkers):
  axes[0].plot(
      walkers_x[i],
      walkers_y[i],
      label=f"Walker {i+1}",
      alpha=0.8,
      linewidth=1.2,
  )
axes[0].plot(0, 0, "ko", markersize=8, label="Inizio (0,0)")
axes[0].set_title("A) Posizioni 2D dei Random Walker", fontsize=13)
axes[0].set_xlabel("Posizione X", fontsize=11)
axes[0].set_ylabel("Posizione Y", fontsize=11)
axes[0].axis("equal")
axes[0].legend()

for i in range(num_walkers):
  axes[1].plot(
      passi,
      r_squared_list[i],
      label=f"Walker {i+1}",
      alpha=0.8,
      linewidth=1.2,
  )

axes[1].set_title(
    "B) Quadrato della distanza $r^2$ vs Numero di passi", fontsize=13
)
axes[1].set_xlabel("Numero di passi ($N$)", fontsize=11)
axes[1].set_ylabel("Distanza al quadrato ($r^2$)", fontsize=11)
axes[1].legend()

plt.tight_layout()
plt.show()