import matplotlib.pyplot as plt
import numpy as np


def f(x):
  return (3 / 1000) * (x**2)


x_min, x_max = 0, 10
y_max = f(x_max)

N_samples = 100000

x_rand = np.random.uniform(x_min, x_max, N_samples)
y_rand = np.random.uniform(0, y_max, N_samples)

accepted_mask = y_rand <= f(x_rand)

accepted_samples = x_rand[accepted_mask][:N_samples]

plt.figure(figsize=(10, 6))

plt.hist(
    accepted_samples,
    bins=50,
    density=True,
    alpha=0.6,
    color="skyblue",
    edgecolor="black",
    label="Campioni (Hit or Miss)",
)

x_vals = np.linspace(x_min, x_max, 500)
plt.plot(
    x_vals, f(x_vals), "r-", linewidth=2.5, label=r"Teorica: $f(x) = 3/1000 x^2$"
)

plt.title("Metodo Hit or Miss", fontsize=14)
plt.xlabel("$x$", fontsize=12)
plt.ylabel("Densità di probabilità", fontsize=12)
plt.legend(fontsize=12)
plt.show()

r = np.random.uniform(0, 1, N_samples)

x_cumul = x_max * (r ** (1 / 3))

plt.figure(figsize=(10, 6))

plt.hist(
    x_cumul,
    bins=50,
    density=True,
    alpha=0.6,
    color="lightgreen",
    edgecolor="black",
    label="Campioni (Metodo della Cumulativa)",
)

x_vals = np.linspace(x_min, x_max, 500)
plt.plot(
    x_vals, f(x_vals), "r-", linewidth=2.5, label=r"Teorica: $f(x) = 3/1000 x^2$"
)

plt.title(
    "Generazione tramite Metodo della Cumulativa", fontsize=14
)
plt.xlabel("$x$", fontsize=12)
plt.ylabel("Densità di probabilità", fontsize=12)
plt.legend(fontsize=12)
plt.show()