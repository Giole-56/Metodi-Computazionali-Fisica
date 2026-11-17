import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

k = 0.1       
m = 4.0       

def V(x):
    return k * x**6

def calcola_periodo(x0_vals, m, k, N_punti=1000):
    
    T_vals = []
    
    for x0 in x0_vals:
        if x0 == 0:
            T_vals.append(0.0)
            continue
            
        x = np.linspace(0, x0 * (1 - 1e-7), N_punti)
        
        V_x0 = V(x0)
        V_x = V(x)
        
        differenza = V_x0 - V_x
        integrand = 1.0 / np.sqrt(differenza)
        
        integrale = simpson(integrand, x=x)
        
        T = np.sqrt(8 * m) * integrale
        T_vals.append(T)
        
    return np.array(T_vals)

x0_values = np.linspace(0.1, 4.5, 100)
T_values = calcola_periodo(x0_values, m, k)

plt.figure(figsize=(8, 5))
plt.plot(x0_values, T_values, color='r', linewidth=2.5, label=r'$T(x_0)$')
plt.title(r"Periodo dell'oscillatore anarmonico in funzione di $x_0$", fontsize=12)
plt.xlabel(r"Punto di partenza $x_0$", fontsize=11)
plt.ylabel(r"Periodo $T$", fontsize=11)
plt.legend(fontsize=11)
plt.show()