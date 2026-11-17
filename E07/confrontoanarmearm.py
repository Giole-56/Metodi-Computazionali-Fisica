import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

k = 0.1       
m = 3.0       

def V_anarm(x):
    return k * x**6

def V_harm(x):
    return k * x**2

def calcola_periodo(V_func, x0_vals, m, k, N_punti=1000):
   
    T_vals = []
    
    for x0 in x0_vals:
        if x0 == 0:
            T_vals.append(0.0)
            continue
            
        x = np.linspace(0, x0 * (1 - 1e-7), N_punti)
        
        V_x0 = V_func(x0)
        V_x = V_func(x)
        
        differenza = V_x0 - V_x
        integrand = 1.0 / np.sqrt(differenza)
        
        integrale = simpson(integrand, x=x)
        
        T = np.sqrt(8 * m) * integrale
        T_vals.append(T)
        
    return np.array(T_vals)

x0_values = np.linspace(0.1, 4.5, 100)

T_anarm = calcola_periodo(V_anarm, x0_values, m, k)
T_harm = calcola_periodo(V_harm, x0_values, m, k)

plt.figure(figsize=(8, 5))
plt.plot(x0_values, T_anarm, color='r', linewidth=2.5, label=r'Anarmonico: $V(x) = k x^6$')
plt.plot(x0_values, T_harm, color='b', linewidth=2.5, label=r'Armonico: $V_a(x) = k x^2$')

plt.title(r"Confronto del periodo", fontsize=12)
plt.xlabel(r"Punto di partenza $x_0$", fontsize=11)
plt.ylabel(r"Periodo $T$", fontsize=11)
plt.legend(fontsize=11)
plt.show()