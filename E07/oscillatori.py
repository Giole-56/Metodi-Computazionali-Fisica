import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

k = 0.1       
m = 1.0       

def V_3_2(x):
    return k * np.abs(x)**(3/2)

def V_2(x):
    return k * x**2

def V_4(x):
    return k * x**4

def V_6(x):
    return k * x**6

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

x0_values = np.linspace(0.3, 4.5, 100)

T_3_2 = calcola_periodo(V_3_2, x0_values, m, k)
T_2   = calcola_periodo(V_2,   x0_values, m, k)
T_4   = calcola_periodo(V_4,   x0_values, m, k)
T_6   = calcola_periodo(V_6,   x0_values, m, k)

plt.figure(figsize=(10, 6))

plt.plot(x0_values, T_3_2, color='r', linewidth=2.5, label=r'$V(x) = k |x|^{3/2}$ ($T \propto x_0^{1/4}$)')
plt.plot(x0_values, T_2,   color='y', linewidth=2.5, label=r'$V(x) = k x^2$ ($T = \mathrm{costante}$)')
plt.plot(x0_values, T_4,   color='royalblue',  linewidth=2.5, label=r'$V(x) = k x^4$ ($T \propto x_0^{-1}$)')
plt.plot(x0_values, T_6,   color='g',     linewidth=2.5, label=r'$V(x) = k x^6$ ($T \propto x_0^{-2}$)')

plt.title(r"Confronto del periodo $T(x_0)$ per diversi potenziali simmetrici", fontsize=13)
plt.xlabel(r"Punto di partenza $x_0$", fontsize=11)
plt.ylabel(r"Periodo $T$", fontsize=11)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()