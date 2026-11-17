import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

parser = argparse.ArgumentParser(
    description="Calcola e confronta il periodo T in funzione di x_0 per diversi potenziali simmetrici."
)
parser.add_argument(
    '--potentials', 
    nargs='+', 
    choices=['3/2', '2', '4', '6', 'all'], 
    default=['all'],
    help="Scegli quali potenziali visualizzare (es. --potentials 2 6 oppure --potentials all). Default: all"
)
args = parser.parse_args()

k = 0.1      
m = 1.0       

potenziali_disponibili = {
    '3/2': {
        'func': lambda x: k * np.abs(x)**(3/2),
        'label': r'$V(x) = k |x|^{3/2}$ ($T \propto x_0^{1/4}$)',
        'color': 'g'
    },
    '2': {
        'func': lambda x: k * x**2,
        'label': r'$V(x) = k x^2$ ($T = \mathrm{costante}$)',
        'color': 'r'
    },
    '4': {
        'func': lambda x: k * x**4,
        'label': r'$V(x) = k x^4$ ($T \propto x_0^{-1}$)',
        'color': 'royalblue'
    },
    '6': {
        'func': lambda x: k * x**6,
        'label': r'$V(x) = k x^6$ ($T \propto x_0^{-2}$)',
        'color': 'y'
    }
}

if 'all' in args.potentials:
    scelte = ['3/2', '2', '4', '6']
else:
    scelte = args.potentials

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

plt.figure(figsize=(10, 6))

for p in scelte:
    dati = potenziali_disponibili[p]
    T_vals = calcola_periodo(dati['func'], x0_values, m, k)
    ls = dati.get('linestyle',)
    plt.plot(x0_values, T_vals, color=dati['color'], linewidth=2.5, linestyle=ls, label=dati['label'])

plt.title(r"Confronto del periodo $T(x_0)$ per i potenziali selezionati", fontsize=13)
plt.xlabel(r"Punto di partenza $x_0$", fontsize=11)
plt.ylabel(r"Periodo $T$", fontsize=11)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()