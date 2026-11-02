import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# L'opzione comment='#' ignora le righe di intestazione/commento nel file
file_path = 'ExoplanetsPars_2025.csv'
try:
  df = pd.read_csv(file_path, comment='#')
  print('File letto con successo!')
except FileNotFoundError:
  print(
      f"Attenzione: Il file '{file_path}' non è stato trovato nella directory"
      ' corrente.'
  )
  df = pd.DataFrame(columns=['pl_orbper', 'pl_bmassj'])

print('\n--- NOMI DELLE COLONNE ---')
print(df.columns.tolist())

print('\n--- ESTRATTO DEL DATAFRAME (Prime 5 righe) ---')
print(df.head())


plt.figure(figsize=(10, 6))

solar_system = {
    'name': [ 'Mercurio', 'Venere', 'Terra', 'Marte', 'Giove', *['Saturno', 'Urano', 'Nettuno'],],
    'pl_orbper': [88.0, 224.7, 365.25, 687.0, 4332.6, 10759.0, 30687.0, 60190.0],
    'pl_bmassj': [ 0.000174,0.00315, 0.00315, 0.000338, 1.000, 0.299, 0.0457, 0.0537],
}
df_solar = pd.DataFrame(solar_system)

plt.scatter( df['pl_orbper'], df['pl_bmassj'], alpha=0.5, color='blue', label='Pianeti Extrasolari', s=15)
plt.scatter( df_solar['pl_orbper'], df_solar['pl_bmassj'], color='red', marker='o', s=18, label='Sistema Solare', zorder=5)

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Periodo Orbitale [giorni] (pl_orbper)', fontsize=12)
plt.ylabel('Massa del Pianeta [Masse di Giove] (pl_bmassj)', fontsize=12)
plt.title( 'Massa vs Periodo Orbitale: Esopianeti vs Sistema Solare', fontsize=14)
plt.legend(loc='lower right', frameon=True)

plt.tight_layout()
plt.show()

df['r2_div_mstellar'] = (df['pl_orbsmax'] ** 2) / df['st_mass']

solar_system['pl_orbsmax'] = [0.387, 0.723, 1.0, 1.524, 5.204, 9.582, 19.201, 30.047]
solar_system['st_mass'] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

df_solar = pd.DataFrame(solar_system)
df_solar['r2_div_mstellar'] = (df_solar['pl_orbsmax'] ** 2) / df_solar['st_mass']

plt.figure(figsize=(10, 6))

plt.scatter( df['pl_orbper'], df['r2_div_mstellar'], alpha=0.5, color='yellow', label='Pianeti Extrasolari', s=15)
plt.scatter( df_solar['pl_orbper'], df_solar['r2_div_mstellar'], color='red', marker='o', s=18, label='Sistema Solare', zorder=5)

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Periodo Orbitale [giorni] (pl_orbper)', fontsize=12)
plt.ylabel(r'$R_{max}^2 / m^*$ [AU$^2 / M_\odot$]', fontsize=12)
plt.title(r'Grandezza $R_{max}^2 / m^*$ vs Periodo Orbitale: Esopianeti vs Sistema Solare', fontsize=14)

plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()

df_transit = df.loc[df['discoverymethod'] == 'Transit']
df_rv = df.loc[df['discoverymethod'] == 'Radial Velocity']

plt.figure(figsize=(10, 6))
plt.scatter(df_transit['pl_orbper'], df_transit['pl_bmassj'], alpha=0.5, color='blue', label='Transit', s=15)
plt.scatter(df_rv['pl_orbper'], df_rv['pl_bmassj'], alpha=0.5, color='orange', label='Radial Velocity', s=15)
# Inseriamo il Sistema Solare anche qui come riferimento!
plt.scatter(df_solar['pl_orbper'], df_solar['pl_bmassj'], color='red', marker='o', s=25, label='Sistema Solare', zorder=5)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Periodo Orbitale [giorni] (pl_orbper)', fontsize=12)
plt.ylabel('Massa del Pianeta [Masse di Giove] (pl_bmassj)', fontsize=12)
plt.title('Massa vs Periodo Orbitale per Metodo di Scoperta + Sistema Solare', fontsize=14)
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
mass_transit = df_transit['pl_bmassj'].dropna()
mass_rv = df_rv['pl_bmassj'].dropna()

if not mass_transit.empty and not mass_rv.empty:
    bins_mass = np.logspace(np.log10(max(mass_transit.min(), 1e-5)), np.log10(mass_transit.max()), 50)
    plt.hist(mass_transit, bins=bins_mass, alpha=0.5, color='blue', label='Transit')
    plt.hist(mass_rv, bins=bins_mass, alpha=0.5, color='orange', label='Radial Velocity')

plt.xscale('log')
plt.xlabel('Massa del Pianeta [Masse di Giove]', fontsize=12)
plt.ylabel('Numero di Esopianeti', fontsize=12)
plt.title('Istogramma Sovrapposto delle Masse (Scala Logaritmica)', fontsize=14)
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
log_mass_transit = np.log10(mass_transit[mass_transit > 0])
log_mass_rv = np.log10(mass_rv[mass_rv > 0])

if not log_mass_transit.empty and not log_mass_rv.empty:
    bins_log = np.linspace(
        min(log_mass_transit.min(), log_mass_rv.min()), 
        max(log_mass_transit.max(), log_mass_rv.max()), 
        50
    )
    plt.hist(log_mass_transit, bins=bins_log, alpha=0.5, color='blue', label='Transit')
    plt.hist(log_mass_rv, bins=bins_log, alpha=0.5, color='orange', label='Radial Velocity')

plt.xlabel(r'$\log_{10}$ (Massa del Pianeta [Masse di Giove])', fontsize=12)
plt.ylabel('Numero di Esopianeti', fontsize=12)
plt.title(r'Istogramma Sovrapposto del $\log_{10}$ della Massa', fontsize=14)
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.show()