import pandas as pd
import matplotlib.pyplot as plt

file_path = 'kplr010666592-2011240104155_slc.csv'
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

time_col = 'TIME'  
flux_col = 'PDCSAP_FLUX'
err_col = 'PDCSAP_FLUX_ERR'

plt.figure(figsize=(12, 5))
plt.plot(df[time_col], df[flux_col], color='blue', linewidth=2, alpha=1)

plt.title('Curva di luce (Flusso in funzione del tempo)', fontsize=14)
plt.xlabel('Tempo(s)', fontsize=12)
plt.ylabel('Flusso(e-/s)', fontsize=12)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(df[time_col], df[flux_col],'o', color='blue', markersize=2)

plt.title('Curva di luce (Flusso in funzione del tempo)', fontsize=14)
plt.xlabel('Tempo(s)', fontsize=12)
plt.ylabel('Flusso(e-/s)', fontsize=12)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))

plt.errorbar(df[time_col],  df[flux_col],  yerr=df[err_col], fmt='o', color='purple', ecolor='purple', markersize=2,)

plt.title('Curva di luce con barre di errore', fontsize=14)
plt.xlabel('Tempo(s)', fontsize=12)
plt.ylabel('Flusso(e-/s)', fontsize=12)

plt.tight_layout()
plt.savefig('curva_di_luce_errori.png', dpi=300)  
plt.savefig('curva_di_luce_errori.pdf')           

plt.show()

t_start = 954.5 
t_end = 955.0

df_zoom = df.loc[(df[time_col] >= t_start) & (df[time_col] <= t_end)]

plt.figure(figsize=(10, 5))

plt.errorbar( df_zoom[time_col], df_zoom[flux_col],fmt='o', yerr=df_zoom[err_col], color='green', ecolor='green', markersize=2,)

plt.title(f'Zoom localizzato (Tempo: da {t_start} a {t_end})', fontsize=14)
plt.xlabel('Tempo(s)', fontsize=12)
plt.ylabel('Flusso(e-/s)', fontsize=12)

plt.tight_layout()
plt.savefig('minimo_zoom_errori.png', dpi=300)
plt.savefig('minimo_zoom_errori.pdf')

plt.show()

fig, ax = plt.subplots(figsize=(12, 5))

ax.errorbar(df[time_col], df[flux_col], yerr=df[err_col], fmt='o', color='blue', ecolor='blue', markersize=2, alpha=1)

ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, 1.030e6 )

axins = ax.inset_axes([0.70, 0.60, 0.25, 0.35])

axins.errorbar(df_zoom[time_col], df_zoom[flux_col], yerr=df_zoom[err_col], fmt='o', color='royalblue', ecolor='royalblue', markersize=2, alpha=0.8)

axins.set_xlim(t_start, t_end)
axins.set_title('Zoom localizzato', fontsize=10)

ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.7)

ax.set_title('Curva di luce con zoom localizzato', fontsize=14)
ax.set_xlabel('Tempo(s)', fontsize=12)
ax.set_ylabel('Flusso(e-/s)', fontsize=12)

plt.tight_layout()
plt.savefig('curva_di_luce_con_inset.png', dpi=300)
plt.savefig('curva_di_luce_con_inset.pdf')

plt.show()
