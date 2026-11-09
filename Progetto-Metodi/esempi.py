import numpy as np
from progetto import Stella
from progetto import Sensore

# Creazione e ordinamento dell'array di stelle tramite NumPy
elenco_stelle = [
    Stella("Sirio", 9940.0),
    Stella("Betelgeuse", 3500.0),
    Stella("Sole", 5750.0),
]

array_stelle = np.array(elenco_stelle)
stelle_ordinate = np.sort(array_stelle)

print("--- STELLE ORDINATE PER TEMPERATURA ---")
for s in stelle_ordinate:
    print(s)

# Configurazione e test del sensore
mio_sensore = Sensore(nome="Sensore Ottico", area=0.5, sensibilita=0.9, tempo_esposizione=1.0)
print("\n--- TEST DEL SENSORE ---")
print(mio_sensore)

flusso_incidente = 20000.0  # fotoni / (s * m^2)
segnale_medio = mio_sensore.calcola_conteggi(flusso_incidente)
misura_effettiva = mio_sensore.simula_misura(flusso_incidente)

print(f"Segnale medio teorico: {segnale_medio:.1f} fotoni")
print(f"Misura simulata: {misura_effettiva} conteggi")