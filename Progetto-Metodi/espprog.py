from pathlib import Path
import numpy as np
from proveprogetto import Atmosfera, Sensore, Stella


def analizza_stelle_configurazioni_e_angoli(
    stelle: list[Stella],
    atm: Atmosfera,
    configurazioni: list[dict],
    sensore: Sensore,
    lambdas: np.ndarray,
    angoli_deg: list[float],
    distanza_orbita_m: float | None = None,
    seed: int = 42,
) -> list[dict]:
    """
    Simula lo spettro stellare, l'attraversamento dell'atmosfera per diverse
    configurazioni (Rayleigh, O3, colonna O3, temperatura O3) e angoli zenitali,
    calcolando anche la risposta statistica del sensore.
    """
    rng = np.random.default_rng(seed)
    risultati = []

    for stella in stelle:
        if distanza_orbita_m is not None:
            d0 = stella.densita_fotoni_a_distanza(
                lambdas,
                distanza_orbita_m=distanza_orbita_m,
            )
        else:
            d0 = stella.densita_fotoni(lambdas)

        for cfg in configurazioni:
            atm.configura(
                rayleigh=cfg.get("rayleigh"),
                ozono=cfg.get("ozono"),
                colonna_ozono=cfg.get("colonna_ozono"),
                temperatura_ozono=cfg.get("temperatura_ozono"),
            )

            for theta_deg in angoli_deg:
                theta_rad = np.deg2rad(theta_deg)
                d_obs = atm.osserva_spettro(lambdas, d0, theta_rad=theta_rad)
                n_atteso = sensore.calcola_conteggi_da_spettro(lambdas, d_obs)
                k_misurato = sensore._campiona_rumore(n_atteso, rng=rng)
                snr = sensore.snr(n_atteso)

                risultati.append(
                    {
                        "stella": stella.nome,
                        "T_K": stella.temperatura,
                        "config": cfg["nome"],
                        "rayleigh": atm.abilita_rayleigh,
                        "ozono": atm.abilita_ozono,
                        "T_O3_K": atm.temperatura_ozono,
                        "colonna_O3": atm.colonna_ozono,
                        "theta_deg": theta_deg,
                        "S_km": atm.spessore_massa_aria(theta_rad) / 1000.0,
                        "N_atteso": n_atteso,
                        "K_misurato": k_misurato,
                        "diff_rel": (k_misurato - n_atteso) / n_atteso if n_atteso > 0 else 0.0,
                        "SNR": snr,
                    }
                )

    return risultati


if __name__ == "__main__":
    file_ozono = Path("data/SCIA O3 Temp cross-section V4.1.DAT")
    ha_ozono = file_ozono.exists()

    stelle_notevoli = [
        Stella("Proxima Centauri (M5.5V)", 3040.0, raggio_m=0.154 * Stella.R_SOLE),
        Stella("Sole (G2V)", 5778.0, raggio_m=1.0 * Stella.R_SOLE),
        Stella("Sirio A (A1V)", 9940.0, raggio_m=1.711 * Stella.R_SOLE),
        Stella("Rigel (B8Ia)", 12100.0, raggio_m=78.9 * Stella.R_SOLE),
    ]

    atm = Atmosfera(
        abilita_rayleigh=True,
        abilita_ozono=ha_ozono,
        colonna_ozono=8.0e22,
        temperatura_ozono=223.0,
        file_ozono=str(file_ozono) if ha_ozono else None,
        unita_lambda_file="nm",
        unita_sigma_file="cm2",
    )

    if not ha_ozono:
        print(f"Attenzione: file '{file_ozono}' non trovato. Le configurazioni con O3 non useranno dati da file.")

    # Configurazioni atmosferiche da confrontare
    configurazioni_atm = [
        {
            "nome": "Solo Rayleigh",
            "rayleigh": True,
            "ozono": False,
            "colonna_ozono": 8.0e22,
            "temperatura_ozono": 223.0,
        },
        {
            "nome": "Rayleigh + O3 (203 K)",
            "rayleigh": True,
            "ozono": ha_ozono,
            "colonna_ozono": 8.0e22,
            "temperatura_ozono": 203.0,
        },
        {
            "nome": "Rayleigh + O3 (273 K)",
            "rayleigh": True,
            "ozono": ha_ozono,
            "colonna_ozono": 8.0e22,
            "temperatura_ozono": 273.0,
        },
        {
            "nome": "Solo O3 (223 K, alta colonna)",
            "rayleigh": False,
            "ozono": ha_ozono,
            "colonna_ozono": 1.4e23,
            "temperatura_ozono": 223.0,
        },
    ]

    sensore = Sensore("Fotodiodo UV-Vis", area=0.1, sensibilita=0.7, tempo_esposizione=1.0)

    if atm._lambda_min_ozono is not None and atm._lambda_max_ozono is not None:
        l_min = max(250e-9, atm._lambda_min_ozono)
        l_max = min(1000e-9, atm._lambda_max_ozono)
        grid_wl = np.linspace(l_min, l_max, 1500)
    else:
        grid_wl = np.linspace(250e-9, 1000e-9, 1500)

    angoli = [0.0, 60.0, 85.0]

    tabella = analizza_stelle_configurazioni_e_angoli(
        stelle=stelle_notevoli,
        atm=atm,
        configurazioni=configurazioni_atm,
        sensore=sensore,
        lambdas=grid_wl,
        angoli_deg=angoli,
        distanza_orbita_m=1.0 * Stella.UA,
        seed=42,
    )

    for r in tabella:
        print(
            f"{r['stella']:27s} | {r['config']:28s} | θ={r['theta_deg']:4.1f}° | "
            f"N̄={r['N_atteso']:.3e} | K={r['K_misurato']:.3e} | SNR={r['SNR']:.2e}"
        )