from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

try:
    from progetto import Atmosfera, Sensore, Stella
except ImportError:
    from proveprogetto import Atmosfera, Sensore, Stella


def trova_file_ozono() -> Path | None:
    """Cerca il file della sezione d'urto dell'ozono anche se il nome o l'estensione variano."""
    base_script = Path(__file__).resolve().parent
    cwd = Path.cwd()
    cartelle = [
        base_script,
        base_script / "data",
        cwd,
        cwd / "data",
        base_script.parent / "data",
        cwd.parent / "data",
    ]
    nomi_esatti = [
        "SCIA O3 Temp cross-section V4.1.DAT",
        "SCIA O3 Temp cross-section V4.0.DAT",
    ]

    # 1. Cerca prima i nomi esatti
    for cartella in cartelle:
        if not cartella.exists() or not cartella.is_dir():
            continue
        for nome in nomi_esatti:
            p = cartella / nome
            if p.exists() and p.is_file():
                return p

    # 2. Cerca qualsiasi file .dat o .txt che contenga 'scia' o 'o3' (case-insensitive)
    for cartella in cartelle:
        if not cartella.exists() or not cartella.is_dir():
            continue
        for p in sorted(cartella.iterdir()):
            if p.is_file() and p.suffix.lower() in {".dat", ".txt", ".asc"}:
                nome_low = p.name.lower()
                if "scia" in nome_low or "o3" in nome_low:
                    return p

    # 3. Se ancora non trovato, mostra i file presenti per fare un controllo rapido
    print("File trovati nella cartella dello script:", [p.name for p in base_script.iterdir() if p.is_file()])
    return None


def analizza_stelle_sensori_configurazioni_e_angoli(
    stelle: list[Stella],
    sensori: list[Sensore],
    atm: Atmosfera,
    configurazioni: list[dict],
    lambdas: np.ndarray,
    angoli_deg: list[float] | np.ndarray,
    distanza_orbita_m: float | None = None,
    seed: int = 42,
) -> list[dict]:
    """
    Simula lo spettro stellare, l'attraversamento dell'atmosfera per diverse
    configurazioni e angoli zenitali, e la risposta statistica di diversi sensori.
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
                percorso_ozono=cfg.get("percorso_ozono"),
                temperatura_ozono=cfg.get("temperatura_ozono"),
            )

            for theta_deg in angoli_deg:
                theta_rad = np.deg2rad(theta_deg)
                d_obs = atm.osserva_spettro(lambdas, d0, theta_rad=theta_rad)

                for sensore in sensori:
                    n_atteso = sensore.calcola_conteggi_da_spettro(lambdas, d_obs)
                    k_misurato = sensore._campiona_rumore(n_atteso, rng=rng)
                    snr = sensore.snr(n_atteso)

                    risultati.append(
                        {
                            "stella": stella.nome,
                            "T_K": stella.temperatura,
                            "sensore": sensore.nome,
                            "config": cfg["nome"],
                            "theta_deg": float(theta_deg),
                            "S_km": atm.spessore_massa_aria(theta_rad) / 1000.0,
                            "d_obs": d_obs,
                            "N_atteso": n_atteso,
                            "K_misurato": k_misurato,
                            "SNR": snr,
                        }
                    )

    return risultati


def plotta_spettri_e_angoli(
    risultati: list[dict],
    lambdas: np.ndarray,
    nome_stella_spettri: str,
    nome_config_angoli: str,
    nome_sensore_angoli: str,
    angoli_spettri_deg: list[float] | None = None,
):
    lambdas_nm = lambdas * 1e9
    stili_config = ["-", "--", ":", "-."]

    # 1. Spettri osservati per la stella scelta
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    subset_spettri = [
        r for r in risultati
        if r["stella"] == nome_stella_spettri
        and (angoli_spettri_deg is None or r["theta_deg"] in angoli_spettri_deg)
    ]

    nomi_cfg = list(dict.fromkeys(r["config"] for r in subset_spettri))
    mappa_stile = {cfg: stili_config[i % len(stili_config)] for i, cfg in enumerate(nomi_cfg)}

    visti = set()
    for r in subset_spettri:
        chiave = (r["config"], r["theta_deg"])
        if chiave in visti:
            continue
        visti.add(chiave)
        ax.plot(
            lambdas_nm,
            np.maximum(r["d_obs"], 1e-30),
            linestyle=mappa_stile[r["config"]],
            linewidth=1.8,
            alpha=0.85,
            label=f"{r['config']} | θ={r['theta_deg']:.0f}°",
        )

    ax.set_title(f"Spettro osservato — {nome_stella_spettri}")
    ax.set_xlabel("Lunghezza d'onda (nm)")
    ax.set_ylabel("Densità spettrale di fotoni osservata")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    # 2. Conteggi attesi N_atteso vs angolo zenitale
    fig2, ax2 = plt.subplots(figsize=(8.5, 5.0))
    nomi_stelle = list(dict.fromkeys(r["stella"] for r in risultati))
    for nome_stella in nomi_stelle:
        punti = sorted(
            [
                r for r in risultati
                if r["stella"] == nome_stella
                and r["config"] == nome_config_angoli
                and r["sensore"] == nome_sensore_angoli
            ],
            key=lambda x: x["theta_deg"],
        )
        if punti:
            angoli = [p["theta_deg"] for p in punti]
            n_vals = [max(p["N_atteso"], 1e-30) for p in punti]
            ax2.plot(angoli, n_vals, marker="o", label=nome_stella)

    ax2.set_title(f"Conteggi attesi vs angolo zenitale ({nome_config_angoli} — {nome_sensore_angoli})")
    ax2.set_xlabel("Angolo zenitale θ (gradi)")
    ax2.set_ylabel("Conteggi attesi N̄")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    ax2.legend()
    plt.tight_layout()


def plotta_trasmittanza_ozono(
    atm: Atmosfera,
    lambdas: np.ndarray,
    angoli_deg: list[float],
    temperature_o3: list[float],
):
    """
    Mostra la trasmittanza atmosferica totale per diverse temperature dell'ozono
    e angoli zenitali, così da isolare l'assorbimento O3 + Rayleigh.
    """
    if not atm.abilita_ozono or (atm._sigma_interp_1d is None and atm._sigma_interp_2d is None):
        print("Ozono non attivo o sezione d'urto non caricata: grafico trasmittanza O3 saltato.")
        return

    lambdas_nm = lambdas * 1e9
    stili_temp = ["-", "--", ":", "-."]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))

    for i, t_o3 in enumerate(temperature_o3):
        ls = stili_temp[i % len(stili_temp)]
        for theta_deg in angoli_deg:
            theta_rad = np.deg2rad(theta_deg)
            t_atm = atm.trasmittanza_totale(
                lambdas,
                theta_rad=theta_rad,
                temperatura_ozono_k=t_o3,
            )
            ax.plot(
                lambdas_nm,
                t_atm,
                linestyle=ls,
                alpha=0.85,
                label=f"T_O3={t_o3:.0f} K | θ={theta_deg:.0f}°",
            )

    ax.set_title("Trasmittanza atmosferica totale (Rayleigh + O3)")
    ax.set_xlabel("Lunghezza d'onda (nm)")
    ax.set_ylabel("Trasmittanza T_atm(λ, θ)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()


def plotta_attesi_vs_misurati(risultati: list[dict]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0))

    nomi_sensori = list(dict.fromkeys(r["sensore"] for r in risultati))
    tutti_n = []
    tutti_k = []

    for nome_sens in nomi_sensori:
        punti = [r for r in risultati if r["sensore"] == nome_sens]
        n_att = np.array([max(p["N_atteso"], 1e-30) for p in punti], dtype=float)
        k_mis = np.array([max(p["K_misurato"], 1e-30) for p in punti], dtype=float)
        sigma = np.sqrt(n_att)
        scarto_sigma = (k_mis - n_att) / np.where(sigma > 0, sigma, 1.0)

        tutti_n.append(n_att)
        tutti_k.append(k_mis)

        ax1.scatter(n_att, k_mis, alpha=0.8, label=nome_sens)
        ax2.scatter(n_att, scarto_sigma, alpha=0.8, label=nome_sens)

    cat_n = np.concatenate(tutti_n)
    cat_k = np.concatenate(tutti_k)
    min_v = min(cat_n.min(), cat_k.min())
    max_v = max(cat_n.max(), cat_k.max())
    linea_1a1 = np.geomspace(min_v, max_v, 200)

    ax1.plot(linea_1a1, linea_1a1, linestyle="--", color="black", label="K = N̄ (ideale)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Conteggi attesi N̄")
    ax1.set_ylabel("Conteggi misurati K")
    ax1.set_title("Valori misurati K vs attesi N̄")
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    ax1.legend()

    ax2.axhline(0.0, linestyle="--", color="black")
    ax2.axhline(1.0, linestyle=":", alpha=0.7)
    ax2.axhline(-1.0, linestyle=":", alpha=0.7)
    ax2.set_xscale("log")
    ax2.set_xlabel("Conteggi attesi N̄")
    ax2.set_ylabel("(K - N̄) / √N̄")
    ax2.set_title("Fluttuazione statistica in unità di σ")
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    ax2.legend()

    plt.tight_layout()


def plotta_effetto_tramonto(
    stella: Stella,
    atm: Atmosfera,
    angoli_tramonto_deg: list[float] | None = None,
    distanza_orbita_m: float | None = None,
    temperatura_ozono_k: float | None = 223.0,
):
    """
    Visualizza l'effetto del tramonto (grandi angoli zenitali theta) nella banda visibile:
    1) Trasmittanza atmosferica T_atm(lambda, theta) al variare di theta.
    2) Spettro osservato normalizzato a 700 nm per evidenziare il predominio relativo del rosso.
    """
    if angoli_tramonto_deg is None:
        angoli_tramonto_deg = [0.0, 60.0, 75.0, 85.0, 88.0, 90.0]

    lambdas = np.linspace(380e-9, 750e-9, 1000)
    lambdas_nm = lambdas * 1e9

    if distanza_orbita_m is not None:
        d0 = stella.densita_fotoni_a_distanza(lambdas, distanza_orbita_m=distanza_orbita_m)
    else:
        d0 = stella.densita_fotoni(lambdas)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # Indice più vicino a 700 nm per normalizzare il peso relativo rispetto al rosso
    idx_700 = int(np.argmin(np.abs(lambdas_nm - 700.0)))

    for theta_deg in angoli_tramonto_deg:
        theta_rad = np.deg2rad(theta_deg)
        t_atm = atm.trasmittanza_totale(
            lambdas,
            theta_rad=theta_rad,
            temperatura_ozono_k=temperatura_ozono_k,
        )
        d_obs = d0 * t_atm
        rif_rosso = max(d_obs[idx_700], 1e-300)
        d_norm_rosso = d_obs / rif_rosso

        s_km = atm.spessore_massa_aria(theta_rad) / 1000.0
        etichetta = f"θ={theta_deg:.0f}° (S={s_km:.1f} km)"

        ax1.plot(lambdas_nm, t_atm, linewidth=1.8, label=etichetta)
        ax2.plot(lambdas_nm, d_norm_rosso, linewidth=1.8, label=etichetta)

    # Riferimenti visivi per le bande cromatiche principali
    ax1.axvspan(400, 490, color="blue", alpha=0.06, label="Blu/Ciano")
    ax1.axvspan(620, 750, color="red", alpha=0.06, label="Rosso")

    ax1.set_title("Trasmittanza atmosferica vs angolo zenitale")
    ax1.set_xlabel("Lunghezza d'onda (nm)")
    ax1.set_ylabel("Trasmittanza T_atm(λ, θ)")
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8)

    ax2.set_title(f"Spettro relativo a 700 nm — {stella.nome}")
    ax2.set_xlabel("Lunghezza d'onda (nm)")
    ax2.set_ylabel("D_obs(λ) / D_obs(700 nm)")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    ax2.legend(fontsize=8)

    plt.tight_layout()


if __name__ == "__main__":
    file_ozono = trova_file_ozono()
    ha_ozono = file_ozono is not None

    if ha_ozono:
        print(f"File ozono caricato da: {file_ozono}")
    else:
        print("Attenzione: file ozono non trovato nei percorsi standard.")

    stelle_notevoli = [
        Stella("Proxima Centauri (M5.5V)", 3040.0, raggio_m=0.154 * Stella.R_SOLE),
        Stella("Sole (G2V)", 5778.0, raggio_m=1.0 * Stella.R_SOLE),
        Stella("Sirio A (A1V)", 9940.0, raggio_m=1.711 * Stella.R_SOLE),
        Stella("Rigel (B8Ia)", 12100.0, raggio_m=78.9 * Stella.R_SOLE),
    ]

    # Sensori con aree ed esposizioni realistiche (conteggi tra decine e milioni di fotoni)
    sensori_test = [
        Sensore("Rivelatore Piccolo", area=1e-9, sensibilita=0.5, tempo_esposizione=1e-4),
        Sensore("Rivelatore Medio", area=1e-7, sensibilita=0.6, tempo_esposizione=1e-3),
        Sensore("Rivelatore Sensibile", area=1e-5, sensibilita=0.7, tempo_esposizione=1e-2),
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
    ]

    angoli = [0.0, 30.0, 45.0, 60.0, 75.0, 85.0, 88.0]

    # =========================================================================
    # CASO 1: BANDA UV (290 - 350 nm) -> EFFETTO DELL'OZONO
    # =========================================================================
    print("\n" + "=" * 75)
    print("ANALISI 1: BANDA UV (290 - 350 nm) - ASSORBIMENTO OZONO")
    print("=" * 75)

    l_min_uv = max(290e-9, atm._lambda_min_ozono) if atm._lambda_min_ozono is not None else 290e-9
    l_max_uv = min(350e-9, atm._lambda_max_ozono) if atm._lambda_max_ozono is not None else 350e-9
    grid_uv = np.linspace(l_min_uv, l_max_uv, 1000)

    tabella_uv = analizza_stelle_sensori_configurazioni_e_angoli(
        stelle=stelle_notevoli,
        sensori=sensori_test,
        atm=atm,
        configurazioni=configurazioni_atm,
        lambdas=grid_uv,
        angoli_deg=angoli,
        distanza_orbita_m=1.0 * Stella.UA,
        seed=42,
    )

    for r in tabella_uv:
        if r["stella"] == "Sole (G2V)" and r["sensore"] == "Rivelatore Sensibile":
            print(
                f"[UV]  {r['config']:22s} | θ={r['theta_deg']:4.1f}° | "
                f"N̄={r['N_atteso']:.3e} | K={r['K_misurato']:.3e} | SNR={r['SNR']:.2e}"
            )

    plotta_spettri_e_angoli(
        risultati=tabella_uv,
        lambdas=grid_uv,
        nome_stella_spettri="Sole (G2V)",
        nome_config_angoli="Rayleigh + O3 (203 K)",
        nome_sensore_angoli="Rivelatore Sensibile",
        angoli_spettri_deg=[0.0, 60.0, 85.0, 88.0],
    )
    plotta_trasmittanza_ozono(
        atm=atm,
        lambdas=grid_uv,
        angoli_deg=[0.0, 60.0, 85.0, 88.0],
        temperature_o3=[203.0, 223.0, 273.0],
    )
    plotta_attesi_vs_misurati(tabella_uv)

    # =========================================================================
    # CASO 2: BANDA VISIBILE (400 - 700 nm) -> EFFETTO TRAMONTO / RAYLEIGH
    # =========================================================================
    print("\n" + "=" * 75)
    print("ANALISI 2: BANDA VISIBILE (400 - 700 nm) - TRAMONTO E RAYLEIGH")
    print("=" * 75)

    grid_vis = np.linspace(400e-9, 700e-9, 1000)

    tabella_vis = analizza_stelle_sensori_configurazioni_e_angoli(
        stelle=stelle_notevoli,
        sensori=sensori_test,
        atm=atm,
        configurazioni=configurazioni_atm,
        lambdas=grid_vis,
        angoli_deg=angoli,
        distanza_orbita_m=1.0 * Stella.UA,
        seed=42,
    )

    for r in tabella_vis:
        if r["stella"] == "Sole (G2V)" and r["sensore"] == "Rivelatore Sensibile":
            print(
                f"[VIS] {r['config']:22s} | θ={r['theta_deg']:4.1f}° | "
                f"N̄={r['N_atteso']:.3e} | K={r['K_misurato']:.3e} | SNR={r['SNR']:.2e}"
            )

    plotta_spettri_e_angoli(
        risultati=tabella_vis,
        lambdas=grid_vis,
        nome_stella_spettri="Sole (G2V)",
        nome_config_angoli="Rayleigh + O3 (203 K)",
        nome_sensore_angoli="Rivelatore Sensibile",
        angoli_spettri_deg=[0.0, 60.0, 85.0, 88.0],
    )
    plotta_trasmittanza_ozono(
        atm=atm,
        lambdas=grid_vis,
        angoli_deg=[0.0, 60.0, 85.0, 88.0],
        temperature_o3=[203.0, 223.0, 273.0],
    )
    plotta_attesi_vs_misurati(tabella_vis)

    # =========================================================================
    # GRAFICO DEDICATO ALL'EFFETTO TRAMONTO (380 - 750 nm)
    # =========================================================================
    sole = next((s for s in stelle_notevoli if "Sole" in s.nome), stelle_notevoli[1])
    plotta_effetto_tramonto(
        stella=sole,
        atm=atm,
        angoli_tramonto_deg=[0.0, 60.0, 75.0, 85.0, 88.0, 90.0],
        distanza_orbita_m=1.0 * Stella.UA,
        temperatura_ozono_k=223.0,
    )

    plt.show()