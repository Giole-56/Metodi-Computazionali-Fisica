from pathlib import Path
import numpy as np
from scipy import integrate, interpolate


class Stella:
    """Rappresenta una stella caratterizzata da un nome, temperatura e raggio opzionale."""

    # Costanti fisiche e astronomiche fondamentali
    h = 6.62607015e-34      # Costante di Planck (J s)
    c = 2.99792458e8        # Velocità della luce nel vuoto (m/s)
    k_B = 1.380649e-23      # Costante di Boltzmann (J/K)
    R_SOLE = 6.9634e8       # Raggio solare medio (m)
    UA = 1.495978707e11     # Unità astronomica (m)

    def __init__(self, nome: str, temperatura: float, raggio_m: float | None = None):
        self.nome = nome
        self.temperatura = float(temperatura)  # Temperatura in Kelvin (K)
        self.raggio_m = float(raggio_m) if raggio_m is not None else None  # Raggio in metri (m)

    def emissione_corpo_nero(self, lambdas: np.ndarray) -> np.ndarray:
        """
        Calcola la radianza spettrale di Planck B(lambda, T)
        in funzione di un array di lunghezze d'onda (in metri).
        """
        esponente = np.clip((self.h * self.c) / (lambdas * self.k_B * self.temperatura), None, 700)
        num = 2 * self.h * self.c**2
        den = (lambdas**5) * (np.exp(esponente) - 1)
        return num / den

    def densita_fotoni(self, lambdas: np.ndarray) -> np.ndarray:
        """
        Calcola la densità spettrale di fotoni D(lambda, T)
        in funzione di un array di lunghezze d'onda (in metri).
        """
        esponente = np.clip((self.h * self.c) / (lambdas * self.k_B * self.temperatura), None, 700)
        num = 2 * self.c
        den = (lambdas**4) * (np.exp(esponente) - 1)
        return num / den

    def densita_fotoni_a_distanza(
        self,
        lambdas: np.ndarray,
        distanza_orbita_m: float,
        raggio_stella_m: float | None = None,
    ) -> np.ndarray:
        """
        Scala la densità spettrale di fotoni dalla superficie stellare alla distanza
        orbitale tramite il fattore geometrico (R_stella / d_orbita)^2.
        """
        r_eff = raggio_stella_m if raggio_stella_m is not None else self.raggio_m
        if r_eff is None:
            raise ValueError(
                "Specifica 'raggio_stella_m' oppure imposta 'raggio_m' nella stella."
            )
        if r_eff <= 0 or distanza_orbita_m <= 0:
            raise ValueError("Raggio stellare e distanza orbitale devono essere maggiori di 0.")

        fattore_geometrico = (r_eff / distanza_orbita_m) ** 2
        return self.densita_fotoni(lambdas) * fattore_geometrico

    # =========================================================================
    # METODI DI INTEGRAZIONE
    # =========================================================================

    def flusso_fotonico_banda(self, lambda_min: float, lambda_max: float, trasmissivita_fn=None) -> float:
        """
        Integra la densità spettrale di fotoni D(lambda, T) tra lambda_min e lambda_max
        usando quadratura adattiva (scipy.integrate.quad).
        """
        def integranda(lam):
            d_lam = self.densita_fotoni(np.array([lam]))[0]
            if trasmissivita_fn is not None:
                return d_lam * trasmissivita_fn(lam)
            return d_lam

        flusso, _ = integrate.quad(integranda, lambda_min, lambda_max, limit=100)
        return float(flusso)

    def flusso_fotonico_da_array(self, lambdas: np.ndarray, trasmissivita: np.ndarray = None) -> float:
        """
        Integra numericamente lo spettro su una griglia discreta di lunghezze d'onda
        tramite la regola di Simpson.
        """
        spettro = self.densita_fotoni(lambdas)
        if trasmissivita is not None:
            spettro = spettro * trasmissivita
        return float(integrate.simpson(y=spettro, x=lambdas))

    def radianza_integrata(
        self,
        lambda_min: float,
        lambda_max: float,
        emisferica: bool = False,
    ) -> float:
        """
        Integra la radianza spettrale B(lambda, T) nella banda [lambda_min, lambda_max].
        """
        potenza, _ = integrate.quad(
            lambda lam: self.emissione_corpo_nero(np.array([lam]))[0],
            lambda_min,
            lambda_max,
            limit=100,
        )
        if emisferica:
            potenza *= np.pi
        return float(potenza)

    # =========================================================================
    # METODI DI DERIVAZIONE
    # =========================================================================

    def derivata_spettro_fotoni(self, lambdas: np.ndarray) -> np.ndarray:
        """
        Calcola la derivata prima dD(lambda, T)/dlambda lungo l'array delle lunghezze d'onda.
        """
        spettro = self.densita_fotoni(lambdas)
        return np.gradient(spettro, lambdas)

    def derivata_radianza(self, lambdas: np.ndarray) -> np.ndarray:
        """
        Calcola la derivata prima dB(lambda, T)/dlambda della curva di Planck.
        """
        radianza = self.emissione_corpo_nero(lambdas)
        return np.gradient(radianza, lambdas)

    def lunghezza_onda_picco_fotoni(
        self,
        lambda_min: float | None = None,
        lambda_max: float | None = None,
        n_punti: int = 10000,
    ) -> float:
        """
        Trova la lunghezza d'onda di massima densità fotonica D(lambda, T) [in metri]
        cercando lo zero della derivata dD/dlambda attorno alla stima teorica.
        """
        if self.temperatura <= 0:
            raise ValueError("La temperatura della stella deve essere maggiore di 0 K.")

        lambda_stima = 3.669e-3 / self.temperatura
        if lambda_min is None:
            lambda_min = lambda_stima * 0.2
        if lambda_max is None:
            lambda_max = lambda_stima * 5.0

        lambdas = np.linspace(lambda_min, lambda_max, n_punti)
        derivata = self.derivata_spettro_fotoni(lambdas)

        idx_zero = np.where(np.diff(np.sign(derivata)) < 0)[0]
        if len(idx_zero) > 0:
            return float(lambdas[idx_zero[0]])
        return float(lambdas[np.argmax(self.densita_fotoni(lambdas))])

    def __lt__(self, altro):
        if not isinstance(altro, Stella):
            return NotImplemented
        return self.temperatura < altro.temperatura

    def __gt__(self, altro):
        if not isinstance(altro, Stella):
            return NotImplemented
        return self.temperatura > altro.temperatura

    def __repr__(self):
        r_info = f", R={self.raggio_m:.3e} m" if self.raggio_m is not None else ""
        return f"Stella(nome='{self.nome}', T={self.temperatura} K{r_info})"


class Sensore:
    """Rappresenta un sensore con area, sensibilità e tempo di esposizione."""

    SOGLIA_GAUSSIANA = 2e9

    def __init__(self, nome: str, area: float, sensibilita, tempo_esposizione: float):
        self.nome = nome
        self.area = float(area)
        self.sensibilita = sensibilita
        self.tempo_esposizione = float(tempo_esposizione)

    def efficienza_spettrale(self, lambdas: np.ndarray) -> np.ndarray:
        """Restituisce l'efficienza quantica epsilon(lambda) lungo l'array di lunghezze d'onda."""
        lam = np.asarray(lambdas, dtype=float)
        if callable(self.sensibilita):
            return np.asarray(self.sensibilita(lam), dtype=float)
        return np.full_like(lam, float(self.sensibilita), dtype=float)

    def calcola_conteggi_da_spettro(self, lambdas: np.ndarray, flusso_spettrale: np.ndarray) -> float:
        """
        Calcola il conteggio medio atteso N_bar integrando sullo spettro
        flusso_spettrale(lambda) * epsilon(lambda) * A * t.
        """
        lam = np.asarray(lambdas, dtype=float)
        flusso = np.asarray(flusso_spettrale, dtype=float)
        qe = self.efficienza_spettrale(lam)
        segnale_spettrale = flusso * qe * self.area * self.tempo_esposizione
        return float(integrate.simpson(y=segnale_spettrale, x=lam))

    def conteggi_cumulativi(self, lambdas: np.ndarray, flusso_spettrale: np.ndarray) -> np.ndarray:
        """Calcola l'integrale cumulativo dei conteggi al variare della lunghezza d'onda."""
        lam = np.asarray(lambdas, dtype=float)
        flusso = np.asarray(flusso_spettrale, dtype=float)
        qe = self.efficienza_spettrale(lam)
        segnale_spettrale = flusso * qe * self.area * self.tempo_esposizione
        return integrate.cumulative_trapezoid(y=segnale_spettrale, x=lam, initial=0.0)

    def tasso_variazione_segnale(self, lambdas: np.ndarray, flusso_spettrale: np.ndarray) -> np.ndarray:
        """Calcola la derivata del segnale spettrale rispetto a lambda."""
        lam = np.asarray(lambdas, dtype=float)
        flusso = np.asarray(flusso_spettrale, dtype=float)
        qe = self.efficienza_spettrale(lam)
        segnale_spettrale = flusso * qe * self.area * self.tempo_esposizione
        return np.gradient(segnale_spettrale, lam)

    def _campiona_rumore(self, segnale_medio: float, rng: np.random.Generator | None = None) -> int:
        """Genera il conteggio misurato K attorno a N_bar (Poisson, o Gaussiana per N_bar molto grande)."""
        if segnale_medio <= 0:
            return 0
        gen = rng if rng is not None else np.random.default_rng()
        if segnale_medio > self.SOGLIA_GAUSSIANA:
            misura = gen.normal(loc=segnale_medio, scale=np.sqrt(segnale_medio))
            return max(0, int(round(misura)))
        return int(gen.poisson(segnale_medio))

    def simula_misura_da_spettro(
        self,
        lambdas: np.ndarray,
        flusso_spettrale: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Calcola N_bar dallo spettro e restituisce una singola misura rumorosa K."""
        segnale_medio = self.calcola_conteggi_da_spettro(lambdas, flusso_spettrale)
        return self._campiona_rumore(segnale_medio, rng=rng)

    @staticmethod
    def snr(segnale_medio: float) -> float:
        """Calcola il rapporto segnale/rumore SNR = sqrt(N_bar)."""
        return float(np.sqrt(segnale_medio)) if segnale_medio > 0 else 0.0

    def __repr__(self):
        sens_str = "funzione(lambda)" if callable(self.sensibilita) else f"{self.sensibilita}"
        return (
            f"Sensore(nome='{self.nome}', area={self.area} m^2, "
            f"sensibilita={sens_str}, tempo_esposizione={self.tempo_esposizione} s)"
        )


class Atmosfera:
    """
    Modella la propagazione della luce nell'atmosfera terrestre includendo:
    - spessore di massa d'aria S(theta)
    - scattering di Rayleigh
    - assorbimento dell'ozono (O3) da sezione d'urto tabulata
    """

    R_TERRA = 6.371e6           # Raggio medio terrestre (m)
    S_ZENITH_DEFAULT = 8000.0   # Spessore efficace allo zenit (m)
    N_ARIA_DEFAULT = 1.00029    # Indice di rifrazione dell'aria
    N_MOLEC_DEFAULT = 2.504e25  # Densità molecolare dell'aria (molecole / m^3)

    def __init__(
        self,
        abilita_rayleigh: bool = True,
        abilita_ozono: bool = True,
        s_zenith: float = S_ZENITH_DEFAULT,
        r_terra: float = R_TERRA,
        n_aria: float = N_ARIA_DEFAULT,
        densita_molecole: float = N_MOLEC_DEFAULT,
        colonna_ozono: float = 8.0e22,        # Colonna efficace O3 (molecole / m^2)
        percorso_ozono: float = 1.0,          # Fattore moltiplicativo del percorso O3
        temperatura_ozono: float = 223.0,     # Temperatura efficace O3 (K)
        temperature_colonne_o3: list[float] | np.ndarray | None = None,
        file_ozono: str | Path | None = None, # Percorso a 'SCIA O3 Temp cross-section V4.1.DAT'
        unita_lambda_file: str = "nm",        # 'nm' oppure 'm'
        unita_sigma_file: str = "cm2",        # 'cm2' oppure 'm2'
    ):
        self.abilita_rayleigh = bool(abilita_rayleigh)
        self.abilita_ozono = bool(abilita_ozono)
        self.s_zenith = float(s_zenith)
        self.r_terra = float(r_terra)
        self.n_aria = float(n_aria)
        self.densita_molecole = float(densita_molecole)
        self.colonna_ozono = float(colonna_ozono)
        self.percorso_ozono = float(percorso_ozono)
        self.temperatura_ozono = float(temperatura_ozono)
        self._temp_colonne_o3 = (
            np.asarray(temperature_colonne_o3, dtype=float)
            if temperature_colonne_o3 is not None
            else None
        )

        self._sigma_interp_1d = None
        self._sigma_interp_2d = None
        self._lambda_min_ozono = None
        self._lambda_max_ozono = None

        if file_ozono is not None:
            self.carica_sezione_urto_ozono(
                str(file_ozono),
                unita_lambda=unita_lambda_file,
                unita_sigma=unita_sigma_file,
                temperature_colonne=self._temp_colonne_o3,
            )

    def configura(
        self,
        rayleigh: bool | None = None,
        ozono: bool | None = None,
        colonna_ozono: float | None = None,
        percorso_ozono: float | None = None,
        temperatura_ozono: float | None = None,
    ):
        """Aggiorna in modo rapido la configurazione fisica dell'atmosfera."""
        if rayleigh is not None:
            self.abilita_rayleigh = bool(rayleigh)
        if ozono is not None:
            self.abilita_ozono = bool(ozono)
        if colonna_ozono is not None:
            self.colonna_ozono = float(colonna_ozono)
        if percorso_ozono is not None:
            self.percorso_ozono = float(percorso_ozono)
        if temperatura_ozono is not None:
            self.temperatura_ozono = float(temperatura_ozono)

    # =========================================================================
    # SPESSORE DI MASSA D'ARIA
    # =========================================================================

    def spessore_orizzontale(self) -> float:
        """Calcola S_oriz = sqrt((R_T + S_zenith)^2 - R_T^2) trascurando la rifrazione."""
        return float(np.sqrt((self.r_terra + self.s_zenith) ** 2 - self.r_terra ** 2))

    def spessore_massa_aria(self, theta_rad: float | np.ndarray) -> float | np.ndarray:
        """Calcola lo spessore efficace S_theta in funzione dell'angolo zenitale theta (in radianti)."""
        cos_t = np.cos(theta_rad)
        rt = self.r_terra
        sz = self.s_zenith
        radicando = np.maximum((rt * cos_t) ** 2 + 2.0 * rt * sz + sz ** 2, 0.0)
        s_theta = np.sqrt(radicando) - rt * cos_t
        if np.ndim(theta_rad) == 0:
            return float(s_theta)
        return s_theta

    # =========================================================================
    # SCATTERING DI RAYLEIGH
    # =========================================================================

    def coefficiente_rayleigh(self, lambdas: np.ndarray) -> np.ndarray:
        """Calcola beta(lambda, n, N) = (8 * pi^3 / (3 * lambda^4 * N)) * (n^2 - 1)^2 [m^-1]."""
        lam = np.asarray(lambdas, dtype=float)
        fattore_rifrazione = (self.n_aria ** 2 - 1.0) ** 2
        return ((8.0 * (np.pi ** 3)) / (3.0 * (lam ** 4) * self.densita_molecole)) * fattore_rifrazione

    def trasmittanza_rayleigh(self, lambdas: np.ndarray, theta_rad: float = 0.0) -> np.ndarray:
        """Calcola exp(-beta(lambda) * S_theta) per lo scattering di Rayleigh."""
        lam = np.asarray(lambdas, dtype=float)
        if not self.abilita_rayleigh:
            return np.ones_like(lam, dtype=float)
        s_path = self.spessore_massa_aria(theta_rad)
        beta = self.coefficiente_rayleigh(lam)
        return np.exp(-np.clip(beta * s_path, 0.0, 700.0))

    # =========================================================================
    # ASSORBIMENTO DELL'OZONO (O3)
    # =========================================================================

    def carica_sezione_urto_ozono(
        self,
        percorso_file: str,
        unita_lambda: str = "nm",
        unita_sigma: str = "cm2",
        temperature_colonne: list[float] | np.ndarray | None = None,
    ):
        """
        Carica la sezione d'urto dell'ozono da file (es. SCIA O3 Temp cross-section V4.1.DAT)
        e costruisce l'interpolatore in lunghezza d'onda [m] e, se disponibili più colonne
        di temperatura, in temperatura [K]. Restituisce sigma in [m^2].
        """
        righe_valide = []
        n_colonne_attese = None

        with open(percorso_file, "r", encoding="utf-8", errors="ignore") as f:
            for riga in f:
                s = riga.strip()
                if not s or s.startswith(("#", "!", ";", "/", "*")):
                    continue
                parti = s.replace(",", ".").split()
                if len(parti) < 2:
                    continue
                try:
                    valori = [float(x) for x in parti]
                except ValueError:
                    continue

                if n_colonne_attese is None:
                    n_colonne_attese = len(valori)
                if len(valori) == n_colonne_attese:
                    righe_valide.append(valori)

        if not righe_valide:
            raise ValueError(f"Nessun dato numerico valido trovato nel file: {percorso_file}")

        dati = np.array(righe_valide, dtype=float)
        lam_raw = dati[:, 0]
        sig_raw = dati[:, 1:]

        if unita_lambda == "nm":
            lam_m = lam_raw * 1e-9
        elif unita_lambda == "m":
            lam_m = lam_raw
        else:
            raise ValueError("unita_lambda deve essere 'nm' o 'm'.")

        if unita_sigma == "cm2":
            sig_m2 = sig_raw * 1e-4
        elif unita_sigma == "m2":
            sig_m2 = sig_raw
        else:
            raise ValueError("unita_sigma deve essere 'cm2' o 'm2'.")

        ordine = np.argsort(lam_m)
        lam_m = lam_m[ordine]
        sig_m2 = sig_m2[ordine]

        lam_uniche, idx_unici = np.unique(lam_m, return_index=True)
        sig_uniche = sig_m2[idx_unici]

        self._lambda_min_ozono = float(lam_uniche[0])
        self._lambda_max_ozono = float(lam_uniche[-1])
        self._sigma_interp_2d = None

        n_temp_cols = sig_uniche.shape[1]
        if n_temp_cols > 1:
            if temperature_colonne is not None and len(temperature_colonne) == n_temp_cols:
                temps = np.asarray(temperature_colonne, dtype=float)
            else:
                temps = np.linspace(203.0, 293.0, n_temp_cols)

            idx_t = np.argsort(temps)
            temps_ord = temps[idx_t]
            sig_ord_t = sig_uniche[:, idx_t]
            self._temp_colonne_o3 = temps_ord

            self._sigma_interp_2d = interpolate.RegularGridInterpolator(
                (lam_uniche, temps_ord),
                sig_ord_t,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            self._sigma_interp_1d = interpolate.interp1d(
                lam_uniche,
                sig_ord_t[:, 0],
                kind="linear",
                bounds_error=False,
                fill_value=0.0,
            )
        else:
            self._sigma_interp_1d = interpolate.interp1d(
                lam_uniche,
                sig_uniche[:, 0],
                kind="linear",
                bounds_error=False,
                fill_value=0.0,
            )

    def sezione_urto_ozono(self, lambdas: np.ndarray, temperatura_k: float | None = None) -> np.ndarray:
        """Restituisce sigma_O3(lambda, T) in [m^2] per le lunghezze d'onda fornite [m]."""
        lam = np.asarray(lambdas, dtype=float)
        if self._sigma_interp_1d is None and self._sigma_interp_2d is None:
            return np.zeros_like(lam, dtype=float)

        t_val = float(self.temperatura_ozono if temperatura_k is None else temperatura_k)
        if self._sigma_interp_2d is not None:
            t_min = float(self._temp_colonne_o3[0])
            t_max = float(self._temp_colonne_o3[-1])
            t_clamped = np.clip(t_val, t_min, t_max)
            fuori_banda_lam = (lam < self._lambda_min_ozono) | (lam > self._lambda_max_ozono)
            lam_clamped = np.clip(lam, self._lambda_min_ozono, self._lambda_max_ozono)
            punti = np.column_stack([lam_clamped, np.full_like(lam_clamped, t_clamped, dtype=float)])
            sigma = self._sigma_interp_2d(punti)
            sigma = np.where(fuori_banda_lam, 0.0, sigma)
            return np.maximum(sigma, 0.0)

        return np.maximum(self._sigma_interp_1d(lam), 0.0)

    def trasmittanza_ozono(
        self,
        lambdas: np.ndarray,
        theta_rad: float = 0.0,
        temperatura_k: float | None = None,
    ) -> np.ndarray:
        """Calcola la trasmittanza dovuta all'ozono: exp(-tau_O3)."""
        lam = np.asarray(lambdas, dtype=float)
        if not self.abilita_ozono or (self._sigma_interp_1d is None and self._sigma_interp_2d is None):
            return np.ones_like(lam, dtype=float)

        fattore_aria = self.spessore_massa_aria(theta_rad) / max(self.s_zenith, 1e-9)
        profondita_ottica_o3 = (
            self.sezione_urto_ozono(lam, temperatura_k=temperatura_k)
            * self.colonna_ozono
            * fattore_aria
            * self.percorso_ozono
        )
        return np.exp(-np.clip(profondita_ottica_o3, 0.0, 700.0))

    # =========================================================================
    # TRASMITTANZA TOTALE E SPETTRO OSSERVATO
    # =========================================================================

    def trasmittanza_totale(
        self,
        lambdas: np.ndarray,
        theta_rad: float = 0.0,
        temperatura_ozono_k: float | None = None,
    ) -> np.ndarray:
        """Combina gli effetti attivi di scattering di Rayleigh e assorbimento dell'ozono."""
        return self.trasmittanza_rayleigh(lambdas, theta_rad) * self.trasmittanza_ozono(
            lambdas,
            theta_rad=theta_rad,
            temperatura_k=temperatura_ozono_k,
        )

    def osserva_spettro(
        self,
        lambdas: np.ndarray,
        densita_fotoni_iniziale: np.ndarray,
        theta_rad: float = 0.0,
        temperatura_ozono_k: float | None = None,
    ) -> np.ndarray:
        """Applica la legge N_obs(lambda) = N_0(lambda) * T_atm(lambda, theta)."""
        return np.asarray(densita_fotoni_iniziale, dtype=float) * self.trasmittanza_totale(
            lambdas,
            theta_rad=theta_rad,
            temperatura_ozono_k=temperatura_ozono_k,
        )

    # =========================================================================
    # METODI DI INTEGRAZIONE E DERIVAZIONE
    # =========================================================================

    def trasmittanza_media_banda(
        self,
        lambda_min: float,
        lambda_max: float,
        theta_rad: float = 0.0,
        n_punti: int = 2000,
    ) -> float:
        """Integra la trasmittanza atmosferica su [lambda_min, lambda_max] e restituisce il valore medio."""
        griglia = np.linspace(lambda_min, lambda_max, n_punti)
        t_vals = self.trasmittanza_totale(griglia, theta_rad=theta_rad)
        integrale = integrate.simpson(y=t_vals, x=griglia)
        return float(integrale / (lambda_max - lambda_min))

    def profondita_ottica_integrata(
        self,
        lambdas: np.ndarray,
        theta_rad: float = 0.0,
    ) -> float:
        """Integra la profondità ottica spettrale tau(lambda) = -ln(T(lambda)) lungo l'array."""
        t_tot = np.maximum(self.trasmittanza_totale(lambdas, theta_rad=theta_rad), 1e-300)
        tau = -np.log(t_tot)
        return float(integrate.simpson(y=tau, x=lambdas))

    def derivata_trasmittanza_lambda(self, lambdas: np.ndarray, theta_rad: float = 0.0) -> np.ndarray:
        """Calcola dT_atm/dlambda rispetto alla lunghezza d'onda."""
        t_tot = self.trasmittanza_totale(lambdas, theta_rad=theta_rad)
        return np.gradient(t_tot, lambdas)

    def derivata_spessore_theta(self, angoli_rad: np.ndarray) -> np.ndarray:
        """Calcola dS_theta/dtheta rispetto all'angolo zenitale (in radianti)."""
        ang = np.asarray(angoli_rad, dtype=float)
        s_vals = self.spessore_massa_aria(ang)
        return np.gradient(s_vals, ang)

    def __repr__(self):
        return (
            f"Atmosfera(rayleigh={self.abilita_rayleigh}, ozono={self.abilita_ozono}, "
            f"T_O3={self.temperatura_ozono} K, S_zenith={self.s_zenith} m)"
        )