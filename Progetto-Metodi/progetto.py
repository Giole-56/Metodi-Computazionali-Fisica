import numpy as np

class Sensore:
    """Rappresenta un sensore con area, sensibilità e tempo di esposizione."""
    
    def __init__(self, nome: str, area: float, sensibilita: float, tempo_esposizione: float):
        self.nome = nome
        self.area = area                            # Area in metri quadri (m^2)
        self.sensibilita = sensibilita              # Efficienza quantica (0-1)
        self.tempo_esposizione = tempo_esposizione  # Tempo di esposizione in secondi (s)

    def calcola_conteggi(self, flusso_fotonico: float) -> float:
        """Calcola il segnale medio atteso dato un flusso fotonico."""
        return flusso_fotonico * self.area * self.tempo_esposizione * self.sensibilita

    def simula_misura(self, flusso_fotonico: float) -> int:
        """Simula la misura con fluttuazione statistica (rumore di Poisson)."""
        segnale_medio = self.calcola_conteggi(flusso_fotonico)
        return np.random.poisson(segnale_medio)

    def __repr__(self):
        return (f"Sensore(nome='{self.nome}', Area={self.area} m^2, "
                f"Sensibilita={self.sensibilita}, Esposizione={self.tempo_esposizione} s)")
        

class Stella:
    """Rappresenta una stella caratterizzata da un nome e dalla sua temperatura."""
    
    def __init__(self, nome: str, temperatura: float):
        self.nome = nome
        self.temperatura = temperatura  # Temperatura in Kelvin (K)

    def __lt__(self, altro):
        if not isinstance(altro, Stella):
            return NotImplemented
        return self.temperatura < altro.temperatura

    def __gt__(self, altro):
        if not isinstance(altro, Stella):
            return NotImplemented
        return self.temperatura > altro.temperatura

    def __repr__(self):
        return f"Stella(nome='{self.nome}', T={self.temperatura} K)"