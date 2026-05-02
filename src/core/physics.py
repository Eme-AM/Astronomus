import numpy as np

class PlanetaryService:
    @staticmethod
    def calculate_density(mass, radius):
        """Calcula densidad física: ρ [g/cm³] ≈ 5.51 * (M/M⊕) / (R/R⊕)³"""
        if radius <= 0: return 0
        return 5.51 * (mass) / (radius ** 3)

    @staticmethod
    def evaluate_habitability_rules(row):
        """Reglas heurísticas (Expert System)"""
        is_habitable = (
            0.5 <= row['pl_rade'] <= 1.8 and
            200 <= row['pl_eqt'] <= 320 and
            0.2 <= row['pl_insol'] <= 1.8
        )
        return 1 if is_habitable else 0
