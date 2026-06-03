# src/core/physics.py

def calc_stefan_boltzmann_temp(insolacion):
    """
    Calcula la Temperatura de Equilibrio (K) basada en la insolación estelar,
    asumiendo un Albedo de Bond y Factor de Redistribución similares a la Tierra.
    """
    return 255 * (insolacion ** 0.25)

def calc_stefan_boltzmann_insol(temperatura):
    """
    Calcula la Insolación Estelar (relativa a la Tierra) requerida
    para alcanzar una determinada Temperatura de Equilibrio.
    """
    return (temperatura / 255) ** 4

def calc_densidad_planetaria(masa_tierra, radio_tierra):
    """
    Calcula la densidad del exoplaneta en g/cm³, tomando como base
    la densidad media de la Tierra (5.51 g/cm³).
    """
    return 5.51 * (masa_tierra / (radio_tierra ** 3))