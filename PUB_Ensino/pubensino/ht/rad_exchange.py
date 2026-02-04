# pubensino/ht/rad_exchange.py
"""Rotinas para radiação térmica (superfícies difusas-cinzentas)."""

from __future__ import annotations
import numpy as np

SIGMA = 5.670374419e-8  # Stefan–Boltzmann (W/m²/K⁴)

def linearized_hr(eps: float, Tm: float) -> float:
    """h_r = 4 ε σ Tm³."""
    eps = float(eps); Tm = float(Tm)
    if not (0.0 < eps <= 1.0):
        raise ValueError("eps deve estar em (0,1].")
    if Tm <= 0:
        raise ValueError("Tm deve ser positivo (K).")
    return 4.0 * eps * SIGMA * Tm**3

def net_to_large_enclosure(A: float, eps: float, Ts, Tsur: float):
    """q = ε σ A (Ts⁴ - Tsur⁴). Aceita Ts escalar ou array."""
    A = float(A); eps = float(eps)
    Ts = np.asarray(Ts, dtype=float); Tsur = float(Tsur)
    if A <= 0:
        raise ValueError("A deve ser positiva.")
    if not (0.0 < eps <= 1.0):
        raise ValueError("eps deve estar em (0,1].")
    if np.any(Ts <= 0) or Tsur <= 0:
        raise ValueError("Temperaturas devem ser positivas (K).")
    return eps * SIGMA * A * (Ts**4 - Tsur**4)

def effective_emissivity_two_surfaces_unitF(eps1: float, eps2: float) -> float:
    """ε_eff = 1 / (1/ε1 + 1/ε2 - 1)."""
    eps1=float(eps1); eps2=float(eps2)
    if not (0.0 < eps1 <= 1.0) or not (0.0 < eps2 <= 1.0):
        raise ValueError("eps1 e eps2 devem estar em (0,1].")
    denom = (1.0/eps1) + (1.0/eps2) - 1.0
    return 1.0/denom

def net_between_two_surfaces_unitF(A: float, eps1: float, eps2: float, T1: float, T2: float) -> float:
    """q = σ (T1⁴ - T2⁴) / ( (1-ε1)/(Aε1) + 1/A + (1-ε2)/(Aε2) )."""
    A=float(A); eps1=float(eps1); eps2=float(eps2); T1=float(T1); T2=float(T2)
    if A <= 0:
        raise ValueError("A deve ser positiva.")
    if not (0.0 < eps1 <= 1.0) or not (0.0 < eps2 <= 1.0):
        raise ValueError("eps1 e eps2 devem estar em (0,1].")
    if T1 <= 0 or T2 <= 0:
        raise ValueError("Temperaturas devem ser positivas (K).")
    R1 = (1.0 - eps1) / (A * eps1)
    Rspace = 1.0 / A
    R2 = (1.0 - eps2) / (A * eps2)
    return SIGMA * (T1**4 - T2**4) / (R1 + Rspace + R2)
