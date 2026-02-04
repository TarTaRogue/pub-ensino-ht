# pubensino/ht/wall1d.py
# Condução 1D em regime permanente em parede plana
# Convenção: temperaturas em Kelvin (K)

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class Wall1DParams:
    k: float    # W/(m·K)
    L: float    # m
    A: float    # m²
    T1: float   # K
    T2: float   # K


def validate_params(p: Wall1DParams) -> None:
    """Validações físicas mínimas."""
    if p.k <= 0:
        raise ValueError("k deve ser > 0 [W/(m·K)].")
    if p.L <= 0:
        raise ValueError("L deve ser > 0 [m].")
    if p.A <= 0:
        raise ValueError("A deve ser > 0 [m²].")
    if p.T1 <= 0 or p.T2 <= 0:
        raise ValueError("Temperaturas devem estar em Kelvin (T > 0 K).")


def solve_wall_steady_1d(p: Wall1DParams, nx: int = 200) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Solução analítica para parede plana 1D, regime permanente:
      T(x) linear; q'' constante.
    Retorna:
      x [m], T(x) [K], q'' [W/m²], Qdot [W]
    """
    validate_params(p)
    if nx < 10:
        raise ValueError("nx deve ser >= 10.")

    x = np.linspace(0.0, p.L, nx)
    T = p.T1 + (p.T2 - p.T1) * (x / p.L)

    dTdx = (p.T2 - p.T1) / p.L
    qpp = -p.k * dTdx           # W/m²
    Qdot = qpp * p.A            # W

    return x, T, qpp, Qdot


def thermal_resistance(p: Wall1DParams) -> float:
    """Resistência térmica por condução (parede plana): R_cond = L/(kA) [K/W]."""
    validate_params(p)
    return p.L / (p.k * p.A)


def check_consistency(p: Wall1DParams) -> float:
    """
    Retorna erro relativo entre:
      Qdot (Fourier) vs Qdot via resistência térmica.
    """
    _, _, _, Qdot = solve_wall_steady_1d(p)
    R = thermal_resistance(p)
    Qdot_R = (p.T1 - p.T2) / R

    rel_err = abs(Qdot_R - Qdot) / (abs(Qdot) + 1e-15)
    return rel_err


def _sanity() -> str:
    return "wall1d OK"


from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass(frozen=True)
class Wall1DConvParams:
    k: float    # W/(m·K)
    L: float    # m
    A: float    # m²
    T1: float   # K
    h: float    # W/(m²·K)
    Tinf: float # K

def validate_params_conv(p: Wall1DConvParams) -> None:
    if p.k <= 0:
        raise ValueError("k deve ser > 0 [W/(m·K)].")
    if p.L <= 0:
        raise ValueError("L deve ser > 0 [m].")
    if p.A <= 0:
        raise ValueError("A deve ser > 0 [m²].")
    if p.h <= 0:
        raise ValueError("h deve ser > 0 [W/(m²·K)].")
    if p.T1 <= 0 or p.Tinf <= 0:
        raise ValueError("Temperaturas devem estar em Kelvin (T > 0 K).")

def solve_wall_steady_1d_conv(p: Wall1DConvParams, nx: int = 200) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Parede plana 1D, regime permanente:
      T(0)=T1 e convecção em x=L: -k dT/dx = h (T(L)-Tinf)
    Retorna:
      x [m], T(x) [K], q'' [W/m²], Qdot [W], Ts=T(L) [K]
    """
    validate_params_conv(p)
    if nx < 10:
        raise ValueError("nx deve ser >= 10.")

    # Fluxo fechado
    qpp = (p.T1 - p.Tinf) / (p.L/p.k + 1.0/p.h)   # W/m²

    # Perfil linear
    x = np.linspace(0.0, p.L, nx)
    T = p.T1 - (qpp/p.k) * x                      # K

    Ts = T[-1]
    Qdot = qpp * p.A
    return x, T, qpp, Qdot, Ts


# ============================
# Condução radial 1D — Cilindro (regime permanente, sem geração)
# ============================

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class Cyl1DParams:
    k: float     # W/(m·K)
    ri: float    # m
    ro: float    # m
    L: float     # m (comprimento do cilindro)
    Ti: float    # K (em r=ri)
    To: float    # K (em r=ro)


def validate_params_cyl(p: Cyl1DParams) -> None:
    if p.k <= 0:
        raise ValueError("k deve ser > 0 [W/(m·K)].")
    if p.L <= 0:
        raise ValueError("L deve ser > 0 [m].")
    if p.ri <= 0 or p.ro <= 0:
        raise ValueError("ri e ro devem ser > 0 [m].")
    if p.ro <= p.ri:
        raise ValueError("Deve valer ro > ri.")
    if p.Ti <= 0 or p.To <= 0:
        raise ValueError("Temperaturas devem estar em Kelvin (T > 0 K).")


def solve_cylinder_steady_1d(p: Cyl1DParams, nr: int = 300) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Casca cilíndrica (1D radial), regime permanente, k const, sem geração.
    BCs: T(ri)=Ti, T(ro)=To.

    Retorna:
      r [m], T(r) [K], Qdot [W], qpp(r) [W/m²] onde qpp(r)=Qdot/(2π r L)
    """
    validate_params_cyl(p)
    if nr < 20:
        raise ValueError("nr deve ser >= 20.")

    r = np.linspace(p.ri, p.ro, nr)

    # Perfil de temperatura (logarítmico)
    lnR = np.log(p.ro / p.ri)
    T = p.Ti + (p.To - p.Ti) * (np.log(r / p.ri) / lnR)

    # Taxa de calor (constante)
    Qdot = 2.0 * np.pi * p.L * p.k * (p.Ti - p.To) / lnR

    # Fluxo radial por área local
    qpp = Qdot / (2.0 * np.pi * r * p.L)

    return r, T, Qdot, qpp


def thermal_resistance_cyl(p: Cyl1DParams) -> float:
    """R_cond,cyl = ln(ro/ri)/(2π k L) [K/W]."""
    validate_params_cyl(p)
    return np.log(p.ro / p.ri) / (2.0 * np.pi * p.k * p.L)


# ============================
# Condução radial 1D — Esfera (regime permanente, sem geração)
# ============================

@dataclass(frozen=True)
class Sph1DParams:
    k: float     # W/(m·K)
    ri: float    # m
    ro: float    # m
    Ti: float    # K (em r=ri)
    To: float    # K (em r=ro)


def validate_params_sph(p: Sph1DParams) -> None:
    if p.k <= 0:
        raise ValueError("k deve ser > 0 [W/(m·K)].")
    if p.ri <= 0 or p.ro <= 0:
        raise ValueError("ri e ro devem ser > 0 [m].")
    if p.ro <= p.ri:
        raise ValueError("Deve valer ro > ri.")
    if p.Ti <= 0 or p.To <= 0:
        raise ValueError("Temperaturas devem estar em Kelvin (T > 0 K).")


def solve_sphere_steady_1d(p: Sph1DParams, nr: int = 300) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Casca esférica (1D radial), regime permanente, k const, sem geração.
    BCs: T(ri)=Ti, T(ro)=To.

    Retorna:
      r [m], T(r) [K], Qdot [W], qpp(r) [W/m²] onde qpp(r)=Qdot/(4π r²)
    """
    validate_params_sph(p)
    if nr < 20:
        raise ValueError("nr deve ser >= 20.")

    r = np.linspace(p.ri, p.ro, nr)

    # Perfil de temperatura (forma 1/r)
    denom = (1.0 / p.ro - 1.0 / p.ri)
    T = p.Ti + (p.To - p.Ti) * ((1.0 / r - 1.0 / p.ri) / denom)

    # Taxa de calor (constante)
    Qdot = 4.0 * np.pi * p.k * (p.Ti - p.To) / (1.0 / p.ri - 1.0 / p.ro)

    # Fluxo radial por área local
    qpp = Qdot / (4.0 * np.pi * r**2)

    return r, T, Qdot, qpp


def thermal_resistance_sph(p: Sph1DParams) -> float:
    """R_cond,sph = (1/(4πk))*(1/ri - 1/ro) [K/W]."""
    validate_params_sph(p)
    return (1.0 / (4.0 * np.pi * p.k)) * (1.0 / p.ri - 1.0 / p.ro)


# ============================
# Cilindro 1D radial com convecção em r=ro (regime permanente, sem geração)
# BCs: T(ri)=Ti e -k dT/dr|ro = h (T(ro)-Tinf)
# ============================

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class Cyl1DConvParams:
    k: float     # W/(m·K)
    ri: float    # m
    ro: float    # m
    L: float     # m (comprimento)
    Ti: float    # K
    h: float     # W/(m²·K)
    Tinf: float  # K


def validate_params_cyl_conv(p: Cyl1DConvParams) -> None:
    if p.k <= 0:
        raise ValueError("k deve ser > 0 [W/(m·K)].")
    if p.h <= 0:
        raise ValueError("h deve ser > 0 [W/(m²·K)].")
    if p.L <= 0:
        raise ValueError("L deve ser > 0 [m].")
    if p.ri <= 0 or p.ro <= 0:
        raise ValueError("ri e ro devem ser > 0 [m].")
    if p.ro <= p.ri:
        raise ValueError("Deve valer ro > ri.")
    if p.Ti <= 0 or p.Tinf <= 0:
        raise ValueError("Temperaturas devem estar em Kelvin (T > 0 K).")


def thermal_resistance_cyl_conv(p: Cyl1DConvParams) -> Tuple[float, float, float]:
    """
    Retorna (R_cond, R_conv, R_tot) [K/W]
    """
    validate_params_cyl_conv(p)
    R_cond = np.log(p.ro / p.ri) / (2.0 * np.pi * p.k * p.L)
    R_conv = 1.0 / (p.h * 2.0 * np.pi * p.ro * p.L)
    return R_cond, R_conv, R_cond + R_conv


def solve_cylinder_steady_1d_conv(p: Cyl1DConvParams, nr: int = 300) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    """
    Casca cilíndrica (1D radial), regime permanente, sem geração.
    BCs: T(ri)=Ti e convecção em r=ro.
    Retorna:
      r [m], T(r) [K], Qdot [W], qpp(r) [W/m²], Ts = T(ro) [K]
    """
    validate_params_cyl_conv(p)
    if nr < 20:
        raise ValueError("nr deve ser >= 20.")

    r = np.linspace(p.ri, p.ro, nr)

    R_cond, R_conv, R_tot = thermal_resistance_cyl_conv(p)

    # Taxa de calor (constante)
    Qdot = (p.Ti - p.Tinf) / R_tot

    # Perfil de temperatura (a partir de Ti e do termo log)
    T = p.Ti - (Qdot / (2.0 * np.pi * p.k * p.L)) * np.log(r / p.ri)

    # Fluxo local por área
    qpp = Qdot / (2.0 * np.pi * r * p.L)

    Ts = T[-1]  # T(ro)
    return r, T, Qdot, qpp, Ts


def biot_cylinder_external(p: Cyl1DConvParams) -> float:
    """
    Número de Biot com comprimento característico Lc = (ro-ri).
    (Definição didática; não é a única possível.)
    """
    validate_params_cyl_conv(p)
    Lc = (p.ro - p.ri)
    return p.h * Lc / p.k


# ============================
# Meio semi-infinito — condução transiente 1D
# Superfície subitamente imposta
# ============================

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from math import erf, sqrt, pi


@dataclass(frozen=True)
class SemiInfiniteParams:
    k: float      # W/(m·K)
    alpha: float  # m²/s
    Ti: float     # K
    Ts: float     # K


def validate_params_semi(p: SemiInfiniteParams) -> None:
    if p.k <= 0:
        raise ValueError("k deve ser > 0.")
    if p.alpha <= 0:
        raise ValueError("alpha deve ser > 0.")
    if p.Ti <= 0 or p.Ts <= 0:
        raise ValueError("Temperaturas devem estar em Kelvin.")


def temperature_semi_infinite(
    x: np.ndarray, t: float, p: SemiInfiniteParams
) -> np.ndarray:
    """
    T(x,t) para meio semi-infinito com Ts imposto em x=0.
    """
    validate_params_semi(p)
    if t <= 0:
        raise ValueError("t deve ser > 0.")

    eta = x / (2.0 * np.sqrt(p.alpha * t))
    T = p.Ts + (p.Ti - p.Ts) * np.array([erf(e) for e in eta])
    return T


def surface_heat_flux(t: float, p: SemiInfiniteParams) -> float:
    """
    Fluxo de calor na superfície x=0.
    """
    validate_params_semi(p)
    if t <= 0:
        raise ValueError("t deve ser > 0.")

    qpp = p.k * (p.Ti - p.Ts) / np.sqrt(pi * p.alpha * t)
    return qpp
