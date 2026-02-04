# pubensino/ht/fins1d.py
import numpy as np

def geometry_rectangular(w, t):
    """Aleta retangular (placa): Ac=wt, P=2(w+t)."""
    Ac = w * t
    P  = 2.0 * (w + t)
    return Ac, P

def geometry_cylindrical(D):
    """Pino cilíndrico: Ac=pi D^2/4, P=pi D."""
    Ac = np.pi * D**2 / 4.0
    P  = np.pi * D
    return Ac, P

def fin_parameter_m(h, P, k, Ac):
    """m = sqrt(h P / (k Ac))."""
    return np.sqrt(h * P / (k * Ac))

def theta_profile(x, L, m, bc, theta_b, h=None, k=None, theta_L=None):
    """
    Solução analítica para theta(x)=T(x)-T_inf em aleta reta de seção constante.

    bc:
      - "adiabatic_tip": dtheta/dx(L)=0
      - "convective_tip": -k Ac dtheta/dx|L = h Ac theta(L)
      - "prescribed_tip": theta(L)=theta_L
      - "infinite_fin": L->infinito
    """
    x = np.asarray(x, dtype=float)

    if bc == "adiabatic_tip":
        return theta_b * np.cosh(m * (L - x)) / np.cosh(m * L)

    if bc == "convective_tip":
        if (h is None) or (k is None):
            raise ValueError("convective_tip requer h e k.")
        beta = h / (k * m)
        num = np.cosh(m * (L - x)) + beta * np.sinh(m * (L - x))
        den = np.cosh(m * L)       + beta * np.sinh(m * L)
        return theta_b * num / den

    if bc == "prescribed_tip":
        if theta_L is None:
            raise ValueError("prescribed_tip requer theta_L.")
        B = (theta_L - theta_b * np.cosh(m * L)) / np.sinh(m * L)
        return theta_b * np.cosh(m * x) + B * np.sinh(m * x)

    if bc == "infinite_fin":
        return theta_b * np.exp(-m * x)

    raise ValueError("BC inválida.")

def heat_rate_Qf(L, m, bc, theta_b, k, Ac, h=None, theta_L=None):
    """Qf = -k Ac dtheta/dx|_{x=0}."""
    if bc == "adiabatic_tip":
        return k * Ac * m * theta_b * np.tanh(m * L)

    if bc == "convective_tip":
        if h is None:
            raise ValueError("convective_tip requer h.")
        beta = h / (k * m)
        num  = np.sinh(m * L) + beta * np.cosh(m * L)
        den  = np.cosh(m * L) + beta * np.sinh(m * L)
        return k * Ac * m * theta_b * (num / den)

    if bc == "prescribed_tip":
        if theta_L is None:
            raise ValueError("prescribed_tip requer theta_L.")
        B = (theta_L - theta_b * np.cosh(m * L)) / np.sinh(m * L)
        return -k * Ac * (m * B)

    if bc == "infinite_fin":
        return k * Ac * m * theta_b

    raise ValueError("BC inválida.")

def fin_areas(P, L, Ac, include_tip=False):
    """Af ~ P L (+Ac se incluir ponta). Abf = Ac."""
    Af = P * L + (Ac if include_tip else 0.0)
    Abf = Ac
    return Af, Abf

def metrics(Qf, h, Af, Abf, theta_b):
    """(eta_f, eps_f)."""
    eta = Qf / (h * Af  * theta_b) if h * Af  * theta_b != 0 else np.nan
    eps = Qf / (h * Abf * theta_b) if h * Abf * theta_b != 0 else np.nan
    return eta, eps

def fin_array(Qf, h, Ab, Abf, Af, theta_b, Nf):
    """
    Modelo de conjunto (nível de sistema):
      Q_total = Nf Qf + h (Ab - Nf Abf) theta_b
      eta_o = Q_total / (h (Ab + Nf Af) theta_b)
    Impõe: Ab - Nf Abf >= 0.
    """
    Nf_max = int(np.floor(Ab / Abf)) if Abf > 0 else 0
    N_use  = min(int(Nf), max(Nf_max, 0))

    A_exposed = Ab - N_use * Abf
    Q_total   = N_use * Qf + h * A_exposed * theta_b

    denom = h * (Ab + N_use * Af) * theta_b
    eta_o = Q_total / denom if denom != 0 else np.nan

    return Q_total, eta_o, N_use, Nf_max, A_exposed
