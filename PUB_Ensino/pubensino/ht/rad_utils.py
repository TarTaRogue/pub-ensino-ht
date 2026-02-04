import numpy as np
import matplotlib.pyplot as plt

H = 6.62607015e-34
C = 2.99792458e8
KB = 1.380649e-23

SIGMA = 5.670374419e-8
WIEN_B = 2.897771955e-3
C2 = H * C / KB

def E_lambda_blackbody(lam_m: np.ndarray, T: float) -> np.ndarray:
    lam_m = np.asarray(lam_m, dtype=float)
    if np.any(lam_m <= 0):
        raise ValueError("lam_m deve ser positivo (m).")
    if T <= 0:
        raise ValueError("T deve ser positivo (K).")
    a = 2.0 * np.pi * H * C**2
    expo = C2 / (lam_m * T)
    denom = np.expm1(expo)
    return a / (lam_m**5 * denom)

def wien_peak_lambda(T: float) -> float:
    if T <= 0:
        raise ValueError("T deve ser positivo (K).")
    return WIEN_B / T

def integrate_spectral_exitance(lam_m: np.ndarray, E_lam: np.ndarray) -> float:
    lam_m = np.asarray(lam_m, dtype=float)
    E_lam = np.asarray(E_lam, dtype=float)
    if lam_m.ndim != 1 or E_lam.ndim != 1:
        raise ValueError("lam_m e E_lam devem ser vetores 1D.")
    if lam_m.size != E_lam.size:
        raise ValueError("lam_m e E_lam devem ter o mesmo tamanho.")
    if np.any(np.diff(lam_m) <= 0):
        raise ValueError("lam_m deve ser estritamente crescente.")
    return float(np.trapz(E_lam, lam_m))

def plot_blackbody_spectra(lam_um: np.ndarray, T_list, show_wien_peaks: bool = True):
    lam_um = np.asarray(lam_um, dtype=float)
    if np.any(lam_um <= 0):
        raise ValueError("lam_um deve ser positivo (μm).")
    lam_m = lam_um * 1e-6

    plt.figure()
    for T in T_list:
        E = E_lambda_blackbody(lam_m, float(T))
        plt.plot(lam_um, E, label=f"T = {T:.0f} K")
        if show_wien_peaks:
            lam_peak_m = wien_peak_lambda(float(T))
            lam_peak_um = lam_peak_m * 1e6
            if lam_um.min() <= lam_peak_um <= lam_um.max():
                E_peak = E_lambda_blackbody(np.array([lam_peak_m]), float(T))[0]
                plt.plot([lam_peak_um], [E_peak], marker="o")

    plt.xlabel(r"Comprimento de onda, $\lambda$ ($\mu$m)")
    plt.ylabel(r"$E_\lambda^b(\lambda,T)$ (W m$^{-3}$)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.title("Espectro de corpo negro (Lei de Planck)")

def validate_stefan_boltzmann(T: float, lam_um_max: float = 200.0) -> dict:
    if T <= 0:
        raise ValueError("T deve ser positivo (K).")
    lam_um = np.linspace(0.01, lam_um_max, 200000)
    lam_m = lam_um * 1e-6
    E = E_lambda_blackbody(lam_m, T)
    Eb_num = integrate_spectral_exitance(lam_m, E)
    Eb_SB = SIGMA * T**4
    rel_err = (Eb_num - Eb_SB) / Eb_SB
    return {"Eb_num": Eb_num, "Eb_SB": Eb_SB, "rel_err": rel_err}
