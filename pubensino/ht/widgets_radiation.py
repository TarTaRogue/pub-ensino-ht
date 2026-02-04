# pubensino/ht/widgets_radiation.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

from . import rad_utils
from . import rad_exchange
from . import view_factors_mc

def blackbody_spectrum():
    T = widgets.IntSlider(value=800, min=200, max=3000, step=50, description="T [K]:")
    lam_max = widgets.FloatSlider(value=30.0, min=5.0, max=200.0, step=1.0, description="λ_max [μm]:")
    npts = widgets.IntSlider(value=4000, min=500, max=20000, step=500, description="N_pts:")
    out = widgets.Output()

    def _update(*args):
        with out:
            clear_output(wait=True)
            lam_um = np.linspace(0.05, lam_max.value, npts.value)
            rad_utils.plot_blackbody_spectra(lam_um, [float(T.value)], show_wien_peaks=True)
            lam_peak = rad_utils.wien_peak_lambda(float(T.value))*1e6
            Eb_SB = rad_exchange.SIGMA * float(T.value)**4
            print(f"λ_pico (Wien) ≈ {lam_peak:.3f} μm")
            print(f"E_b = σT^4 ≈ {Eb_SB:.3e} W/m²")
            plt.show()

    for w in [T, lam_max, npts]:
        w.observe(_update, "value")
    _update()
    display(widgets.VBox([widgets.HBox([T, lam_max, npts]), out]))

def grey_enclosure():
    A = widgets.FloatText(value=0.10, description="A [m²]:")
    eps = widgets.FloatSlider(value=0.8, min=0.05, max=1.0, step=0.01, description="ε:")
    Ts = widgets.FloatSlider(value=600.0, min=250.0, max=2000.0, step=10.0, description="T_s [K]:")
    Tsur = widgets.FloatSlider(value=300.0, min=250.0, max=1200.0, step=10.0, description="T_sur [K]:")
    out = widgets.Output()

    def _update(*args):
        with out:
            clear_output(wait=True)
            q = float(rad_exchange.net_to_large_enclosure(A.value, eps.value, Ts.value, Tsur.value))
            Tm = 0.5*(Ts.value + Tsur.value)
            hr = rad_exchange.linearized_hr(eps.value, Tm)
            q_lin = hr * A.value * (Ts.value - Tsur.value)
            print("=== Superfície ↔ Envoltória grande (F=1) ===")
            print(f"q_rad = {q:.3f} W")
            print(f"h_r = {hr:.3f} W/m²K | q_lin = {q_lin:.3f} W")

            Ts_vec = np.linspace(max(1.0, Ts.value*0.6), Ts.value*1.4, 200)
            q_vec = rad_exchange.net_to_large_enclosure(A.value, eps.value, Ts_vec, Tsur.value)
            plt.figure()
            plt.plot(Ts_vec, q_vec)
            plt.xlabel(r"$T_s$ [K]"); plt.ylabel(r"$q_{rad}$ [W]")
            plt.grid(True, alpha=0.3)
            plt.title("Não-linearidade de $q_{rad}(T_s)$")
            plt.show()

    for w in [A, eps, Ts, Tsur]:
        w.observe(_update, "value")
    _update()
    display(widgets.VBox([widgets.HBox([A, eps]), widgets.HBox([Ts, Tsur]), out]))

def two_surfaces_unitF():
    A = widgets.FloatText(value=0.10, description="A [m²]:")
    eps1 = widgets.FloatSlider(value=0.8, min=0.05, max=1.0, step=0.01, description="ε1:")
    eps2 = widgets.FloatSlider(value=0.6, min=0.05, max=1.0, step=0.01, description="ε2:")
    T1 = widgets.FloatSlider(value=800.0, min=250.0, max=2000.0, step=10.0, description="T1 [K]:")
    T2 = widgets.FloatSlider(value=300.0, min=250.0, max=2000.0, step=10.0, description="T2 [K]:")
    out = widgets.Output()

    def _update(*args):
        with out:
            clear_output(wait=True)
            eps_eff = rad_exchange.effective_emissivity_two_surfaces_unitF(eps1.value, eps2.value)
            q = rad_exchange.net_between_two_surfaces_unitF(A.value, eps1.value, eps2.value, T1.value, T2.value)
            print("=== Duas superfícies, F12=1 ===")
            print(f"ε_eff = {eps_eff:.4f} | q = {q:.3f} W")

            eps2_vec = np.linspace(0.05, 1.0, 200)
            q_vec = np.array([rad_exchange.net_between_two_surfaces_unitF(A.value, eps1.value, e2, T1.value, T2.value) for e2 in eps2_vec])
            plt.figure()
            plt.plot(eps2_vec, q_vec)
            plt.xlabel(r"$\epsilon_2$"); plt.ylabel(r"$q$ [W]")
            plt.grid(True, alpha=0.3)
            plt.title("Efeito de $\epsilon_2$")
            plt.show()

    for w in [A, eps1, eps2, T1, T2]:
        w.observe(_update, "value")
    _update()
    display(widgets.VBox([widgets.HBox([A, eps1, eps2]), widgets.HBox([T1, T2]), out]))

def view_factor_parallel_rectangles():
    Lx1 = widgets.FloatText(value=0.20, description="Lx1 [m]:")
    Ly1 = widgets.FloatText(value=0.20, description="Ly1 [m]:")
    Lx2 = widgets.FloatText(value=0.20, description="Lx2 [m]:")
    Ly2 = widgets.FloatText(value=0.20, description="Ly2 [m]:")
    H  = widgets.FloatText(value=0.10, description="H [m]:")
    dx = widgets.FloatText(value=0.00, description="dx [m]:")
    dy = widgets.FloatText(value=0.00, description="dy [m]:")
    n = widgets.IntSlider(value=20000, min=2000, max=200000, step=2000, description="N MC:")
    out = widgets.Output()

    def _update(*args):
        with out:
            clear_output(wait=True)
            res12 = view_factors_mc.view_factor_parallel_rectangles_mc(int(n.value), Lx1.value, Ly1.value, Lx2.value, Ly2.value, H.value, dx.value, dy.value, seed=0)
            res21 = view_factors_mc.view_factor_parallel_rectangles_mc(int(n.value), Lx2.value, Ly2.value, Lx1.value, Ly1.value, H.value, -dx.value, -dy.value, seed=1)
            A1, A2 = res12["A1"], res12["A2"]
            recip_err = abs(A1*res12["F12"] - A2*res21["F12"])
            print("=== Fator de forma (MC) — retângulos paralelos ===")
            print(f"F12 ≈ {res12['F12']:.5f} (± {res12['sigma_F12']:.5f}, 1σ)")
            print(f"F21 ≈ {res21['F12']:.5f} | |A1F12-A2F21| ≈ {recip_err:.2e}")

            plt.figure()
            x1 = np.array([-Lx1.value/2, Lx1.value/2, Lx1.value/2, -Lx1.value/2, -Lx1.value/2])
            y1 = np.array([-Ly1.value/2, -Ly1.value/2, Ly1.value/2, Ly1.value/2, -Ly1.value/2])
            x2 = np.array([-Lx2.value/2, Lx2.value/2, Lx2.value/2, -Lx2.value/2, -Lx2.value/2]) + dx.value
            y2 = np.array([-Ly2.value/2, -Ly2.value/2, Ly2.value/2, Ly2.value/2, -Ly2.value/2]) + dy.value
            plt.plot(x1, y1, label="Retângulo 1"); plt.plot(x2, y2, label="Retângulo 2")
            plt.axis("equal"); plt.grid(True, alpha=0.3)
            plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.legend()
            plt.title("Projeção no plano xy")
            plt.show()

    for w in [Lx1, Ly1, Lx2, Ly2, H, dx, dy, n]:
        w.observe(_update, "value")
    _update()
    display(widgets.VBox([widgets.HBox([Lx1, Ly1, Lx2, Ly2]), widgets.HBox([H, dx, dy, n]), out]))
