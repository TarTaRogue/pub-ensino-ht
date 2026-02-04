# pubensino/ht/widgets_fins1d.py
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output

from . import fins1d
from . import viz_fins1d

def fin_single():
    geom = widgets.Dropdown(
        options=[("Retangular (w,t)", "rect"), ("Cilíndrica (D)", "cyl")],
        value="rect", description="Geometria:"
    )

    w = widgets.FloatText(value=0.02, description="w [m]:")
    t = widgets.FloatText(value=0.001, description="t [m]:")
    D = widgets.FloatText(value=0.005, description="D [m]:")

    k = widgets.FloatText(value=200.0, description="k [W/mK]:")
    h = widgets.FloatText(value=50.0,  description="h [W/m²K]:")
    L = widgets.FloatText(value=0.05,  description="L [m]:")

    Tb   = widgets.FloatText(value=373.0, description="T_b [K]:")
    Tinf = widgets.FloatText(value=293.0, description="T_inf [K]:")

    bc = widgets.Dropdown(
        options=[
            ("Ponta adiabática", "adiabatic_tip"),
            ("Ponta convectiva", "convective_tip"),
            ("Ponta a T prescrita", "prescribed_tip"),
            ("Aleta infinita", "infinite_fin"),
        ],
        value="adiabatic_tip", description="BC ponta:"
    )

    Ttip = widgets.FloatText(value=303.0, description="T_ponta [K]:")
    include_tip = widgets.Checkbox(value=False, description="Incluir ponta em A_f")

    out = widgets.Output()

    def _toggle():
        if geom.value == "rect":
            w.layout.display, t.layout.display, D.layout.display = "", "", "none"
        else:
            w.layout.display, t.layout.display, D.layout.display = "none", "none", ""

        Ttip.layout.display = "" if bc.value == "prescribed_tip" else "none"

    def _update(*args):
        with out:
            clear_output(wait=True)
            _toggle()

            if geom.value == "rect":
                Ac, P = fins1d.geometry_rectangular(w.value, t.value)
                geom_str = "Retangular"
            else:
                Ac, P = fins1d.geometry_cylindrical(D.value)
                geom_str = "Cilíndrica"

            theta_b = Tb.value - Tinf.value
            m = fins1d.fin_parameter_m(h.value, P, k.value, Ac)

            if bc.value == "infinite_fin":
                L_plot = max(L.value, 5.0 / max(m, 1e-12))
            else:
                L_plot = L.value
            x = np.linspace(0.0, L_plot, 400)

            theta_L = (Ttip.value - Tinf.value) if bc.value == "prescribed_tip" else None

            theta = fins1d.theta_profile(x, L.value, m, bc.value, theta_b, h=h.value, k=k.value, theta_L=theta_L)
            Qf = fins1d.heat_rate_Qf(L.value, m, bc.value, theta_b, k=k.value, Ac=Ac, h=h.value, theta_L=theta_L)

            Af, Abf = fins1d.fin_areas(P, L.value, Ac, include_tip=include_tip.value)
            eta, eps = fins1d.metrics(Qf, h.value, Af, Abf, theta_b)

            print("=== Aleta individual ===")
            print(f"Geometria: {geom_str}")
            print(f"A_c = {Ac:.6e} m² | P = {P:.6e} m")
            print(f"m = {m:.6e} 1/m | mL = {(m*L.value):.3f}")
            print(f"Q_f = {Qf:.6f} W")
            print(f"η_f = {eta:.6f} | ε_f = {eps:.6f}")

            T = theta + Tinf.value
            viz_fins1d.plot_profile(x, T, f"T(x) — {bc.label}")

    for wid in [geom, w, t, D, k, h, L, Tb, Tinf, bc, Ttip, include_tip]:
        wid.observe(_update, "value")

    _update()

    uiL = widgets.VBox([geom, w, t, D, k, h, L])
    uiR = widgets.VBox([Tb, Tinf, bc, Ttip, include_tip])
    display(widgets.HBox([uiL, uiR]), out)

def fin_array():
    # conjunto típico: aletas retangulares
    w = widgets.FloatText(value=0.02,  description="w [m]:")
    t = widgets.FloatText(value=0.001, description="t [m]:")

    L = widgets.FloatText(value=0.05,  description="L [m]:")
    k = widgets.FloatText(value=200.0, description="k [W/mK]:")
    h = widgets.FloatText(value=50.0,  description="h [W/m²K]:")

    Tb   = widgets.FloatText(value=373.0, description="T_b [K]:")
    Tinf = widgets.FloatText(value=293.0, description="T_inf [K]:")

    bc = widgets.Dropdown(
        options=[
            ("Ponta adiabática", "adiabatic_tip"),
            ("Ponta convectiva", "convective_tip"),
            ("Aleta infinita", "infinite_fin"),
        ],
        value="adiabatic_tip", description="BC ponta:"
    )

    Ab = widgets.FloatText(value=0.01, description="A_base [m²]:")
    Nf = widgets.IntSlider(value=10, min=1, max=300, step=1, description="N_f:")

    include_tip = widgets.Checkbox(value=False, description="Incluir ponta em A_f")
    out = widgets.Output()

    def _update(*args):
        with out:
            clear_output(wait=True)

            Ac, P = fins1d.geometry_rectangular(w.value, t.value)
            theta_b = Tb.value - Tinf.value
            m = fins1d.fin_parameter_m(h.value, P, k.value, Ac)

            Qf = fins1d.heat_rate_Qf(L.value, m, bc.value, theta_b, k=k.value, Ac=Ac, h=h.value)

            Af, Abf = fins1d.fin_areas(P, L.value, Ac, include_tip=include_tip.value)

            Q_total, eta_o, N_use, N_max, A_exp = fins1d.fin_array(Qf, h.value, Ab.value, Abf, Af, theta_b, Nf.value)

            print("=== Conjunto de aletas ===")
            print(f"A_base = {Ab.value:.6e} m² | A_bf = {Abf:.6e} m²")
            print(f"N_f solicitado = {Nf.value} | N_f usado = {N_use} | N_f,max = {N_max}")
            print(f"Área exposta = {A_exp:.6e} m²")
            print(f"Q_f (1 aleta) = {Qf:.6f} W")
            print(f"Q_total = {Q_total:.6f} W | η_o = {eta_o:.6f}")

            if N_max >= 1:
                Ns = np.arange(1, N_max + 1)
                Qs = np.zeros_like(Ns, dtype=float)
                et = np.zeros_like(Ns, dtype=float)
                for i, n in enumerate(Ns):
                    Qs[i], et[i], *_ = fins1d.fin_array(Qf, h.value, Ab.value, Abf, Af, theta_b, int(n))
                viz_fins1d.plot_vs_Nf(Ns, Qs, et)

    for wid in [w, t, L, k, h, Tb, Tinf, bc, Ab, Nf, include_tip]:
        wid.observe(_update, "value")

    _update()
    display(widgets.VBox([
        widgets.HBox([w, t, L]),
        widgets.HBox([k, h]),
        widgets.HBox([Tb, Tinf]),
        widgets.HBox([bc, include_tip]),
        widgets.HBox([Ab, Nf]),
    ]), out)
