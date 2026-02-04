# pubensino/ht/widgets_wall1d.py
# UI mínima (ipywidgets) para Notebook 1

import ipywidgets as widgets
from IPython.display import display

from pubensino.ht.wall1d import Wall1DParams, solve_wall_steady_1d
from pubensino.ht.viz_wall1d import plot_temperature_profile


def wall1d_interactive():
    """
    Cria sliders para (k, L, A, T1, T2) e plota T(x).
    Interatividade mínima: atualizar quando o slider muda.
    """

    # Sliders (faixas conservadoras e coerentes)
    w_k  = widgets.FloatLogSlider(value=15.0, base=10, min=-1, max=3, step=0.01,
                                  description='k [W/mK]', continuous_update=False)
    w_L  = widgets.FloatSlider(value=0.02, min=0.001, max=0.20, step=0.001,
                               description='L [m]', continuous_update=False)
    w_A  = widgets.FloatSlider(value=1.0, min=0.01, max=5.0, step=0.01,
                               description='A [m²]', continuous_update=False)

    # Temperaturas em K
    w_T1 = widgets.FloatSlider(value=350.0, min=250.0, max=600.0, step=1.0,
                               description='T1 [K]', continuous_update=False)
    w_T2 = widgets.FloatSlider(value=300.0, min=250.0, max=600.0, step=1.0,
                               description='T2 [K]', continuous_update=False)

    out = widgets.Output()

    def update(_=None):
        out.clear_output(wait=True)

        p = Wall1DParams(
            k=w_k.value,
            L=w_L.value,
            A=w_A.value,
            T1=w_T1.value,
            T2=w_T2.value
        )

        with out:
            x, T, qpp, Qdot = solve_wall_steady_1d(p, nx=250)
            plot_temperature_profile(x, T, p, qpp, Qdot)

    for w in [w_k, w_L, w_A, w_T1, w_T2]:
        w.observe(update, names="value")

    controls = widgets.VBox([
        widgets.HTML("<b>Parâmetros</b>"),
        w_k, w_L, w_A,
        widgets.HBox([w_T1, w_T2]),
    ])

    update()
    display(widgets.VBox([controls, out]))


def wall1d_conv_interactive():
    import ipywidgets as widgets
    from IPython.display import display

    from pubensino.ht.wall1d import Wall1DConvParams, solve_wall_steady_1d_conv
    from pubensino.ht.viz_wall1d import plot_temperature_profile_conv

    w_k  = widgets.FloatLogSlider(value=15.0, base=10, min=-1, max=3, step=0.01,
                                  description='k [W/mK]', continuous_update=False)
    w_L  = widgets.FloatSlider(value=0.02, min=0.001, max=0.20, step=0.001,
                               description='L [m]', continuous_update=False)
    w_A  = widgets.FloatSlider(value=1.0, min=0.01, max=5.0, step=0.01,
                               description='A [m²]', continuous_update=False)

    w_h  = widgets.FloatLogSlider(value=100.0, base=10, min=0, max=5, step=0.01,
                                  description='h [W/m²K]', continuous_update=False)

    w_T1   = widgets.FloatSlider(value=350.0, min=250.0, max=600.0, step=1.0,
                                 description='T1 [K]', continuous_update=False)
    w_Tinf = widgets.FloatSlider(value=300.0, min=250.0, max=600.0, step=1.0,
                                 description='T∞ [K]', continuous_update=False)

    out = widgets.Output()

    def update(_=None):
        out.clear_output(wait=True)
        p = Wall1DConvParams(k=w_k.value, L=w_L.value, A=w_A.value, T1=w_T1.value, h=w_h.value, Tinf=w_Tinf.value)
        with out:
            x, T, qpp, Qdot, Ts = solve_wall_steady_1d_conv(p, nx=250)
            plot_temperature_profile_conv(x, T, p, qpp, Qdot, Ts)

    for w in [w_k, w_L, w_A, w_h, w_T1, w_Tinf]:
        w.observe(update, names="value")

    controls = widgets.VBox([
        widgets.HTML("<b>Parâmetros (convecção em x=L)</b>"),
        w_k, w_L, w_A, w_h, w_T1, w_Tinf,
    ])

    update()
    display(widgets.VBox([controls, out]))

def cylinder_interactive():
    import ipywidgets as widgets
    from IPython.display import display

    from pubensino.ht.wall1d import Cyl1DParams, solve_cylinder_steady_1d
    from pubensino.ht.viz_wall1d import plot_cylinder_profiles

    w_k  = widgets.FloatLogSlider(value=15.0, base=10, min=-1, max=3, step=0.01,
                                  description='k [W/mK]', continuous_update=False)
    w_ri = widgets.FloatSlider(value=0.01, min=1e-3, max=0.20, step=1e-3,
                               description='ri [m]', continuous_update=False)
    w_ro = widgets.FloatSlider(value=0.03, min=2e-3, max=0.30, step=1e-3,
                               description='ro [m]', continuous_update=False)
    w_L  = widgets.FloatSlider(value=1.0, min=0.05, max=5.0, step=0.05,
                               description='L [m]', continuous_update=False)

    w_Ti = widgets.FloatSlider(value=350.0, min=250.0, max=700.0, step=1.0,
                               description='Ti [K]', continuous_update=False)
    w_To = widgets.FloatSlider(value=300.0, min=250.0, max=700.0, step=1.0,
                               description='To [K]', continuous_update=False)

    out = widgets.Output()

    def update(_=None):
        out.clear_output(wait=True)
        # garantir ro > ri
        ri = w_ri.value
        ro = max(w_ro.value, ri + 1e-4)
        if ro != w_ro.value:
            w_ro.value = ro

        p = Cyl1DParams(k=w_k.value, ri=ri, ro=ro, L=w_L.value, Ti=w_Ti.value, To=w_To.value)
        with out:
            r, T, Qdot, qpp = solve_cylinder_steady_1d(p, nr=350)
            plot_cylinder_profiles(r, T, qpp, p, Qdot)

    for w in [w_k, w_ri, w_ro, w_L, w_Ti, w_To]:
        w.observe(update, names="value")

    controls = widgets.VBox([
        widgets.HTML("<b>Casca cilíndrica (1D radial, regime permanente)</b>"),
        w_k, w_ri, w_ro, w_L, w_Ti, w_To,
    ])

    update()
    display(widgets.VBox([controls, out]))


def sphere_interactive():
    import ipywidgets as widgets
    from IPython.display import display

    from pubensino.ht.wall1d import Sph1DParams, solve_sphere_steady_1d
    from pubensino.ht.viz_wall1d import plot_sphere_profiles

    w_k  = widgets.FloatLogSlider(value=15.0, base=10, min=-1, max=3, step=0.01,
                                  description='k [W/mK]', continuous_update=False)
    w_ri = widgets.FloatSlider(value=0.01, min=1e-3, max=0.20, step=1e-3,
                               description='ri [m]', continuous_update=False)
    w_ro = widgets.FloatSlider(value=0.03, min=2e-3, max=0.30, step=1e-3,
                               description='ro [m]', continuous_update=False)

    w_Ti = widgets.FloatSlider(value=350.0, min=250.0, max=700.0, step=1.0,
                               description='Ti [K]', continuous_update=False)
    w_To = widgets.FloatSlider(value=300.0, min=250.0, max=700.0, step=1.0,
                               description='To [K]', continuous_update=False)

    out = widgets.Output()

    def update(_=None):
        out.clear_output(wait=True)
        ri = w_ri.value
        ro = max(w_ro.value, ri + 1e-4)
        if ro != w_ro.value:
            w_ro.value = ro

        p = Sph1DParams(k=w_k.value, ri=ri, ro=ro, Ti=w_Ti.value, To=w_To.value)
        with out:
            r, T, Qdot, qpp = solve_sphere_steady_1d(p, nr=350)
            plot_sphere_profiles(r, T, qpp, p, Qdot)

    for w in [w_k, w_ri, w_ro, w_Ti, w_To]:
        w.observe(update, names="value")

    controls = widgets.VBox([
        widgets.HTML("<b>Casca esférica (1D radial, regime permanente)</b>"),
        w_k, w_ri, w_ro, w_Ti, w_To,
    ])

    update()
    display(widgets.VBox([controls, out]))


def cylinder_conv_interactive():
    import ipywidgets as widgets
    from IPython.display import display

    from pubensino.ht.wall1d import (
        Cyl1DConvParams,
        solve_cylinder_steady_1d_conv,
        thermal_resistance_cyl_conv,
        biot_cylinder_external
    )
    from pubensino.ht.viz_wall1d import plot_cylinder_profiles_conv

    w_k  = widgets.FloatLogSlider(value=15.0, base=10, min=-1, max=3, step=0.01,
                                  description='k [W/mK]', continuous_update=False)

    w_ri = widgets.FloatSlider(value=0.01, min=1e-3, max=0.20, step=1e-3,
                               description='ri [m]', continuous_update=False)
    w_ro = widgets.FloatSlider(value=0.03, min=2e-3, max=0.30, step=1e-3,
                               description='ro [m]', continuous_update=False)

    w_L  = widgets.FloatSlider(value=1.0, min=0.05, max=5.0, step=0.05,
                               description='L [m]', continuous_update=False)

    w_h  = widgets.FloatLogSlider(value=100.0, base=10, min=0, max=5, step=0.01,
                                  description='h [W/m²K]', continuous_update=False)

    w_Ti   = widgets.FloatSlider(value=350.0, min=250.0, max=800.0, step=1.0,
                                 description='Ti [K]', continuous_update=False)
    w_Tinf = widgets.FloatSlider(value=300.0, min=250.0, max=800.0, step=1.0,
                                 description='T∞ [K]', continuous_update=False)

    out = widgets.Output()

    def update(_=None):
        out.clear_output(wait=True)

        ri = w_ri.value
        ro = max(w_ro.value, ri + 1e-4)  # garante ro>ri
        if ro != w_ro.value:
            w_ro.value = ro

        p = Cyl1DConvParams(k=w_k.value, ri=ri, ro=ro, L=w_L.value, Ti=w_Ti.value, h=w_h.value, Tinf=w_Tinf.value)

        with out:
            r, T, Qdot, qpp, Ts = solve_cylinder_steady_1d_conv(p, nr=350)
            R_cond, R_conv, _ = thermal_resistance_cyl_conv(p)
            Bi = biot_cylinder_external(p)
            plot_cylinder_profiles_conv(r, T, qpp, p, Qdot, Ts, Bi, R_cond, R_conv)

    for w in [w_k, w_ri, w_ro, w_L, w_h, w_Ti, w_Tinf]:
        w.observe(update, names="value")

    controls = widgets.VBox([
        widgets.HTML("<b>Cilindro 1D (T(ri)=Ti; convecção em r=ro)</b>"),
        w_k, w_ri, w_ro, w_L, w_h, w_Ti, w_Tinf,
    ])

    update()
    display(widgets.VBox([controls, out]))


def semi_infinite_interactive():
    import ipywidgets as widgets
    from IPython.display import display
    import numpy as np

    from pubensino.ht.wall1d import (
        SemiInfiniteParams,
        temperature_semi_infinite,
        surface_heat_flux
    )
    from pubensino.ht.viz_wall1d import plot_semi_infinite_profiles_time

    w_k = widgets.FloatLogSlider(value=15.0, base=10, min=-1, max=3, step=0.01,
                                 description='k [W/mK]', continuous_update=False)

    w_alpha = widgets.FloatLogSlider(value=1e-5, base=10, min=-7, max=-3, step=0.01,
                                     description='α [m²/s]', continuous_update=False)

    w_Ti = widgets.FloatSlider(value=350.0, min=250.0, max=800.0, step=1.0,
                               description='Ti [K]', continuous_update=False)

    w_Ts = widgets.FloatSlider(value=300.0, min=250.0, max=800.0, step=1.0,
                               description='Ts [K]', continuous_update=False)

    w_t = widgets.FloatLogSlider(value=1.0, base=10, min=-2, max=4, step=0.01,
                                 description='t_ref [s]', continuous_update=False)
    w_xmax = widgets.FloatLogSlider(
    value=0.01, base=10, min=-4, max=0, step=0.01,
    description='x_max [m]', continuous_update=False
)


    # quantas curvas de tempo mostrar
    w_span = widgets.Dropdown(
        options=[("t/10, t, 10t", "decade"), ("t/4, t/2, t, 2t, 4t", "octave")],
        value="decade",
        description="tempos",
    )

    out = widgets.Output()

    def update(_=None):
        out.clear_output(wait=True)

        p = SemiInfiniteParams(
            k=w_k.value,
            alpha=w_alpha.value,
            Ti=w_Ti.value,
            Ts=w_Ts.value
        )

        t_ref = float(w_t.value)

        # Lista de tempos ao redor de t_ref
        if w_span.value == "decade":
            t_list = np.array([t_ref/10.0, t_ref, 10.0*t_ref], dtype=float)
        else:
            t_list = np.array([t_ref/4.0, t_ref/2.0, t_ref, 2.0*t_ref, 4.0*t_ref], dtype=float)

        # Evitar t muito pequeno (por segurança numérica e física do modelo)
        t_list = np.clip(t_list, 1e-9, None)

        x_max = float(w_xmax.value)
        x = np.linspace(0.0, x_max, 350)


        # Computar perfis e fluxos
        T_list = []
        qpp_list = []
        for tt in t_list:
            T_list.append(temperature_semi_infinite(x, tt, p))
            qpp_list.append(surface_heat_flux(tt, p))

        with out:
            plot_semi_infinite_profiles_time(x, T_list, t_list, p, qpp_list, t_ref)

    for w in [w_k, w_alpha, w_Ti, w_Ts, w_t, w_xmax, w_span,w_xmax]:
        w.observe(update, names="value")

    controls = widgets.VBox([
        widgets.HTML("<b>Meio semi-infinito — múltiplos tempos</b>"),
        w_k, w_alpha, w_Ti, w_Ts, w_t, w_span
    ])

    update()
    display(widgets.VBox([controls, out]))
