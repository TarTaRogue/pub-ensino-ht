# pubensino/ht/widgets_natconv.py
# Ferramentas interativas (ipywidgets) para Convecção Natural — superfície isotérmica.
# Padrão alinhado aos demais widgets_*.py: Output + _update observando os controles.

from __future__ import annotations

import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown

from . import natconv_iso as nc
from . import viz_natconv as viz


# ------------------------------------------------------------
# Ferramenta 1 — Placa vertical isotérmica (ferramenta principal)
# ------------------------------------------------------------
def vertical_plate():
    """
    Explorador da convecção natural em placa vertical isotérmica.

    Integra: configuração física, grupos adimensionais (Ra, Gr, Pr) avaliados
    na temperatura de filme, solução de similaridade (perfis de velocidade e
    temperatura), espessura da camada-limite e a correlação Nu×Ra de
    Churchill & Chu, com saída de h e q''. Para o regime laminar (Ra ≤ 1e9),
    compara o Nu da similaridade com o da correlação.
    """
    fluido = widgets.Dropdown(
        options=[("Ar (propriedades em T_f)", "ar"),
                 ("Água (Pr alto, ~300 K)", "agua"),
                 ("Mercúrio (Pr baixo, ~300 K)", "mercurio"),
                 ("Personalizado", "custom")],
        value="ar", description="Fluido:",
    )

    Ts   = widgets.FloatSlider(value=350.0, min=290.0, max=900.0, step=5.0,
                               description="T_s [K]:", continuous_update=False)
    Tinf = widgets.FloatSlider(value=300.0, min=250.0, max=600.0, step=5.0,
                               description="T_∞ [K]:", continuous_update=False)
    L    = widgets.FloatSlider(value=0.30, min=0.02, max=3.0, step=0.02,
                               description="L [m]:", continuous_update=False)

    # Propriedades personalizadas (ocultas quando fluido = ar)
    nu_w    = widgets.FloatText(value=15.89e-6, description="ν [m²/s]:")
    alpha_w = widgets.FloatText(value=22.5e-6,  description="α [m²/s]:")
    k_w     = widgets.FloatText(value=0.0263,   description="k [W/mK]:")
    beta_w  = widgets.FloatText(value=3.33e-3,  description="β [1/K]:")

    out = widgets.Output()

    def _toggle():
        disp = "" if fluido.value == "custom" else "none"
        for w in [nu_w, alpha_w, k_w, beta_w]:
            w.layout.display = disp

    def _props():
        Tf = nc.film_temperature(Ts.value, Tinf.value)
        if fluido.value == "ar":
            return nc.air_properties(Tf), Tf
        if fluido.value == "custom":
            return nc.custom_properties(nu_w.value, alpha_w.value, k_w.value, beta_w.value), Tf
        return nc.preset_properties(fluido.value), Tf

    def _update(*args):
        with out:
            clear_output(wait=True)
            _toggle()

            if Ts.value <= 0 or Tinf.value <= 0:
                print("Temperaturas devem estar em Kelvin (>0).")
                return
            if abs(Ts.value - Tinf.value) < 1e-6:
                print("Defina T_s ≠ T_∞ para haver convecção natural.")
                return

            p, Tf = _props()
            s = nc.convection_summary(Ts.value, Tinf.value, L.value, p, "vertical_plate")
            regime = nc.regime_vertical_plate(s["Ra"])

            sim = nc.solve_similarity(p.Pr)
            x = np.linspace(0.0, L.value, 250)
            delta = nc.bl_thickness_vertical(x, Ts.value, Tinf.value, p)

            # Relatório numérico
            print("=== Placa vertical isotérmica ===")
            print(f"T_f = {Tf:.1f} K  |  ΔT = {abs(Ts.value-Tinf.value):.1f} K")
            print(f"Propriedades: ν={p.nu:.3e}  α={p.alpha:.3e}  "
                  f"k={p.k:.4f}  β={p.beta:.3e}  Pr={p.Pr:.3f}")
            if fluido.value in ("agua", "mercurio"):
                print("  (propriedades constantes de referência; só o ar é "
                      "avaliado em T_f)")
            print(f"Ra = {s['Ra']:.3e}  |  Gr = {s['Gr']:.3e}  |  regime: {regime}")
            print(f"Nu (Churchill–Chu) = {s['Nu']:.2f}")
            print(f"h = {s['h']:.2f} W/m²K  |  q'' = {s['qpp']:.1f} W/m²")

            # Sentido do escoamento e do fluxo de calor (depende do sinal de ΔT)
            if Ts.value >= Tinf.value:
                print("Sentido: placa mais quente que o fluido → o fluido sobe "
                      "junto à parede; o calor sai da placa (q'' > 0).")
            else:
                print("Sentido: placa mais fria que o fluido → o fluido desce "
                      "junto à parede; o calor entra na placa (q'' < 0).")
                print("As magnitudes (Ra, Nu, h, |q''|) coincidem com as de um "
                      "caso quente de mesmo |ΔT|: na aproximação de Boussinesq o "
                      "problema é simétrico — o que se inverte é o sentido do "
                      "escoamento e do fluxo de calor (ver configuração física).")

            if regime == "laminar":
                Gr_L = s["Gr"]
                Nu_sim = nc.nu_avg_similarity(Gr_L, sim["minus_theta0p"])
                Nu_cc_lam = nc.nu_vertical_plate_laminar(s["Ra"], p.Pr)
                dif = abs(Nu_sim - Nu_cc_lam) / Nu_cc_lam * 100
                print(f"Nu (similaridade Ostrach)   = {Nu_sim:.2f}")
                print(f"Nu (Churchill–Chu laminar)  = {Nu_cc_lam:.2f}  "
                      f"(diferença {dif:.1f}%)")
            else:
                print("Obs.: Ra > 1e9 → regime turbulento. Os perfis de "
                      "similaridade ilustram o caso laminar (conceito).")

            viz.dashboard_vertical_plate(Ts.value, Tinf.value, L.value, p, s, sim, x, delta)

    for w in [fluido, Ts, Tinf, L, nu_w, alpha_w, k_w, beta_w]:
        w.observe(_update, "value")

    _update()

    controls = widgets.VBox([
        widgets.HBox([fluido]),
        widgets.HBox([Ts, Tinf, L]),
        widgets.HBox([nu_w, alpha_w, k_w, beta_w]),
    ])
    display(controls, out)


# ------------------------------------------------------------
# Ferramenta 2 — Comparação de geometrias isotérmicas
# ------------------------------------------------------------
def geometry_comparison():
    """
    Compara as correlações Nu×Ra de várias geometrias isotérmicas em convecção
    natural e marca o ponto de operação de cada uma para um mesmo fluido,
    ΔT e comprimento característico.
    """
    fluido = widgets.Dropdown(
        options=[("Ar (propriedades em T_f)", "ar"),
                 ("Água (Pr alto, ~300 K)", "agua"),
                 ("Mercúrio (Pr baixo, ~300 K)", "mercurio"),
                 ("Personalizado", "custom")],
        value="ar", description="Fluido:",
    )
    Ts   = widgets.FloatSlider(value=350.0, min=290.0, max=900.0, step=5.0,
                               description="T_s [K]:", continuous_update=False)
    Tinf = widgets.FloatSlider(value=300.0, min=250.0, max=600.0, step=5.0,
                               description="T_∞ [K]:", continuous_update=False)
    Lc   = widgets.FloatSlider(value=0.10, min=0.01, max=2.0, step=0.01,
                               description="Lc [m]:", continuous_update=False)

    nu_w    = widgets.FloatText(value=15.89e-6, description="ν [m²/s]:")
    alpha_w = widgets.FloatText(value=22.5e-6,  description="α [m²/s]:")
    k_w     = widgets.FloatText(value=0.0263,   description="k [W/mK]:")
    beta_w  = widgets.FloatText(value=3.33e-3,  description="β [1/K]:")

    out = widgets.Output()

    def _toggle():
        disp = "" if fluido.value == "custom" else "none"
        for w in [nu_w, alpha_w, k_w, beta_w]:
            w.layout.display = disp

    def _props():
        Tf = nc.film_temperature(Ts.value, Tinf.value)
        if fluido.value == "ar":
            return nc.air_properties(Tf), Tf
        if fluido.value == "custom":
            return nc.custom_properties(nu_w.value, alpha_w.value, k_w.value, beta_w.value), Tf
        return nc.preset_properties(fluido.value), Tf

    def _update(*args):
        with out:
            clear_output(wait=True)
            _toggle()
            if abs(Ts.value - Tinf.value) < 1e-6:
                print("Defina T_s ≠ T_∞ para haver convecção natural.")
                return

            p, Tf = _props()
            Ra = nc.rayleigh(Ts.value, Tinf.value, Lc.value, p)

            print("=== Comparação de geometrias (mesmo Lc) ===")
            print(f"T_f = {Tf:.1f} K | Pr = {p.Pr:.3f} | Lc = {Lc.value:.3g} m | "
                  f"Ra = {Ra:.3e}")
            print(f"{'geometria':<42}{'Nu':>10}{'h [W/m²K]':>14}{'q'' [W/m²]':>14}")
            points = {}
            dT = abs(Ts.value - Tinf.value)
            fora = False
            for key, g in nc.GEOMETRIES.items():
                Nu = g["nu"](Ra, p.Pr)
                h = Nu * p.k / Lc.value
                qpp = h * dT
                points[key] = (Ra, Nu)
                # Verificação das faixas de validade da correlação
                Ra_lo, Ra_hi = g["Ra_range"]
                Pr_lo, Pr_hi = g["Pr_range"]
                issues = []
                if not (Pr_lo <= p.Pr <= Pr_hi):
                    issues.append("Pr")
                if not (Ra_lo <= Ra <= Ra_hi):
                    issues.append("Ra")
                obs = ""
                if issues:
                    fora = True
                    obs = "  (*) fora da faixa: " + ", ".join(issues)
                print(f"{g['label']:<42}{Nu:>10.2f}{h:>14.2f}{qpp:>14.1f}{obs}")
            if fora:
                print("(*) o valor de Nu foi extrapolado além da faixa de "
                      "validade da correlação — use com cautela.")

            viz.compare_geometries(p.Pr, points=points)

    for w in [fluido, Ts, Tinf, Lc, nu_w, alpha_w, k_w, beta_w]:
        w.observe(_update, "value")

    # Texto das correlações aplicadas (uma vez, no topo da ferramenta), para o
    # estudante saber exatamente quais equações estão sendo usadas.
    linhas = ["**Correlações de Nu aplicadas (convecção natural, "
              "superfícies isotérmicas):**", ""]
    for g in nc.GEOMETRIES.values():
        Ra_lo, Ra_hi = g["Ra_range"]
        Pr_lo, Pr_hi = g["Pr_range"]
        pr_txt = "todo Pr" if Pr_lo <= 0 else f"Pr ≥ {Pr_lo:g}"
        linhas.append(f"**{g['label']}**")
        linhas.append(f"$${g['corr']}$$")
        linhas.append(f"Faixa de validade: {Ra_lo:g} ≤ Ra ≤ {Ra_hi:g} "
                      f"&nbsp;•&nbsp; {pr_txt}")
    linhas.append("_Lc é o comprimento característico de cada geometria; as "
                  "propriedades entram via Ra e Pr. Fonte: Bergman/Incropera — "
                  "confira constantes e faixas com o livro do curso._")
    display(Markdown("\n\n".join(linhas)))

    _update()
    display(widgets.VBox([
        widgets.HBox([fluido]),
        widgets.HBox([Ts, Tinf, Lc]),
        widgets.HBox([nu_w, alpha_w, k_w, beta_w]),
    ]), out)
