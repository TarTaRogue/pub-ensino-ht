# pubensino/ht/viz_natconv.py
# Rotinas de visualização para o notebook de Convecção Natural (superfície isotérmica).
# Estilo alinhado aos demais viz_*.py: figuras matplotlib simples, grid leve e
# painel numérico no canto.

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from . import natconv_iso as nc


def _info_box(ax, txt):
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, va="bottom", ha="left",
            fontsize=9, bbox=dict(boxstyle="round", alpha=0.10))


def _bl_edge(eta, prof, frac):
    """Maior eta em que 'prof' ainda é >= frac — usado como borda da camada-limite."""
    idx = np.where(prof >= frac)[0]
    return float(eta[idx[-1]]) if len(idx) else float(eta[-1])


def plot_configuration(ax, Ts, Tinf, sim=None, Pr=None):
    """
    Esquema didático da placa vertical isotérmica imersa em fluido em repouso.

    Sentido (sinal de Ts - Tinf):
      * placa quente (Ts > Tinf): o fluido aquece, fica menos denso e SOBE — a
        camada-limite nasce no bordo inferior e cresce para cima; o calor sai.
      * placa fria (Ts < Tinf): o fluido resfria, fica mais denso e DESCE — a
        camada-limite nasce no bordo superior e cresce para baixo; o calor entra.

    Quando a solução de similaridade `sim` é fornecida, o desenho fica fiel ao
    fluido: o perfil de velocidade usa a forma real de f'(eta) (pico junto à
    parede e cauda) e são traçadas DUAS camadas-limite — de velocidade
    (delta_u) e térmica (delta_t) — na proporção real delta_t/delta_u = eta_t/eta_u
    extraída dos perfis. Assim, para Pr alto (água) a CL térmica aparece bem
    mais fina que a de velocidade; para Pr baixo (mercúrio), comparável ou mais
    espessa.
    """
    quente = Ts >= Tinf
    flow = 1.0 if quente else -1.0                 # +1 sobe, -1 desce
    cor = "#c0392b" if quente else "#2471a3"

    # Placa vertical (em x=0), altura normalizada [0,1]
    ax.add_patch(Rectangle((-0.06, 0.0), 0.06, 1.0,
                           facecolor=cor, edgecolor="k", alpha=0.85, zorder=3))

    # Forma do perfil de velocidade e razão de espessuras a partir da similaridade
    if sim is not None:
        eta = np.asarray(sim["eta"], dtype=float)
        fpn = np.asarray(sim["fp"], dtype=float); fpn = fpn / fpn.max()
        thn = np.asarray(sim["theta"], dtype=float)
        eta_u = _bl_edge(eta, fpn, 0.01)           # borda da CL de velocidade
        eta_t = _bl_edge(eta, thn, 0.01)           # borda da CL térmica
        eta_ref = max(eta_u, eta_t)
        mask = eta <= eta_u
        prof_x = eta[mask] / eta_ref               # fração da extensão de referência
        prof_u = fpn[mask]
        ratio_u, ratio_t = eta_u / eta_ref, eta_t / eta_ref
    else:
        # fallback qualitativo (Pr ~ 1): forma genérica, uma única camada-limite
        s = np.linspace(0.0, 1.0, 80)
        prof_u = s * np.exp(1.0 - 5.0 * s); prof_u = prof_u / prof_u.max()
        prof_x = s
        ratio_u = ratio_t = 1.0

    d_max = 0.45                                   # extensão da CL mais espessa, no topo
    y = np.linspace(0.0, 1.0, 200)
    xi = np.clip(y if quente else (1.0 - y), 0.0, None)
    delta_u = d_max * ratio_u * xi ** 0.25
    delta_t = d_max * ratio_t * xi ** 0.25

    # Envelopes das camadas-limite
    ax.plot(delta_u, y, color="#117a65", lw=1.6, ls="--", zorder=2,
            label=r"CL de velocidade $\delta_u$")
    if sim is not None:
        ax.plot(delta_t, y, color=cor, lw=1.6, ls=":", zorder=2,
                label=r"CL térmica $\delta_t$")

    # Perfis de velocidade (forma real f') em estações ao longo do escoamento
    amp = 0.15
    x_peak_frac = prof_x[int(np.argmax(prof_u))]
    for j, xq in enumerate([0.34, 0.64, 0.94]):
        y0 = xq if quente else (1.0 - xq)          # altura da estação
        scale = d_max * xq ** 0.25                 # extensão local de referência
        xs = prof_x * scale                        # eixo horizontal = distância normal
        ys = y0 + flow * amp * prof_u              # deflexão vertical = magnitude de u
        lbl = "perfil de $u$" if j == 0 else None
        ax.plot(xs, ys, color="#117a65", lw=1.7, zorder=4, label=lbl)
        ax.plot([0.0, ratio_u * scale], [y0, y0], color="#117a65", lw=0.7,
                ls=":", alpha=0.6, zorder=4)       # eixo normal local (rake)
        ax.annotate("", xy=(x_peak_frac * scale, y0 + flow * amp * 1.18),
                    xytext=(x_peak_frac * scale, y0),
                    arrowprops=dict(arrowstyle="->", color="#117a65", lw=1.2))

    # Seta global do sentido do escoamento, junto à placa
    ya0, ya1 = (0.12, 0.88) if quente else (0.88, 0.12)
    ax.annotate("", xy=(0.115, ya1), xytext=(0.115, ya0),
                arrowprops=dict(arrowstyle="-|>", color="#117a65", lw=2.2, alpha=0.5))

    # Sentido do fluxo de calor q'': sai da placa (quente) ou entra (fria)
    if quente:
        ax.annotate("", xy=(0.095, 0.50), xytext=(0.0, 0.50),
                    arrowprops=dict(arrowstyle="-|>", color=cor, lw=1.8))
        ax.text(0.105, 0.50, r"$q''$ sai", color=cor, fontsize=9, va="center")
    else:
        ax.annotate("", xy=(0.0, 0.50), xytext=(0.095, 0.50),
                    arrowprops=dict(arrowstyle="-|>", color=cor, lw=1.8))
        ax.text(0.105, 0.50, r"$q''$ entra", color=cor, fontsize=9, va="center")

    # Gravidade (sempre para baixo)
    ax.annotate("g", xy=(0.92, 0.12), xytext=(0.92, 0.30),
                arrowprops=dict(arrowstyle="->", color="k", lw=2.0),
                ha="center", fontsize=12)

    ax.text(-0.03, 1.05, r"$T_s$", ha="center", fontsize=12, color=cor)
    ax.text(0.72, 0.55, r"$T_\infty$" + "\n(repouso)", ha="center",
            fontsize=10, color="0.4")
    sub = "fluido aquece e sobe" if quente else "fluido resfria e desce"
    if Pr is not None:
        sub += f"   |   Pr = {Pr:.3g}"
    ax.text(0.5, -0.11, f"placa {'quente' if quente else 'fria'} — {sub}",
            ha="center", fontsize=9, color=cor)

    ax.set_xlim(-0.10, 1.05)
    ax.set_ylim(-0.16, 1.15)
    ax.set_xlabel("distância normal à placa →")
    ax.set_ylabel("altura ao longo da placa →")
    ax.set_title("Configuração física")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)


def plot_similarity(ax, sim, Pr):
    """Perfis de similaridade: velocidade f'(eta) e temperatura theta(eta)."""
    ax.plot(sim["fp"], sim["eta"], lw=2.5, color="#117a65", label=r"$f'(\eta)$ (velocidade)")
    ax.plot(sim["theta"], sim["eta"], lw=2.5, color="#c0392b", label=r"$\theta(\eta)$ (temperatura)")
    ax.set_xlabel(r"$f'(\eta)$  e  $\theta(\eta)$")
    ax.set_ylabel(r"$\eta = (Gr_x/4)^{1/4}\,(y/x)$")
    ax.set_title("Perfis de similaridade (Ostrach)")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, min(sim["eta"][-1], 8.0))
    _info_box(ax, f"Pr = {Pr:.3g}\n"
                  fr"$-\theta'(0)$ = {sim['minus_theta0p']:.4f}")
    ax.legend(loc="upper right", fontsize=9)


def plot_bl_growth(ax, x, delta, L):
    """Espessura da camada-limite delta(x) ao longo da placa."""
    ax.plot(x * 1e3, delta * 1e3, lw=2.5, color="0.3")
    ax.fill_between(x * 1e3, 0, delta * 1e3, alpha=0.12, color="0.5")
    ax.set_xlabel("x — altura na placa [mm]")
    ax.set_ylabel(r"$\delta(x)$ [mm]")
    ax.set_title("Crescimento da camada-limite (laminar)")
    ax.grid(True, alpha=0.25)
    _info_box(ax, f"L = {L*1e3:.0f} mm\n"
                  fr"$\delta(L)$ = {delta[-1]*1e3:.2f} mm")


def plot_nu_ra(ax, geometry, Pr, Ra_pt=None, Nu_pt=None):
    """
    Curva Nu × Ra (log-log) para a geometria escolhida, com o ponto de operação.
    Mostra o comportamento de lei de potência e a faixa de validade.
    """
    g = nc.GEOMETRIES[geometry]
    Ra_lo, Ra_hi = g["Ra_range"]
    Ra = np.logspace(np.log10(Ra_lo), np.log10(Ra_hi), 300)
    Nu = np.array([g["nu"](r, Pr) for r in Ra])

    ax.loglog(Ra, Nu, lw=2.5, color="#1f4e79", label=g["label"])

    if geometry == "vertical_plate":
        ax.axvline(1e9, color="0.5", ls=":", lw=1.5)
        ax.text(1e9, Nu.min() * 1.3, " transição\n laminar→turb.\n (Ra≈10⁹)",
                fontsize=8, color="0.4", va="bottom")

    if Ra_pt is not None and Nu_pt is not None:
        ax.loglog([Ra_pt], [Nu_pt], "o", ms=10, color="#c0392b", zorder=5,
                  label="ponto de operação")
        ax.annotate(f"  Ra={Ra_pt:.2e}\n  Nu={Nu_pt:.1f}",
                    xy=(Ra_pt, Nu_pt), fontsize=9, color="#c0392b",
                    va="center")

    ax.set_xlabel("Ra")
    ax.set_ylabel("Nu")
    ax.set_title("Correlação Nu × Ra")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)


# ------------------------------------------------------------
# Painéis compostos usados pelas ferramentas interativas
# ------------------------------------------------------------

def dashboard_vertical_plate(Ts, Tinf, L, props, summary, sim, x, delta):
    """Figura 2x2 para a ferramenta da placa vertical."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    plot_configuration(axes[0, 0], Ts, Tinf, sim=sim, Pr=props.Pr)
    plot_similarity(axes[0, 1], sim, props.Pr)
    plot_bl_growth(axes[1, 0], x, delta, L)
    plot_nu_ra(axes[1, 1], "vertical_plate", props.Pr,
               Ra_pt=summary["Ra"], Nu_pt=summary["Nu"])
    fig.suptitle("Convecção natural — placa vertical isotérmica", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def compare_geometries(Pr, points=None):
    """
    Sobrepõe Nu × Ra de várias geometrias isotérmicas para comparação.
    points: dict opcional {geometry: (Ra, Nu)} para marcar pontos de operação.
    """
    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors = ["#1f4e79", "#c0392b", "#117a65", "#b9770e", "#6c3483"]
    for (key, g), c in zip(nc.GEOMETRIES.items(), colors):
        Ra_lo, Ra_hi = g["Ra_range"]
        Ra = np.logspace(np.log10(Ra_lo), np.log10(Ra_hi), 300)
        Nu = np.array([g["nu"](r, Pr) for r in Ra])
        ax.loglog(Ra, Nu, lw=2.2, color=c, label=g["label"])
        if points and key in points:
            Ra_pt, Nu_pt = points[key]
            ax.loglog([Ra_pt], [Nu_pt], "o", ms=9, color=c, zorder=5)

    ax.set_xlabel("Ra")
    ax.set_ylabel("Nu")
    ax.set_title(f"Comparação de geometrias isotérmicas (Pr = {Pr:.3g})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    plt.show()
