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


def plot_configuration(ax, Ts, Tinf):
    """
    Esquema didático: placa vertical aquecida (Ts) imersa em fluido em repouso
    (Tinf), com a camada-limite que cresce ao longo da altura e o perfil de
    velocidade característico da convecção natural (sobe junto à parede e
    retorna a zero longe dela).
    """
    quente = Ts >= Tinf

    # Placa vertical (em x=0), altura normalizada [0,1]
    ax.add_patch(Rectangle((-0.06, 0.0), 0.06, 1.0,
                           facecolor=("#c0392b" if quente else "#2471a3"),
                           edgecolor="k", alpha=0.85, zorder=3))

    # Envelope da camada-limite delta(y) ~ y^(1/4) (forma qualitativa)
    y = np.linspace(0.001, 1.0, 200)
    delta = 0.45 * y**0.25
    ax.plot(delta, y, color="0.35", lw=1.8, ls="--", zorder=2,
            label=r"camada-limite $\delta(x)$")

    # Perfis de velocidade u em algumas alturas (forma f'(eta): sobe e volta a 0).
    # Desenhamos o perfil "deitado": o eixo horizontal local representa u, e a
    # curva fica ancorada na altura y0.
    s = np.linspace(0.0, 1.0, 60)                 # s = distância normal / delta
    u_shape = 4.2 * s * np.exp(-2.3 * s)          # forma do perfil de velocidade
    u_shape = u_shape / u_shape.max()             # normalizado a [0,1]
    for j, y0 in enumerate([0.25, 0.55, 0.85]):
        d = 0.45 * y0**0.25                        # espessura local
        u = u_shape * d * 0.9                      # amplitude proporcional a delta
        lbl = "perfil de velocidade $u$" if j == 0 else None
        ax.plot(u, np.full_like(u, y0), color="#117a65", lw=1.6, zorder=4, label=lbl)
        ax.annotate("", xy=(0.035, y0 + 0.07), xytext=(0.035, y0 - 0.03),
                    arrowprops=dict(arrowstyle="->", color="#117a65", lw=1.4))

    # Seta da gravidade
    ax.annotate("g", xy=(0.85, 0.12), xytext=(0.85, 0.30),
                arrowprops=dict(arrowstyle="->", color="k", lw=2.0),
                ha="center", fontsize=12)

    ax.text(-0.03, 1.05, r"$T_s$", ha="center", fontsize=12,
            color=("#c0392b" if quente else "#2471a3"))
    ax.text(0.70, 0.85, r"$T_\infty$ (repouso)", ha="center", fontsize=11, color="0.3")
    ax.text(0.30, -0.07, "u(x) sobe e retorna a zero", ha="center",
            fontsize=9, color="#117a65")

    ax.set_xlim(-0.10, 1.0)
    ax.set_ylim(-0.12, 1.15)
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
    plot_configuration(axes[0, 0], Ts, Tinf)
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
