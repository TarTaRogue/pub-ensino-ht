# pubensino/ht/viz_wall1d.py
# Rotinas de visualização (plot simples) para Notebook 1

import matplotlib.pyplot as plt


def plot_temperature_profile(x, T, p, qpp, Qdot) -> None:
    """
    Plot mínimo: T(x) em K.
    Inclui painel numérico com q'' e Qdot.
    """
    fig, ax = plt.subplots()
    ax.plot(x, T, linewidth=2.5, label="T(x)")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("T [K]")
    ax.set_title("Parede plana 1D — condução em regime permanente")
    ax.grid(True, alpha=0.25)

    # Painel numérico (mínimo)
    txt = (
        f"k = {p.k:.4g} W/(m·K)\n"
        f"L = {p.L:.4g} m\n"
        f"A = {p.A:.4g} m²\n"
        f"T1 = {p.T1:.4g} K,  T2 = {p.T2:.4g} K\n"
        f"q'' = {qpp:.4g} W/m²\n"
        f"Q̇ = {Qdot:.4g} W"
    )
    ax.text(
        0.02, 0.02, txt, transform=ax.transAxes,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round", alpha=0.10)
    )

    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

def plot_temperature_profile_conv(x, T, p, qpp, Qdot, Ts) -> None:
    fig, ax = plt.subplots()
    ax.plot(x, T, linewidth=2.5, label="T(x)")

    # Marca Ts em x=L
    ax.scatter([p.L], [Ts], zorder=3, label=r"$T_s=T(L)$")

    # Linha de T_inf
    ax.axhline(p.Tinf, linewidth=1.5, linestyle="--", label=r"$T_\infty$")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("T [K]")
    ax.set_title("Parede 1D — T(0)=T1 e convecção em x=L")
    ax.grid(True, alpha=0.25)

    txt = (
        f"k = {p.k:.4g} W/(m·K)\n"
        f"L = {p.L:.4g} m\n"
        f"A = {p.A:.4g} m²\n"
        f"h = {p.h:.4g} W/(m²·K)\n"
        f"T1 = {p.T1:.4g} K,  T∞ = {p.Tinf:.4g} K\n"
        f"Ts = {Ts:.4g} K\n"
        f"q'' = {qpp:.4g} W/m²\n"
        f"Q̇ = {Qdot:.4g} W"
    )
    ax.text(0.02, 0.02, txt, transform=ax.transAxes,
            va="bottom", ha="left", bbox=dict(boxstyle="round", alpha=0.10))

    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_cylinder_profiles(r, T, qpp, p, Qdot) -> None:
    """Duas curvas na mesma figura: T(r) e q''(r) (eixos separados)."""
    fig, ax1 = plt.subplots()

    ax1.plot(r, T, linewidth=2.5, label="T(r)")
    ax1.set_xlabel("r [m]")
    ax1.set_ylabel("T [K]")
    ax1.set_title("Casca cilíndrica 1D — regime permanente")
    ax1.grid(True, alpha=0.25)

    # Segundo eixo para q''(r)
    ax2 = ax1.twinx()
    ax2.plot(r, qpp, linewidth=2.0, linestyle="--", label="q''(r)")
    ax2.set_ylabel("q''(r) [W/m²]")

    txt = (
        f"k = {p.k:.4g} W/(m·K)\n"
        f"ri = {p.ri:.4g} m, ro = {p.ro:.4g} m\n"
        f"L = {p.L:.4g} m\n"
        f"Ti = {p.Ti:.4g} K, To = {p.To:.4g} K\n"
        f"Q̇ = {Qdot:.4g} W"
    )
    ax1.text(0.02, 0.02, txt, transform=ax1.transAxes,
             va="bottom", ha="left", bbox=dict(boxstyle="round", alpha=0.10))

    # Legenda combinada
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="best")

    plt.tight_layout()
    plt.show()


def plot_sphere_profiles(r, T, qpp, p, Qdot) -> None:
    """Duas curvas na mesma figura: T(r) e q''(r) (eixos separados)."""
    fig, ax1 = plt.subplots()

    ax1.plot(r, T, linewidth=2.5, label="T(r)")
    ax1.set_xlabel("r [m]")
    ax1.set_ylabel("T [K]")
    ax1.set_title("Casca esférica 1D — regime permanente")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(r, qpp, linewidth=2.0, linestyle="--", label="q''(r)")
    ax2.set_ylabel("q''(r) [W/m²]")

    txt = (
        f"k = {p.k:.4g} W/(m·K)\n"
        f"ri = {p.ri:.4g} m, ro = {p.ro:.4g} m\n"
        f"Ti = {p.Ti:.4g} K, To = {p.To:.4g} K\n"
        f"Q̇ = {Qdot:.4g} W"
    )
    ax1.text(0.02, 0.02, txt, transform=ax1.transAxes,
             va="bottom", ha="left", bbox=dict(boxstyle="round", alpha=0.10))

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="best")

    plt.tight_layout()
    plt.show()


def plot_cylinder_profiles_conv(r, T, qpp, p, Qdot, Ts, Bi, R_cond, R_conv) -> None:
    """
    Figura única com dois eixos:
      - T(r) (eixo esquerdo)
      - q''(r) (eixo direito)
    Inclui Ts=T(ro) e linha horizontal em Tinf.
    """
    fig, ax1 = plt.subplots()

    ax1.plot(r, T, linewidth=2.5, label="T(r)")
    ax1.scatter([p.ro], [Ts], zorder=3, label=r"$T_s=T(r_o)$")
    ax1.axhline(p.Tinf, linewidth=1.5, linestyle="--", label=r"$T_\infty$")

    ax1.set_xlabel("r [m]")
    ax1.set_ylabel("T [K]")
    ax1.set_title("Cilindro 1D — T(ri)=Ti e convecção em r=ro")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(r, qpp, linewidth=2.0, linestyle="--", label="q''(r)")
    ax2.set_ylabel("q''(r) [W/m²]")

    txt = (
        f"k = {p.k:.4g} W/(m·K)\n"
        f"h = {p.h:.4g} W/(m²·K)\n"
        f"ri = {p.ri:.4g} m, ro = {p.ro:.4g} m, L = {p.L:.4g} m\n"
        f"Ti = {p.Ti:.4g} K, T∞ = {p.Tinf:.4g} K\n"
        f"Ts = {Ts:.4g} K\n"
        f"Q̇ = {Qdot:.4g} W\n"
        f"Bi = {Bi:.4g}\n"
        f"Rcond = {R_cond:.4g} K/W\n"
        f"Rconv = {R_conv:.4g} K/W"
    )
    ax1.text(0.02, 0.02, txt, transform=ax1.transAxes,
             va="bottom", ha="left", bbox=dict(boxstyle="round", alpha=0.10))

    # legenda combinada
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="best")

    plt.tight_layout()
    plt.show()


def plot_semi_infinite_profiles_time(
    x, T_list, t_list, p, qpp_list, t_ref
) -> None:
    """
    Figura única:
      - Curvas T(x,t) para vários tempos (t_list)
      - Inset: q''(0,t) vs t (para os mesmos tempos)
    """
    fig, ax = plt.subplots()

    # Perfis T(x,t)
    for T, tt in zip(T_list, t_list):
        ax.plot(x, T, linewidth=2.0, label=f"t = {tt:.3g} s")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("T [K]")
    ax.set_title("Meio semi-infinito — condução transiente (múltiplos tempos)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    # Painel numérico (referência do slider)
    txt = (
        f"k = {p.k:.4g} W/(m·K)\n"
        f"α = {p.alpha:.4g} m²/s\n"
        f"Ti = {p.Ti:.4g} K\n"
        f"Ts = {p.Ts:.4g} K\n"
        f"t_ref = {t_ref:.4g} s"
    )
    ax.text(
        0.02, 0.02, txt, transform=ax.transAxes,
        va="bottom", ha="left", bbox=dict(boxstyle="round", alpha=0.10)
    )

    # Inset: q''(0,t) vs t (decai ~ t^{-1/2})
    ax_in = ax.inset_axes([0.62, 0.12, 0.35, 0.30])  # [x0, y0, w, h]
    ax_in.plot(t_list, qpp_list, marker="o", linewidth=1.5)
    ax_in.set_xscale("log")
    ax_in.set_yscale("log")
    ax_in.set_xlabel("t [s]", fontsize=9)
    ax_in.set_ylabel("q''(0,t) [W/m²]", fontsize=9)
    ax_in.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()

