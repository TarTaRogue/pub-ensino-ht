# pubensino/ht/natconv_iso.py
# Convecção natural (livre) em superfícies isotérmicas
# Convenção: temperaturas em Kelvin (K); unidades SI.
#
# Conteúdo:
#   (1) Propriedades do fluido avaliadas na temperatura de filme T_f=(Ts+Tinf)/2
#   (2) Grupos adimensionais Ra, Gr, Pr
#   (3) Correlações de Nu para geometrias isotérmicas clássicas
#   (4) Solução de similaridade (Ostrach) da camada-limite laminar na placa
#       vertical isotérmica — resolução numérica das equações de transporte
#
# Referência das correlações: Bergman/Incropera, "Fundamentos de Transferência
# de Calor e de Massa". Verifique as tabelas/constantes com o livro do seu curso.

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
from scipy.integrate import solve_bvp

G = 9.80665  # aceleração da gravidade [m/s²]


# ============================================================
# (1) Propriedades do fluido
# ============================================================

# Ar seco a 1 atm — tabela tipo Incropera (Tab. A.4).
# Colunas: T[K], nu[m²/s], alpha[m²/s], k[W/(m·K)], Pr.
# OBS.: confira estes valores com a tabela do seu curso antes de usar em prova.
_AIR_T     = np.array([250, 300, 350, 400, 450, 500, 550, 600], dtype=float)        # K
_AIR_NU    = np.array([11.44, 15.89, 20.92, 26.41, 32.39, 38.79, 45.57, 52.69]) * 1e-6   # m²/s
_AIR_ALPHA = np.array([15.9, 22.5, 29.9, 38.3, 47.2, 56.7, 66.7, 76.9]) * 1e-6           # m²/s
_AIR_K     = np.array([22.3, 26.3, 30.0, 33.8, 37.3, 40.7, 43.9, 46.9]) * 1e-3           # W/(m·K)
_AIR_PR    = np.array([0.720, 0.707, 0.700, 0.690, 0.686, 0.684, 0.683, 0.685])


@dataclass(frozen=True)
class FluidProps:
    """Propriedades termofísicas avaliadas em uma dada temperatura."""
    nu: float     # viscosidade cinemática [m²/s]
    alpha: float  # difusividade térmica [m²/s]
    k: float      # condutividade térmica [W/(m·K)]
    beta: float   # coef. de expansão volumétrica [1/K]
    Pr: float     # número de Prandtl [-]


def film_temperature(Ts: float, Tinf: float) -> float:
    """Temperatura de filme T_f = (Ts + Tinf)/2 [K]."""
    return 0.5 * (Ts + Tinf)


def air_properties(Tf: float) -> FluidProps:
    """
    Propriedades do ar a 1 atm avaliadas em T_f [K], por interpolação da tabela.
    Gás ideal: beta = 1/T_f.
    """
    if Tf <= 0:
        raise ValueError("T_f deve estar em Kelvin (>0).")
    if not (_AIR_T[0] <= Tf <= _AIR_T[-1]):
        # extrapolação é permitida pelo np.interp (clampa nas bordas), mas avisamos
        pass
    nu    = float(np.interp(Tf, _AIR_T, _AIR_NU))
    alpha = float(np.interp(Tf, _AIR_T, _AIR_ALPHA))
    k     = float(np.interp(Tf, _AIR_T, _AIR_K))
    Pr    = float(np.interp(Tf, _AIR_T, _AIR_PR))
    return FluidProps(nu=nu, alpha=alpha, k=k, beta=1.0 / Tf, Pr=Pr)


def custom_properties(nu: float, alpha: float, k: float, beta: float) -> FluidProps:
    """Constrói FluidProps a partir de propriedades constantes informadas (Pr=nu/alpha)."""
    for name, val in [("nu", nu), ("alpha", alpha), ("k", k), ("beta", beta)]:
        if val <= 0:
            raise ValueError(f"{name} deve ser > 0.")
    return FluidProps(nu=nu, alpha=alpha, k=k, beta=beta, Pr=nu / alpha)


# Fluidos pré-definidos com propriedades CONSTANTES, avaliadas na temperatura de
# referência indicada (Tref). Permitem explorar o efeito do número de Prandtl
# nas correlações e na solução de similaridade:
#   - água:     Pr alto  (~5,8 a 300 K) -> camada-limite térmica fina, perfil de
#               temperatura abrupto junto à parede (-theta'(0) grande);
#   - mercúrio: Pr baixo (~0,025 a 300 K) -> camada-limite térmica espessa, muito
#               maior que a de velocidade (típico de metais líquidos).
# Fonte: Bergman/Incropera (Tab. A.6 água saturada; Tab. A.5 mercúrio).
# OBS.: propriedades de líquidos variam fortemente com a temperatura (sobretudo
# o beta da água); aqui são mantidas constantes por simplicidade didática. Para
# um ponto específico, use o modo "Personalizado". Confira os valores com o livro.
PRESET_FLUIDS: Dict[str, Dict] = {
    "agua": dict(label="Água (Pr alto, ~300 K)", Tref=300.0,
                 nu=8.55e-7, alpha=1.47e-7, k=0.613, beta=276.1e-6),
    "mercurio": dict(label="Mercúrio (Pr baixo, ~300 K)", Tref=300.0,
                     nu=1.13e-7, alpha=4.56e-6, k=8.54, beta=1.82e-4),
}


def preset_properties(name: str) -> FluidProps:
    """Propriedades constantes de um fluido pré-definido (ver PRESET_FLUIDS)."""
    if name not in PRESET_FLUIDS:
        raise ValueError(f"Fluido pré-definido desconhecido: {name}")
    f = PRESET_FLUIDS[name]
    return custom_properties(f["nu"], f["alpha"], f["k"], f["beta"])


# ============================================================
# (2) Grupos adimensionais
# ============================================================

def rayleigh(Ts: float, Tinf: float, Lc: float, props: FluidProps) -> float:
    """Ra = g·beta·|Ts-Tinf|·Lc³ / (nu·alpha)."""
    dT = abs(Ts - Tinf)
    return G * props.beta * dT * Lc**3 / (props.nu * props.alpha)


def grashof(Ts: float, Tinf: float, Lc: float, props: FluidProps) -> float:
    """Gr = g·beta·|Ts-Tinf|·Lc³ / nu²  (note que Ra = Gr·Pr)."""
    dT = abs(Ts - Tinf)
    return G * props.beta * dT * Lc**3 / props.nu**2


# ============================================================
# (3) Correlações de Nusselt — superfícies isotérmicas
# ============================================================

def nu_vertical_plate(Ra: float, Pr: float) -> float:
    """
    Placa vertical isotérmica — Churchill & Chu, válida em toda a faixa de Ra:
        Nu = { 0.825 + 0.387 Ra^(1/6) / [1+(0.492/Pr)^(9/16)]^(8/27) }²
    """
    return (0.825 + (0.387 * Ra**(1/6)) /
            (1.0 + (0.492 / Pr)**(9/16))**(8/27))**2


def nu_vertical_plate_laminar(Ra: float, Pr: float) -> float:
    """
    Placa vertical isotérmica, faixa laminar (Ra ≤ 1e9) — Churchill & Chu:
        Nu = 0.68 + 0.670 Ra^(1/4) / [1+(0.492/Pr)^(9/16)]^(4/9)
    """
    return 0.68 + (0.670 * Ra**0.25) / (1.0 + (0.492 / Pr)**(9/16))**(4/9)


def nu_horizontal_cylinder(Ra: float, Pr: float) -> float:
    """
    Cilindro horizontal longo isotérmico — Churchill & Chu (Ra ≤ 1e12), Lc = D:
        Nu = { 0.60 + 0.387 Ra^(1/6) / [1+(0.559/Pr)^(9/16)]^(8/27) }²
    """
    return (0.60 + (0.387 * Ra**(1/6)) /
            (1.0 + (0.559 / Pr)**(9/16))**(8/27))**2


def nu_sphere(Ra: float, Pr: float) -> float:
    """
    Esfera isotérmica — Churchill (Ra ≤ 1e11, Pr ≥ 0.7), Lc = D:
        Nu = 2 + 0.589 Ra^(1/4) / [1+(0.469/Pr)^(9/16)]^(4/9)
    """
    return 2.0 + (0.589 * Ra**0.25) / (1.0 + (0.469 / Pr)**(9/16))**(4/9)


def nu_horizontal_plate_hot_up(Ra: float) -> float:
    """
    Placa horizontal — superfície quente voltada p/ cima (ou fria p/ baixo).
    Lc = A_s/P. Faixas (Incropera):
        Nu = 0.54 Ra^(1/4)   (1e4 ≤ Ra ≤ 1e7)
        Nu = 0.15 Ra^(1/3)   (1e7 ≤ Ra ≤ 1e11)
    """
    if Ra < 1e7:
        return 0.54 * Ra**0.25
    return 0.15 * Ra**(1/3)


def nu_horizontal_plate_hot_down(Ra: float) -> float:
    """
    Placa horizontal — superfície quente voltada p/ baixo (ou fria p/ cima).
    Lc = A_s/P:
        Nu = 0.27 Ra^(1/4)   (1e5 ≤ Ra ≤ 1e10)
    """
    return 0.27 * Ra**0.25


# Mapa de geometrias para uso nas ferramentas interativas.
# Cada entrada: (rótulo, função Nu, comprimento característico, faixa de validade)
GEOMETRIES = {
    "vertical_plate": dict(
        label="Placa vertical (L = altura)",
        nu=lambda Ra, Pr: nu_vertical_plate(Ra, Pr),
        Lc_label="L (altura)",
        Ra_range=(1e-1, 1e13),
        Pr_range=(0.0, float("inf")),
        corr=r"Nu = \left\{0{,}825 + \dfrac{0{,}387\,Ra^{1/6}}"
             r"{\left[1+(0{,}492/Pr)^{9/16}\right]^{8/27}}\right\}^{2}",
    ),
    "horizontal_cylinder": dict(
        label="Cilindro horizontal (L = D)",
        nu=lambda Ra, Pr: nu_horizontal_cylinder(Ra, Pr),
        Lc_label="D (diâmetro)",
        Ra_range=(1e-5, 1e12),
        Pr_range=(0.0, float("inf")),
        corr=r"Nu = \left\{0{,}60 + \dfrac{0{,}387\,Ra^{1/6}}"
             r"{\left[1+(0{,}559/Pr)^{9/16}\right]^{8/27}}\right\}^{2}",
    ),
    "sphere": dict(
        label="Esfera (L = D)",
        nu=lambda Ra, Pr: nu_sphere(Ra, Pr),
        Lc_label="D (diâmetro)",
        Ra_range=(1e0, 1e11),
        Pr_range=(0.7, float("inf")),
        corr=r"Nu = 2 + \dfrac{0{,}589\,Ra^{1/4}}"
             r"{\left[1+(0{,}469/Pr)^{9/16}\right]^{4/9}}",
    ),
    "hplate_hot_up": dict(
        label="Placa horizontal, face quente p/ cima (L = A/P)",
        nu=lambda Ra, Pr: nu_horizontal_plate_hot_up(Ra),
        Lc_label="Lc = A/P",
        Ra_range=(1e4, 1e11),
        Pr_range=(0.7, float("inf")),
        corr=r"Nu = 0{,}54\,Ra^{1/4}\ (10^{4}\!\le\!Ra\!\le\!10^{7});\ \ "
             r"Nu = 0{,}15\,Ra^{1/3}\ (10^{7}\!\le\!Ra\!\le\!10^{11})",
    ),
    "hplate_hot_down": dict(
        label="Placa horizontal, face quente p/ baixo (L = A/P)",
        nu=lambda Ra, Pr: nu_horizontal_plate_hot_down(Ra),
        Lc_label="Lc = A/P",
        Ra_range=(1e5, 1e10),
        Pr_range=(0.7, float("inf")),
        corr=r"Nu = 0{,}27\,Ra^{1/4}\ (10^{5}\!\le\!Ra\!\le\!10^{10})",
    ),
}


def regime_vertical_plate(Ra: float) -> str:
    """Classifica o regime na placa vertical (transição em Ra ≈ 1e9)."""
    return "laminar" if Ra <= 1e9 else "turbulento"


def convection_summary(Ts: float, Tinf: float, Lc: float, props: FluidProps,
                       geometry: str) -> Dict:
    """
    Resumo de engenharia para uma geometria isotérmica:
      Ra, Gr, Nu (correlação), h, q'' e q'' por unidade de área.
    h = Nu·k/Lc ; q'' = h·|Ts-Tinf| [W/m²].
    """
    if geometry not in GEOMETRIES:
        raise ValueError(f"Geometria desconhecida: {geometry}")
    Ra = rayleigh(Ts, Tinf, Lc, props)
    Gr = grashof(Ts, Tinf, Lc, props)
    Nu = GEOMETRIES[geometry]["nu"](Ra, props.Pr)
    h = Nu * props.k / Lc
    qpp = h * abs(Ts - Tinf)
    return dict(Ra=Ra, Gr=Gr, Pr=props.Pr, Nu=Nu, h=h, qpp=qpp)


# ============================================================
# (4) Solução de similaridade — placa vertical isotérmica laminar
#     (Ostrach, 1953). Resolução numérica das equações de transporte.
# ============================================================
#
# Variável de similaridade:  eta = (Gr_x/4)^(1/4) · (y/x)
# Função corrente -> f(eta); temperatura adimensional theta = (T-Tinf)/(Ts-Tinf).
# Sistema acoplado (camada-limite, aproximação de Boussinesq):
#       f''' + 3 f f'' - 2 (f')² + theta = 0
#       theta'' + 3 Pr f theta' = 0
# Condições de contorno:
#       f(0)=0, f'(0)=0, theta(0)=1
#       f'(eta->inf)=0, theta(eta->inf)=0
# A velocidade local: u = (2 nu / x) · Gr_x^(1/2) · f'(eta).

def solve_similarity(Pr: float, eta_max: float | None = None,
                     n: int = 400) -> Dict[str, np.ndarray]:
    """
    Resolve o sistema de similaridade de Ostrach para Pr dado.

    Retorna dict com:
      eta, f, fp (=f'), fpp (=f''), theta, thetap (=theta'),
      minus_theta0p = -theta'(0)  (gradiente de temperatura na parede),
      fp_max, eta_fpmax (pico do perfil de velocidade).
    """
    if Pr <= 0:
        raise ValueError("Pr deve ser > 0.")

    # A camada de velocidade se estende mais para Pr baixo; a térmica encolhe
    # para Pr alto. Ajustamos o domínio para garantir convergência.
    if eta_max is None:
        eta_max = float(np.clip(8.0 / np.sqrt(min(Pr, 1.0)), 8.0, 30.0))

    eta = np.linspace(0.0, eta_max, max(n, 200))

    # Estado: y = [f, f', f'', theta, theta']
    def odes(x, y):
        f, fp, fpp, th, thp = y
        fppp = -3.0 * f * fpp + 2.0 * fp**2 - th
        thpp = -3.0 * Pr * f * thp
        return np.vstack([fp, fpp, fppp, thp, thpp])

    def bc(ya, yb):
        return np.array([ya[0], ya[1], ya[3] - 1.0, yb[1], yb[3]])

    # Palpite inicial fisicamente plausível
    Y = np.zeros((5, eta.size))
    Y[0] = 1.0 - np.exp(-eta)         # f
    Y[1] = eta * np.exp(-eta)         # f'  (sobe e volta a zero)
    Y[2] = (1.0 - eta) * np.exp(-eta) # f''
    Y[3] = np.exp(-eta)               # theta
    Y[4] = -np.exp(-eta)              # theta'

    sol = solve_bvp(odes, bc, eta, Y, max_nodes=40000, tol=1e-6)
    if not sol.success:
        raise RuntimeError(f"solve_bvp não convergiu (Pr={Pr}): {sol.message}")

    etas = np.linspace(0.0, eta_max, max(n, 200))
    ys = sol.sol(etas)
    fp = ys[1]
    i_peak = int(np.argmax(fp))

    return dict(
        eta=etas, f=ys[0], fp=fp, fpp=ys[2], theta=ys[3], thetap=ys[4],
        minus_theta0p=float(-ys[4, 0]),
        fp_max=float(fp[i_peak]), eta_fpmax=float(etas[i_peak]),
    )


def nu_local_similarity(Gr_x: float, minus_theta0p: float) -> float:
    """Nu_x local pela solução de similaridade: Nu_x = (Gr_x/4)^(1/4)·(-theta'(0))."""
    return (Gr_x / 4.0)**0.25 * minus_theta0p


def nu_avg_similarity(Gr_L: float, minus_theta0p: float) -> float:
    """
    Nu médio (0..L) laminar pela similaridade:
        Nu_L = (4/3)·(Gr_L/4)^(1/4)·(-theta'(0)) = (4/3)·Nu_{x=L}.
    """
    return (4.0 / 3.0) * nu_local_similarity(Gr_L, minus_theta0p)


def bl_thickness_vertical(x: np.ndarray, Ts: float, Tinf: float,
                          props: FluidProps) -> np.ndarray:
    """
    Estimativa integral da espessura da camada-limite na placa vertical
    laminar (Incropera):
        delta/x = 3.93 · Pr^(-1/2) · (0.952 + Pr)^(1/4) · Gr_x^(-1/4)
    Retorna delta(x) [m]. Em x=0 retorna 0.
    """
    x = np.asarray(x, dtype=float)
    delta = np.zeros_like(x)
    mask = x > 0
    Gr_x = grashof(Ts, Tinf, x[mask], props)
    Pr = props.Pr
    delta[mask] = x[mask] * 3.93 * Pr**(-0.5) * (0.952 + Pr)**0.25 * Gr_x**(-0.25)
    return delta


# ============================================================
# (5) Análise de escala (Bejan) — placa vertical isotérmica
# ============================================================
#
# Equações da camada-limite (Boussinesq), x ao longo da placa, y normal:
#   continuidade:  u_x + v_y = 0
#   momento (x):   u u_x + v u_y = nu u_yy + g beta (T - Tinf)
#   energia:       u T_x + v T_y = alpha T_yy
#
# Escalas: x ~ H ; y ~ delta_t ; u ~ U ; v ~ U delta_t / H (continuidade).
# Balanço de ENERGIA (convecção ~ condução na camada térmica):
#       U dT/H ~ alpha dT/delta_t²   =>   U ~ alpha H / delta_t²
# Com isso, a razão inércia/atrito no momento vale:
#       (U²/H) / (nu U/delta_t²) = U delta_t²/(nu H) = alpha/nu = 1/Pr.
# Logo o EMPUXO (g beta dT), que sempre move o fluido, equilibra:
#   * o ATRITO,  se Pr >> 1  (inércia desprezível)  -> delta_t/H ~ Ra^(-1/4),      Nu ~ Ra^(1/4)
#   * a INÉRCIA, se Pr << 1  (atrito desprezível)    -> delta_t/H ~ (Ra Pr)^(-1/4), Nu ~ (Ra Pr)^(1/4)
# A análise de escala fornece o EXPOENTE; o coeficiente O(1) vem da
# similaridade/correlação (ver reduced_nu_*).

def scale_momentum_balance(Pr: float) -> Dict:
    """
    Termos do balanço de momento por análise de escala, normalizados de modo que
    o EMPUXO (que equilibra o maior deles) valha 1. Como inércia/atrito ~ 1/Pr:
      * Pr >= 1: atrito ~ empuxo = 1 ; inércia ~ 1/Pr   (balanço atrito–empuxo)
      * Pr <  1: inércia ~ empuxo = 1 ; atrito ~ Pr     (balanço inércia–empuxo)
    Perto de Pr ~ 1 (aqui 1/3 < Pr < 3) os três termos são comparáveis: é uma
    região de transição, não um dos limites assintóticos.
    """
    if Pr <= 0:
        raise ValueError("Pr deve ser > 0.")
    transition = (1.0 / 3.0) < Pr < 3.0
    if Pr >= 1.0:
        inertia, friction = 1.0 / Pr, 1.0
        regime, driver = "alto_Pr", "atrito–empuxo"
    else:
        inertia, friction = 1.0, Pr
        regime, driver = "baixo_Pr", "inércia–empuxo"
    if transition:
        driver = "transição (Pr ~ 1)"
    return dict(inertia=inertia, friction=friction, buoyancy=1.0,
                inertia_over_friction=1.0 / Pr, regime=regime,
                driver=driver, transition=transition)


def scale_estimates_vertical(Ra: float, Pr: float) -> Dict:
    """
    Estimativas de escala (Bejan) para delta_t/H e Nu na placa vertical isotérmica,
    nos dois regimes (prefator 1 — apenas o expoente é fornecido pela análise):
      * alto  Pr (atrito–empuxo):  delta_t/H ~ Ra^(-1/4),      Nu ~ Ra^(1/4)
      * baixo Pr (inércia–empuxo): delta_t/H ~ (Ra Pr)^(-1/4), Nu ~ (Ra Pr)^(1/4)
    """
    if Ra <= 0 or Pr <= 0:
        raise ValueError("Ra e Pr devem ser > 0.")
    hi = dict(delta_t_over_H=Ra ** -0.25, Nu=Ra ** 0.25, law="Nu ~ Ra^(1/4)")
    lo = dict(delta_t_over_H=(Ra * Pr) ** -0.25, Nu=(Ra * Pr) ** 0.25,
              law="Nu ~ (Ra·Pr)^(1/4)")
    regime = "alto_Pr" if Pr >= 1.0 else "baixo_Pr"
    return dict(regime=regime, chosen=(hi if Pr >= 1.0 else lo),
                high_Pr=hi, low_Pr=lo)


def reduced_nu_similarity(Pr: float) -> float:
    """
    Nu reduzido = Nu_médio / Ra^(1/4) pela solução de similaridade (função só de
    Pr no regime laminar). Com Nu_médio = (4/3)(Gr/4)^(1/4)(−theta'(0)) e Gr=Ra/Pr:
        Nu/Ra^(1/4) = (4/3)·(1/(4 Pr))^(1/4)·(−theta'(0)).
    """
    sim = solve_similarity(Pr)
    return (4.0 / 3.0) * (1.0 / (4.0 * Pr)) ** 0.25 * sim["minus_theta0p"]


def reduced_nu_churchill(Pr: float) -> float:
    """
    Nu reduzido = Nu/Ra^(1/4) pela correlação laminar de Churchill–Chu (só a
    parte que escala com Ra^(1/4); função só de Pr):
        0.670 / [1 + (0.492/Pr)^(9/16)]^(4/9).
    Assíntotas: -> 0.670 para Pr->inf ; -> 0.800·Pr^(1/4) para Pr->0.
    """
    if Pr <= 0:
        raise ValueError("Pr deve ser > 0.")
    return 0.670 / (1.0 + (0.492 / Pr) ** (9.0 / 16.0)) ** (4.0 / 9.0)


def _sanity() -> str:
    return "natconv_iso OK"
