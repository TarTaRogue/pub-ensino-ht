# pubensino/ht/view_factors_mc.py
"""Estimativa Monte Carlo de fatores de forma entre retângulos paralelos."""

from __future__ import annotations
import numpy as np

def sample_rect(n: int, Lx: float, Ly: float, cx: float=0.0, cy: float=0.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = (rng.random(n) - 0.5) * Lx + cx
    y = (rng.random(n) - 0.5) * Ly + cy
    return x, y

def view_factor_parallel_rectangles_mc(
    n: int, Lx1: float, Ly1: float, Lx2: float, Ly2: float, H: float,
    dx: float = 0.0, dy: float = 0.0, seed: int | None = 0
) -> dict:
    if n <= 0: raise ValueError("n deve ser positivo.")
    for val, name in [(Lx1,'Lx1'),(Ly1,'Ly1'),(Lx2,'Lx2'),(Ly2,'Ly2')]:
        if val <= 0: raise ValueError(f"{name} deve ser positivo.")
    if H <= 0: raise ValueError("H deve ser positivo.")

    rng = np.random.default_rng(seed)
    x1, y1 = sample_rect(n, Lx1, Ly1, 0.0, 0.0, rng)
    x2, y2 = sample_rect(n, Lx2, Ly2, dx,  dy,  rng)

    Rx = x2 - x1; Ry = y2 - y1
    R2 = Rx*Rx + Ry*Ry + H*H
    integrand = (H*H) / (np.pi * (R2**2))  # H^2 / (pi R^4)

    A2 = Lx2 * Ly2
    F12 = A2 * float(np.mean(integrand))
    sigma_mean = float(np.std(integrand, ddof=1) / np.sqrt(n)) if n > 1 else float('nan')
    return {"F12": F12, "sigma_F12": A2*sigma_mean, "A1": Lx1*Ly1, "A2": A2}
