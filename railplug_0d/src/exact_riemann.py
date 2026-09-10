"""Exact solution of the 1D planar Euler Riemann problem (Toro, "Riemann
Solvers and Numerical Methods for Fluid Dynamics", 3rd ed., ch. 4).

Used only as an independent ground truth to validate `radial_hydro.py`'s
HLLC finite-volume scheme against the classic Sod shock-tube problem --
it is not used by the discharge-channel simulation itself.
"""
from __future__ import annotations

import numpy as np


def _pressure_function(p: float, rho_k: float, p_k: float, gamma: float) -> tuple[float, float]:
    """f_K(p) and its derivative, for one side (K = L or R) of the Riemann problem."""
    c_k = np.sqrt(gamma * p_k / rho_k)
    if p > p_k:  # shock
        A_k = 2.0 / ((gamma + 1.0) * rho_k)
        B_k = (gamma - 1.0) / (gamma + 1.0) * p_k
        f = (p - p_k) * np.sqrt(A_k / (p + B_k))
        fd = np.sqrt(A_k / (p + B_k)) * (1.0 - 0.5 * (p - p_k) / (p + B_k))
    else:  # rarefaction
        f = (2.0 * c_k / (gamma - 1.0)) * ((p / p_k) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        fd = (1.0 / (rho_k * c_k)) * (p / p_k) ** (-(gamma + 1.0) / (2.0 * gamma))
    return f, fd


def _solve_star_pressure(rhoL, uL, pL, rhoR, uR, pR, gamma, tol=1e-8, max_iter=100) -> float:
    p_old = 0.5 * (pL + pR)
    p_old = max(p_old, 1e-8)
    for _ in range(max_iter):
        fL, fdL = _pressure_function(p_old, rhoL, pL, gamma)
        fR, fdR = _pressure_function(p_old, rhoR, pR, gamma)
        f = fL + fR + (uR - uL)
        fd = fdL + fdR
        p_new = p_old - f / fd
        if p_new < 0.0:
            p_new = tol
        if abs(p_new - p_old) / (0.5 * (p_new + p_old)) < tol:
            p_old = p_new
            break
        p_old = p_new
    return p_old


def sod_exact_solution(
    x: np.ndarray, t: float,
    rhoL: float = 1.0, uL: float = 0.0, pL: float = 1.0,
    rhoR: float = 0.125, uR: float = 0.0, pR: float = 0.1,
    x0: float = 0.5, gamma: float = 1.4,
):
    """Exact (rho, u, p) at positions `x`, time `t`, for a Riemann problem
    with diaphragm initially at x0. Defaults are the standard Sod test."""
    cL = np.sqrt(gamma * pL / rhoL)
    cR = np.sqrt(gamma * pR / rhoR)

    p_star = _solve_star_pressure(rhoL, uL, pL, rhoR, uR, pR, gamma)
    fL, _ = _pressure_function(p_star, rhoL, pL, gamma)
    fR, _ = _pressure_function(p_star, rhoR, pR, gamma)
    u_star = 0.5 * (uL + uR) + 0.5 * (fR - fL)

    rho = np.empty_like(x)
    u = np.empty_like(x)
    p = np.empty_like(x)

    S = (x - x0) / max(t, 1e-30)

    for i, s in enumerate(S):
        if s <= u_star:
            # Left of the contact: left wave (shock or rarefaction), then star-left state.
            if p_star > pL:  # left shock
                q = np.sqrt(((gamma + 1.0) / (2.0 * gamma)) * (p_star / pL) + (gamma - 1.0) / (2.0 * gamma))
                S_shock = uL - cL * q
                if s < S_shock:
                    rho[i], u[i], p[i] = rhoL, uL, pL
                else:
                    rho_star_L = rhoL * ((p_star / pL) + (gamma - 1.0) / (gamma + 1.0)) / (
                        (gamma - 1.0) / (gamma + 1.0) * (p_star / pL) + 1.0
                    )
                    rho[i], u[i], p[i] = rho_star_L, u_star, p_star
            else:  # left rarefaction
                c_star_L = cL * (p_star / pL) ** ((gamma - 1.0) / (2.0 * gamma))
                S_head = uL - cL
                S_tail = u_star - c_star_L
                if s < S_head:
                    rho[i], u[i], p[i] = rhoL, uL, pL
                elif s > S_tail:
                    rho_star_L = rhoL * (p_star / pL) ** (1.0 / gamma)
                    rho[i], u[i], p[i] = rho_star_L, u_star, p_star
                else:  # inside the fan
                    u_fan = (2.0 / (gamma + 1.0)) * (cL + (gamma - 1.0) / 2.0 * uL + s)
                    c_fan = (2.0 / (gamma + 1.0)) * (cL + (gamma - 1.0) / 2.0 * (uL - s))
                    rho_fan = rhoL * (c_fan / cL) ** (2.0 / (gamma - 1.0))
                    p_fan = pL * (c_fan / cL) ** (2.0 * gamma / (gamma - 1.0))
                    rho[i], u[i], p[i] = rho_fan, u_fan, p_fan
        else:
            # Right of the contact: star-right state, then right wave.
            if p_star > pR:  # right shock
                q = np.sqrt(((gamma + 1.0) / (2.0 * gamma)) * (p_star / pR) + (gamma - 1.0) / (2.0 * gamma))
                S_shock = uR + cR * q
                if s > S_shock:
                    rho[i], u[i], p[i] = rhoR, uR, pR
                else:
                    rho_star_R = rhoR * ((p_star / pR) + (gamma - 1.0) / (gamma + 1.0)) / (
                        (gamma - 1.0) / (gamma + 1.0) * (p_star / pR) + 1.0
                    )
                    rho[i], u[i], p[i] = rho_star_R, u_star, p_star
            else:  # right rarefaction
                c_star_R = cR * (p_star / pR) ** ((gamma - 1.0) / (2.0 * gamma))
                S_head = uR + cR
                S_tail = u_star + c_star_R
                if s > S_head:
                    rho[i], u[i], p[i] = rhoR, uR, pR
                elif s < S_tail:
                    rho_star_R = rhoR * (p_star / pR) ** (1.0 / gamma)
                    rho[i], u[i], p[i] = rho_star_R, u_star, p_star
                else:  # inside the fan
                    u_fan = (2.0 / (gamma + 1.0)) * (-cR + (gamma - 1.0) / 2.0 * uR + s)
                    c_fan = (2.0 / (gamma + 1.0)) * (cR - (gamma - 1.0) / 2.0 * (uR - s))
                    rho_fan = rhoR * (c_fan / cR) ** (2.0 / (gamma - 1.0))
                    p_fan = pR * (c_fan / cR) ** (2.0 * gamma / (gamma - 1.0))
                    rho[i], u[i], p[i] = rho_fan, u_fan, p_fan

    return rho, u, p
