# zscan_closed_SA.py
# =============================================================================
# Closed aperture Z-scan — Saturable Absorption only, no Kerr effect
#
# Physical model for SA (Notes, Eq. 4):
#   alpha(I) = alpha_0 / (1 + I / I_sat)
#
# Full propagation through sample (Notes, Eq. 1):
#   dI/dz' = -alpha(I) * I = -[alpha_0 / (1 + I/I_sat)] * I
#
# Integrated numerically via RK4 — no approximations applied.
#
# Exit field construction:
#   The SA modifies only the beam amplitude, not the phase (no Kerr here).
#   The radially varying amplitude transmission is recovered from the
#   propagated intensity profile:
#
#   t_SA(r) = sqrt(I_out(r) / I_in(r))           (amplitude transmission)
#
#   The exit field at the sample plane is then (Sheik-Bahae Eq. 7, adapted):
#   E_exit(r) = E_in(r) * t_SA(r)                (pure amplitude modification)
#
#   This is analogous to Eq. (7) from Sheik-Bahae but with the nonlinear
#   phase term exp(iΔΦ) = 1 (no Kerr) and exp(-αL/2) replaced by the
#   full radially varying SA amplitude transmission t_SA(r).
#
# Free-space propagation to aperture (Sheik-Bahae Eqs. 8-11):
#   The SA-modified exit field is decomposed into Gaussian modes via the
#   GD framework. Because t_SA(r) is not a pure Gaussian, higher-order
#   modes carry the non-Gaussian amplitude distortion from bleaching.
#   Each mode propagates independently to the aperture using Eq. (9),
#   and the aperture power is computed via Eq. (10).
#
# This correctly captures the z-dependent asymmetry of the closed aperture
# scan — the beam geometry is converging on one side of focus and diverging
# on the other, so the aperture clips different amounts of power even for
# a symmetric amplitude modification at the sample.
#
# Expected behaviour:
#   Pure SA enhances the pre-focal peak and reduces the post-focal valley
#   compared to pure Kerr — opposite effect to 2PA (Sheik-Bahae Section V).
#   The curve remains asymmetric because the aperture breaks the z-symmetry.
# =============================================================================

import math
import numpy as np
from beam import GaussianBeam
from config import S

# =============================================================================
# SA parameters — consistent with zscan_open_SA.py
#
# ALPHA_0 * L = 0.50  (moderate linear absorption, non-zero linear reference)
# I0 / I_SAT  = 1.00  (at saturation threshold, moderate bleaching)
# =============================================================================
ALPHA_0 = 185.0    # linear absorption coefficient (m^-1)
I_SAT   = 2.1e12   # saturation intensity (W/m^2)


# =============================================================================
# RK4 propagation — full SA model
#
# Solves (Notes, Eq. 1):
#   dI/dz' = -[alpha_0 / (1 + I/I_sat)] * I
#
# No approximation — full nonlinear Beer-Lambert propagation
# =============================================================================

def propagate_SA_intensity(I_in, alpha_0, I_sat, L, N_steps=200):
    """
    Propagate intensity through sample under full SA model via RK4.

    Implements Notes Eq. (1) with Notes Eq. (4):
        dI/dz' = -[alpha_0 / (1 + I/I_sat)] * I

    Parameters
    ----------
    I_in    : input intensity profile (W/m^2), shape (N_r,)
    alpha_0 : linear absorption coefficient (m^-1)
    I_sat   : saturation intensity (W/m^2)
    L       : sample thickness (m)
    N_steps : RK4 integration steps through sample

    Returns
    -------
    I_out : transmitted intensity profile (W/m^2), shape (N_r,)
    """
    dz = L / N_steps

    def dIdz(I):
        return -(alpha_0 / (1.0 + I / I_sat)) * I

    I = np.array(I_in, dtype=float)

    for _ in range(N_steps):
        k1 = dIdz(I)
        k2 = dIdz(I + 0.5 * dz * k1)
        k3 = dIdz(I + 0.5 * dz * k2)
        k4 = dIdz(I + dz * k3)
        I  = I + (dz / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return I


# =============================================================================
# SA amplitude transmission profile
#
# Recovers the radially varying field amplitude transmission from the
# propagated intensity:
#
#   t_SA(r) = sqrt(I_out(r) / I_in(r))
#
# This is the amplitude analogue of the phase term exp(iΔΦ) in
# Sheik-Bahae Eq. (7). For pure SA with no Kerr, the exit field is:
#
#   E_exit(r) = E_in(r) * t_SA(r)      (Sheik-Bahae Eq. 7, adapted)
#
# t_SA(r) varies with r because bleaching is stronger at r=0 (peak
# irradiance) and weaker in the beam wings — this non-Gaussian profile
# is what drives higher-order mode content in the GD expansion.
# =============================================================================

def sa_amplitude_transmission(r_grid, z, beam, alpha_0, I_sat, N_steps=200):
    """
    Compute radially varying SA amplitude transmission t_SA(r).

    t_SA(r) = sqrt(I_out(r) / I_in(r))

    Parameters
    ----------
    r_grid  : radial grid at sample plane (m), shape (N_r,)
    z       : sample position (m)
    beam    : GaussianBeam instance
    alpha_0 : linear absorption coefficient (m^-1)
    I_sat   : saturation intensity (W/m^2)
    N_steps : RK4 steps

    Returns
    -------
    t_SA : real amplitude transmission profile, shape (N_r,)
    """
    wz  = beam._wz(z)
    Iz  = beam._Iz(z)

    # Radial intensity at sample entrance — Gaussian beam profile
    I_in = Iz * np.exp(-2.0 * r_grid**2 / wz**2)

    # Propagate through sample via full SA model
    I_out = propagate_SA_intensity(I_in, alpha_0, I_sat, beam.L, N_steps)

    # Amplitude transmission — avoid division by zero in beam wings
    t_SA = np.where(I_in > 1e-30, np.sqrt(I_out / I_in), 0.0)

    return t_SA


# =============================================================================
# GD mode projection of SA-modified exit field
#
# The SA-modified exit field at the sample plane is:
#   E_exit(r) = E_in(r) * t_SA(r)
#
# where E_in(r) = E0 * (w0/w(z)) * exp(-r^2/w(z)^2)  [Sheik-Bahae Eq. 2]
#
# Because t_SA(r) is not purely Gaussian, E_exit(r) cannot be described
# by a single Gaussian mode. We project it onto the GD basis by computing
# the overlap integral of E_exit(r) with each Gaussian mode at the sample
# plane, then propagate each mode to the aperture using Sheik-Bahae Eq. (9).
#
# Mode m has beam waist at sample plane:
#   w_m0^2 = w(z)^2 / (2m+1)           [Sheik-Bahae Eq. 9 parameters]
#
# The overlap coefficient for mode m is:
#   c_m = integral[ E_exit(r) * phi_m(r) * r dr ]
#       / integral[ |phi_m(r)|^2 * r dr ]
#
# where phi_m(r) = exp(-r^2 / w_m0^2) is the mth basis Gaussian.
#
# Each mode is then propagated to the aperture via Sheik-Bahae Eq. (9).
# =============================================================================

def aperture_field_SA_GD(z, beam, alpha_0, I_sat, r_a, n_terms=35, N_steps=200):
    """
    Reconstruct E at aperture plane for pure SA via GD mode projection.

    Projects SA-modified exit field onto Gaussian mode basis, then
    propagates each mode to aperture using Sheik-Bahae Eq. (9).

    Parameters
    ----------
    z       : sample position (m)
    beam    : GaussianBeam instance
    alpha_0 : linear absorption coefficient (m^-1)
    I_sat   : saturation intensity (W/m^2)
    r_a     : aperture radius (m)
    n_terms : number of GD modes
    N_steps : RK4 steps through sample

    Returns
    -------
    E_ap   : complex field at aperture plane, shape (N_r_ap,)
    r_ap   : radial grid at aperture (m), shape (N_r_ap,)
    """
    d   = beam.d_det
    k   = beam.k
    lam = beam.lam
    w0  = beam.w0
    z0  = beam.z0

    w2z = w0**2 * (1.0 + (z / z0)**2)
    Rz  = np.inf if abs(z) < 1e-20 else z * (1.0 + (z0 / z)**2)
    g   = 1.0 + d / Rz if np.isfinite(Rz) else 1.0

    # Fine radial grid at sample plane for accurate mode projection
    r_sample = np.linspace(0.0, 5.0 * beam._wz(z), 600)

    # SA amplitude transmission at sample plane
    t_SA = sa_amplitude_transmission(r_sample, z, beam, alpha_0, I_sat, N_steps)

    # Incident Gaussian field amplitude at sample entrance (proportional)
    wz  = beam._wz(z)
    E_in_profile = np.exp(-r_sample**2 / wz**2)   # normalised amplitude

    # SA-modified exit field (real — no phase from Kerr)
    E_exit = E_in_profile * t_SA

    # Output grid at aperture plane
    r_ap = np.linspace(0.0, r_a, 400)
    E_ap = np.zeros(len(r_ap), dtype=complex)

    for m in range(n_terms + 1):

        # Mode m waist at sample plane (Sheik-Bahae Eq. 9 parameters)
        w_m0_sq = w2z / (2*m + 1)
        w_m0    = np.sqrt(w_m0_sq)

        # Gaussian basis function for mode m at sample plane
        phi_m = np.exp(-r_sample**2 / w_m0_sq)

        # Overlap coefficient — project E_exit onto mode m
        # c_m = <E_exit, phi_m> / <phi_m, phi_m>
        num   = np.trapezoid(E_exit * phi_m * r_sample, r_sample)
        denom = np.trapezoid(phi_m**2 * r_sample, r_sample)
        c_m   = num / denom if abs(denom) > 1e-30 else 0.0

        if abs(c_m) < 1e-30:
            continue

        # Propagate mode m to aperture — Sheik-Bahae Eq. (9)
        d_m       = np.pi * w_m0_sq / lam
        denom_m   = g**2 + (d / d_m)**2
        w_m_sq    = w_m0_sq * denom_m
        one_minus = 1.0 - g / denom_m
        R_m       = d / one_minus if abs(one_minus) > 1e-20 else np.inf
        theta_m   = np.arctan((d / d_m) / g)
        amp_m     = np.sqrt(w_m0_sq / w_m_sq)

        curv    = 1j * k * r_ap**2 / (2.0 * R_m) if np.isfinite(R_m) else 0.0
        gauss_m = np.exp(-r_ap**2 / w_m_sq + curv)

        E_ap += c_m * amp_m * gauss_m * np.exp(1j * theta_m)

    return E_ap, r_ap


def integrate_aperture_power(E_ap, r_grid):
    """Sheik-Bahae Eq. (10): numerical integration of |E_ap|^2 * r dr."""
    return np.trapezoid(np.abs(E_ap)**2 * r_grid, r_grid)


# =============================================================================
# Closed aperture transmittance — pure SA, no Kerr
# Sheik-Bahae Eq. (11): T(z) = P_NL / P_lin
# =============================================================================

def T_closed_SA(beam, alpha_0, I_sat, S, n_terms=35, N_steps=200):
    """
    Closed aperture normalised transmittance T(z) for pure SA bleaching.

    No Kerr effect — amplitude modification only.
    Uses full SA RK4 propagation + GD mode projection + Eq. (10)/(11).

    Parameters
    ----------
    beam    : GaussianBeam instance
    alpha_0 : linear absorption coefficient (m^-1)
    I_sat   : saturation intensity (W/m^2)
    S       : aperture linear transmittance
    n_terms : number of GD modes
    N_steps : RK4 steps through sample

    Returns
    -------
    T : ndarray, shape (n_z,)
    """
    r_a = beam.aperture_radius(S)
    out = np.empty(len(beam.z_arr))

    for i, z in enumerate(beam.z_arr):

        # SA-modified field at aperture — nonlinear amplitude, zero phase
        E_NL, r_ap = aperture_field_SA_GD(
            z, beam, alpha_0, I_sat, r_a, n_terms, N_steps)

        # Linear reference — uniform amplitude transmission exp(-alpha_0*L/2)
        # At I -> 0, SA reduces to linear: t_SA -> exp(-alpha_0*L/2)
        # Use same GD propagation with flat amplitude profile
        wz           = beam._wz(z)
        r_sample     = np.linspace(0.0, 5.0 * wz, 600)
        t_lin        = np.exp(-alpha_0 * beam.L / 2.0)
        E_exit_lin   = np.exp(-r_sample**2 / wz**2) * t_lin

        # Project linear exit field onto GD modes
        w2z = beam.w0**2 * (1.0 + (z / beam.z0)**2)
        Rz  = np.inf if abs(z) < 1e-20 else z * (1.0 + (beam.z0 / z)**2)
        g   = 1.0 + beam.d_det / Rz if np.isfinite(Rz) else 1.0

        E_lin_ap = np.zeros(len(r_ap), dtype=complex)

        for m in range(n_terms + 1):
            w_m0_sq = w2z / (2*m + 1)
            w_m0    = np.sqrt(w_m0_sq)
            phi_m   = np.exp(-r_sample**2 / w_m0_sq)

            num   = np.trapezoid(E_exit_lin * phi_m * r_sample, r_sample)
            denom = np.trapezoid(phi_m**2   * r_sample, r_sample)
            c_m   = num / denom if abs(denom) > 1e-30 else 0.0

            if abs(c_m) < 1e-30:
                continue

            d_m       = np.pi * w_m0_sq / beam.lam
            denom_m   = g**2 + (beam.d_det / d_m)**2
            w_m_sq    = w_m0_sq * denom_m
            one_minus = 1.0 - g / denom_m
            R_m       = beam.d_det / one_minus if abs(one_minus) > 1e-20 else np.inf
            theta_m   = np.arctan((beam.d_det / d_m) / g)
            amp_m     = np.sqrt(w_m0_sq / w_m_sq)

            curv    = 1j * beam.k * r_ap**2 / (2.0 * R_m) if np.isfinite(R_m) else 0.0
            gauss_m = np.exp(-r_ap**2 / w_m_sq + curv)

            E_lin_ap += c_m * amp_m * gauss_m * np.exp(1j * theta_m)

        P_NL  = integrate_aperture_power(E_NL,     r_ap)
        P_lin = integrate_aperture_power(E_lin_ap, r_ap)

        out[i] = P_NL / P_lin if P_lin > 1e-30 else 1.0

    return out


# =============================================================================
# Run standalone
# =============================================================================

if __name__ == '__main__':

    import matplotlib
    import matplotlib.pyplot as plt
    import os

    matplotlib.rcParams.update({
        'font.family':       'DejaVu Sans',
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         True,
        'grid.alpha':        0.25,
        'grid.linestyle':    '--',
        'lines.linewidth':   2.2,
        'figure.facecolor':  'white',
        'axes.facecolor':    'white',
        'legend.framealpha': 0.9,
        'legend.edgecolor':  '#cccccc',
    })

    SAVE_DIR = r'C:\PhD\Plots'

    def save_fig(fig, filename):
        os.makedirs(SAVE_DIR, exist_ok=True)
        filepath = os.path.join(SAVE_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f'Saved: {filepath}')

    from zscan_closed_GD import T_closed_GD
    from config import BETA, GAMMA

    beam = GaussianBeam()
    beam.summary()

    print(f'\nI0 / I_sat  = {beam.I0 / I_SAT:.3f}')
    print(f'alpha_0 * L = {ALPHA_0 * beam.L:.4f}')

    PARAM_STR = (f'lambda={beam.lam*1e9:.0f}nm  w0={beam.w0*1e6:.0f}um  '
                 f'z0={beam.z0*1e3:.2f}mm  L={beam.L*1e3:.1f}mm  '
                 f'I0={beam.I0:.2e} W/m2  S={S}\n'
                 f'alpha0={ALPHA_0:.0f} m^-1  '
                 f'I_sat={I_SAT:.2e} W/m2  '
                 f'I0/I_sat={beam.I0/I_SAT:.2f}')

    # ------------------------------------------------------------------
    # Plot 1: pure SA closed aperture — single curve
    # Expected: asymmetric, peak enhanced relative to pure Kerr,
    # valley reduced — opposite effect to 2PA
    # ------------------------------------------------------------------
    print('\nComputing closed aperture SA (pure bleaching)...')
    T_SA   = T_closed_SA(beam, ALPHA_0, I_SAT, S)
    Tpv_SA = np.max(T_SA) - np.min(T_SA)
    print(f'  Peak T  = {np.max(T_SA):.4f}')
    print(f'  Valley T = {np.min(T_SA):.4f}')
    print(f'  Tpv     = {Tpv_SA:.4f}')

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(beam.x_norm, T_SA, color='steelblue',
            label=f'Pure SA (no Kerr)   Tpv={Tpv_SA:.4f}')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.axvline(0.0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('z / z0', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalised T(z)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Closed aperture — Pure SA bleaching, no Kerr\n'
        'GD mode projection + Sheik-Bahae Eqs. (9)-(11)\n'
        + PARAM_STR,
        fontsize=12, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    save_fig(fig, 'closed_SA_pure.png')
    plt.show()

    # ------------------------------------------------------------------
    # Plot 2: pure Kerr (beta=0) from existing GD script
    # ------------------------------------------------------------------
    print('Computing closed aperture pure Kerr (beta=0)...')
    T_Kerr   = T_closed_GD(beam, 0.0, GAMMA, S)
    Tpv_Kerr = np.max(T_Kerr) - np.min(T_Kerr)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(beam.x_norm, T_Kerr, color='red',
            label=f'Pure Kerr (no SA)   Tpv={Tpv_Kerr:.4f}')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.axvline(0.0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('z / z0', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalised T(z)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Closed aperture — Pure Kerr (GD, beta=0)\n'
        'Sheik-Bahae Eqs. (8)-(11)\n'
        + PARAM_STR,
        fontsize=12, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    save_fig(fig, 'closed_Kerr_pure.png')
    plt.show()

    # ------------------------------------------------------------------
    # Plot 3: SA vs Kerr vs Kerr+2PA comparison
    # ------------------------------------------------------------------
    print('Computing GD Kerr + 2PA reference...')
    T_full   = T_closed_GD(beam, BETA, GAMMA, S)
    Tpv_full = np.max(T_full) - np.min(T_full)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(beam.x_norm, T_SA,   color='steelblue',
            label=f'Pure SA (no Kerr)         Tpv={Tpv_SA:.4f}')
    ax.plot(beam.x_norm, T_Kerr, color='red',
            label=f'Pure Kerr (no SA)         Tpv={Tpv_Kerr:.4f}')
    ax.plot(beam.x_norm, T_full, color='green', linestyle='--',
            label=f'GD: Kerr + 2PA (beta=5.8)  Tpv={Tpv_full:.4f}')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.axvline(0.0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('z / z0', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalised T(z)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Closed aperture — SA vs Kerr vs Kerr+2PA\n'
        + PARAM_STR,
        fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig(fig, 'closed_SA_vs_Kerr_comparison.png')
    plt.show()