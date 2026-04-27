# zscan_closed_GD.py
# =============================================================================
# Closed aperture Z-scan — full Gaussian decomposition, Eqs. (8)-(11)
# Sheik-Bahae et al. 1990
# =============================================================================

import math
import numpy as np
from beam import GaussianBeam
from config import BETA, GAMMA, S


def coupling_product(m, beta, gamma, k):
    """
    Eq. (28) coupling factor accounting for simultaneous refractive
    and absorptive nonlinearity. Returns nan if series overflows.
    """
    if m == 0:
        return 1.0
    eta  = beta / (2.0 * k * gamma) if gamma > 1e-30 else 0.0
    prod = 1.0 + 0j
    for n in range(1, m + 1):
        prod *= (1.0 + 1j * (2*n - 1) * eta)
        if not np.isfinite(abs(prod)):
            return np.nan
    return prod


def aperture_field_GD(z, dPhi, beam, beta, gamma, r_a, n_terms=35):
    """
    Reconstruct E at aperture plane via GD mode sum, Eq. (9).

    Parameters
    ----------
    z      : sample position (m)
    dPhi   : on-axis nonlinear phase shift at z (rad)
    beam   : GaussianBeam instance
    beta   : two-photon absorption coefficient (m/W)
    gamma  : nonlinear refractive index (m^2/W)
    r_a    : aperture radius (m)
    n_terms: number of GD terms

    Returns
    -------
    E_ap   : complex field at aperture plane, shape (N_r,)
    r_grid : radial grid (m), shape (N_r,)
    """
    d   = beam.d_det
    k   = beam.k
    lam = beam.lam
    w0  = beam.w0
    z0  = beam.z0

    w2z = w0**2 * (1.0 + (z / z0)**2)
    Rz  = np.inf if abs(z) < 1e-20 else z * (1.0 + (z0 / z)**2)
    g   = 1.0 + d / Rz if np.isfinite(Rz) else 1.0

    r_grid = np.linspace(0.0, r_a, 400)
    E_ap   = np.zeros(len(r_grid), dtype=complex)

    for m in range(n_terms + 1):
        cp = coupling_product(m, beta, gamma, k)
        if not np.isfinite(abs(cp)):
            break

        fm        = ((1j * dPhi)**m / float(math.factorial(m))) * cp
        w_m0_sq   = w2z / (2*m + 1)
        d_m       = np.pi * w_m0_sq / lam
        denom_m   = g**2 + (d / d_m)**2
        w_m_sq    = w_m0_sq * denom_m
        one_minus = 1.0 - g / denom_m
        R_m       = d / one_minus if abs(one_minus) > 1e-20 else np.inf
        theta_m   = np.arctan((d / d_m) / g)
        curv      = 1j * k * r_grid**2 / (2.0 * R_m) if np.isfinite(R_m) else 0.0
        gauss_m   = np.exp(-r_grid**2 / w_m_sq + curv)
        amp_m     = np.sqrt(w_m0_sq / w_m_sq)
        E_ap     += fm * amp_m * gauss_m * np.exp(1j * theta_m)

    return E_ap, r_grid


def integrate_aperture_power(E_ap, r_grid):
    """Eq. (10): numerical integration of |E_ap|^2 * r dr."""
    return np.trapezoid(np.abs(E_ap)**2 * r_grid, r_grid)


def T_closed_GD(beam, beta, gamma, S, n_terms=35):
    """
    Closed aperture normalised transmittance T(z) via full GD, Eq. (11).

    Parameters
    ----------
    beam   : GaussianBeam instance
    beta   : two-photon absorption coefficient (m/W)
    gamma  : nonlinear refractive index (m^2/W)
    S      : aperture linear transmittance
    n_terms: GD series terms

    Returns
    -------
    T : ndarray, shape (n_z,)
    """
    r_a = beam.aperture_radius(S)
    k   = beam.k
    out = np.empty(len(beam.z_arr))

    for i, z in enumerate(beam.z_arr):
        dPhi = k * gamma * beam._Iz(z) * beam.Leff

        E_NL,  r = aperture_field_GD(z, dPhi, beam, beta, gamma, r_a, n_terms)
        E_lin, _ = aperture_field_GD(z, 0.0,  beam, beta, gamma, r_a, n_terms)

        P_NL  = integrate_aperture_power(E_NL,  r)
        P_lin = integrate_aperture_power(E_lin, r)
        out[i] = P_NL / P_lin if P_lin > 1e-30 else 1.0

    return out


# =============================================================================
# Run standalone
# =============================================================================

if __name__ == '__main__':

    import matplotlib
    import matplotlib.pyplot as plt

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

    beam = GaussianBeam()
    beam.summary()

    PARAM_STR = (f'lambda={beam.lam*1e9:.0f}nm  w0={beam.w0*1e6:.0f}um  '
                 f'z0={beam.z0*1e3:.2f}mm  L={beam.L*1e3:.1f}mm  '
                 f'I0={beam.I0:.2e} W/m2  '
                 f'beta={BETA*1e11:.1f} cm/GW  '
                 f'gamma={GAMMA:.1e} m2/W  S={S}')
    
    SAVE_DIR = r'C:\PhD\Plots'

    def save_fig(fig, filename):
        import os
        os.makedirs(SAVE_DIR, exist_ok=True)
        filepath = os.path.join(SAVE_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f'Saved: {filepath}')

    # ------------------------------------------------------------------
    # Plot 1: reference parameters, single curve
    # ------------------------------------------------------------------
    print('Computing GD at reference parameters...')
    T_ref = T_closed_GD(beam, BETA, GAMMA, S)
    Tpv   = np.max(T_ref) - np.min(T_ref)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(beam.x_norm, T_ref, color='red',
            label=f'Full GD   Tpv={Tpv:.4f}')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.axvline(0.0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('z / z0', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalised T(z)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Closed aperture — Full Gaussian decomposition (Eqs. 8–11)\n' + PARAM_STR,
        fontsize=12, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    save_fig(fig, 'closed_gd_single.png')
    plt.show()

    # ------------------------------------------------------------------
    # Plot 2: beta sweep
    # ------------------------------------------------------------------
    BETA_VALS_cmGW = [0.0, 1.0, 2.9, 5.8, 8.7, 11.6]
    BETA_VALS_SI   = [b * 1e-11 for b in BETA_VALS_cmGW]
    colors         = plt.cm.plasma(np.linspace(0.1, 0.9, len(BETA_VALS_SI)))

    fig, ax = plt.subplots(figsize=(11, 7))

    for beta, col, b_cgw in zip(BETA_VALS_SI, colors, BETA_VALS_cmGW):
        print(f'  beta={b_cgw:.1f} cm/GW...')
        T   = T_closed_GD(beam, beta, GAMMA, S)
        Tpv = np.max(T) - np.min(T)
        lbl = (f'beta=0 (pure Kerr)   Tpv={Tpv:.4f}' if b_cgw == 0.0
               else f'beta={b_cgw:.1f} cm/GW   Tpv={Tpv:.4f}')
        ax.plot(beam.x_norm, T, color=col, label=lbl)

    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.axvline(0.0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('z / z0', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalised T(z)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Closed aperture (S=0.4) — GD, varying beta\n'
        '2PA suppresses peak and enhances valley with increasing beta\n' + PARAM_STR,
        fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig(fig, 'closed_gd_beta_sweep.png')
    plt.show()