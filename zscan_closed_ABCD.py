# zscan_closed_ABCD.py
# =============================================================================
# Closed aperture Z-scan — ABCD thin-lens propagation + Eq. (10) power integral
# Single Gaussian beam tracked via complex q parameter through:
#   free-space (focus to sample) -> thin lens (NL interaction) -> free-space (to detector)
# Compare against zscan_closed_GD.py to see where single-Gaussian assumption breaks down
# =============================================================================

import numpy as np
from beam import GaussianBeam
from config import BETA, GAMMA, S


def abcd_q_to_wR(q, lam):
    """
    Extract beam radius w and wavefront radius R from complex q parameter.
    1/q = 1/R - i*lam/(pi*w^2)
    """
    q_inv = 1.0 / q
    w_sq  = -lam / (np.pi * np.imag(q_inv))
    R     = 1.0 / np.real(q_inv) if abs(np.real(q_inv)) > 1e-30 else np.inf
    return np.sqrt(w_sq), R


def aperture_field_ABCD(z, dPhi, beam, r_a):
    """
    Propagate beam through ABCD system and return single-Gaussian E at aperture.

    Propagation sequence:
      1. Construct incoming q at sample plane from free-space Gaussian beam
      2. Apply thin lens with f = k*w(z)^2 / (4*dPhi)
         (from parabolic approximation of nonlinear phase exp(i*dPhi*exp(-2r^2/w^2)))
      3. Free-space propagation to detector at d = D_DET

    Parameters
    ----------
    z    : sample position (m)
    dPhi : on-axis nonlinear phase shift at z (rad), signed
    beam : GaussianBeam instance
    r_a  : aperture radius (m)

    Returns
    -------
    E_ap   : complex Gaussian field at aperture, shape (N_r,)
    r_grid : radial grid (m), shape (N_r,)
    """
    d   = beam.d_det
    k   = beam.k
    lam = beam.lam
    w0  = beam.w0
    z0  = beam.z0

    w2z = w0**2 * (1.0 + (z / z0)**2)
    Rz  = np.inf if abs(z) < 1e-20 else z * (1.0 + (z0 / z)**2)

    # --- Step 1: incoming q at sample plane ---
    if np.isfinite(Rz):
        q_in = 1.0 / (1.0/Rz - 1j * lam / (np.pi * w2z))
    else:
        q_in = 1j * np.pi * w2z / lam   # at focus: pure imaginary = i*z0

    # --- Step 2: thin lens ---
    # f = k*w(z)^2 / (4*dPhi)
    # dPhi=0 means no interaction, pass through unchanged
    if abs(dPhi) < 1e-30:
        q_out = q_in
    else:
        f     = k * w2z / (4.0 * dPhi)
        q_out = q_in / (1.0 - q_in / f)

    # --- Step 3: free-space to detector ---
    q_det = q_out + d

    w_det, R_det = abcd_q_to_wR(q_det, lam)

    # --- Build single Gaussian E field on r_grid ---
    r_grid = np.linspace(0.0, r_a, 400)
    curv   = 1j * k * r_grid**2 / (2.0 * R_det) if np.isfinite(R_det) else 0.0
    E_ap   = np.exp(-r_grid**2 / w_det**2 + curv)

    return E_ap, r_grid


def integrate_aperture_power(E_ap, r_grid):
    """Eq. (10): numerical integration of |E_ap|^2 * r dr."""
    return np.trapezoid(np.abs(E_ap)**2 * r_grid, r_grid)


def T_closed_ABCD(beam, gamma, S):
    """
    Closed aperture normalised transmittance T(z) via ABCD thin-lens + Eq. (10).

    Note: no beta dependence here — the ABCD model tracks only the refractive
    lensing effect. Absorption modifies the beam amplitude but not its Gaussian
    shape, so the normalisation P_NL/P_lin cancels it out for a single Gaussian.

    Parameters
    ----------
    beam  : GaussianBeam instance
    gamma : nonlinear refractive index (m^2/W), negative = self-defocusing
    S     : aperture linear transmittance

    Returns
    -------
    T : ndarray, shape (n_z,)
    """
    r_a = beam.aperture_radius(S)
    k   = beam.k
    out = np.empty(len(beam.z_arr))

    for i, z in enumerate(beam.z_arr):
        dPhi = k * gamma * beam._Iz(z) * beam.Leff

        E_NL,  r = aperture_field_ABCD(z, dPhi, beam, r_a)
        E_lin, _ = aperture_field_ABCD(z, 0.0,  beam, r_a)

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

    from zscan_closed_GD import T_closed_GD

    beam = GaussianBeam()
    beam.summary()

    PARAM_STR = (f'lambda={beam.lam*1e9:.0f}nm  w0={beam.w0*1e6:.0f}um  '
                 f'z0={beam.z0*1e3:.2f}mm  L={beam.L*1e3:.1f}mm  '
                 f'I0={beam.I0:.2e} W/m2  '
                 f'gamma={GAMMA:.1e} m2/W  S={S}')
    SAVE_DIR = r'C:\PhD\Plots'

    def save_fig(fig, filename):
        import os
        os.makedirs(SAVE_DIR, exist_ok=True)
        filepath = os.path.join(SAVE_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f'Saved: {filepath}')

    # ------------------------------------------------------------------
    # Plot 1: ABCD only
    # ------------------------------------------------------------------
    print('Computing ABCD...')
    T_ABCD = T_closed_ABCD(beam, GAMMA, S)
    Tpv_ABCD = np.max(T_ABCD) - np.min(T_ABCD)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(beam.x_norm, T_ABCD, color='steelblue',
            label=f'ABCD thin-lens   Tpv={Tpv_ABCD:.4f}')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.axvline(0.0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('z / z0', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalised T(z)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Closed aperture — ABCD thin-lens + Eq. (10) power integral\n' + PARAM_STR,
        fontsize=12, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    save_fig(fig, 'closed_abcd_single.png')
    plt.show()

    # ------------------------------------------------------------------
    # Plot 2: GD vs ABCD comparison + residual
    # ------------------------------------------------------------------
    print('Computing GD...')
    T_GD = T_closed_GD(beam, 0.0, GAMMA, S)
    Tpv_GD  = np.max(T_GD)  - np.min(T_GD)
    residual = T_GD - T_ABCD

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].plot(beam.x_norm, T_GD,   color='red',
                 label=f'Full GD       Tpv={Tpv_GD:.4f}')
    axes[0].plot(beam.x_norm, T_ABCD, color='steelblue', linestyle='--',
                 label=f'ABCD thin-lens  Tpv={Tpv_ABCD:.4f}')
    axes[0].axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    axes[0].axvline(0.0, color='gray', linestyle=':', alpha=0.4)
    axes[0].set_xlabel('z / z0', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Normalised T(z)', fontsize=14, fontweight='bold')
    axes[0].set_title('GD vs ABCD thin-lens\n(same Eq.(10) power integral)',
                      fontsize=13, fontweight='bold')
    axes[0].legend()

    axes[1].plot(beam.x_norm, residual, color='purple',
                 label=f'GD - ABCD   max|diff|={np.max(np.abs(residual)):.4f}')
    axes[1].axhline(0.0, color='gray', linestyle=':', alpha=0.6)
    axes[1].set_xlabel('z / z0', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('T_GD - T_ABCD', fontsize=14, fontweight='bold')
    axes[1].set_title('Residual\n(zero = methods agree perfectly)',
                      fontsize=13, fontweight='bold')
    axes[1].legend()

    plt.suptitle('Full GD vs ABCD Thin-Lens — Same Aperture Power Integral\n' + PARAM_STR,
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'closed_gd_vs_abcd_comparison.png')
    plt.show()