# zscan_thick_open.py
# =============================================================================
# Open aperture Z-scan — Thick medium
# Sheik-Bahae et al., Optical Engineering 30(8), 1228-1235 (1991)
#
# Physical model:
#   Open aperture (S=1) collects ALL transmitted light regardless of
#   beam distortion from nonlinear refraction. Therefore T_open is
#   insensitive to n2 (Kerr) and measures ONLY nonlinear absorption (beta).
#
#   For a thick sample, the beam intensity varies along the propagation
#   axis inside the medium. We integrate the 2PA loss at each slice:
#
#   dI/dz' = -beta * I^2   (two-photon absorption)
#
#   This is solved analytically slice by slice:
#   I_out = I_in / (1 + beta * I_in * dL)
#
#   The total transmitted power is then integrated over the beam profile
#   at the sample exit and normalised by the linear reference.
#
# Why open aperture matters for your experiment:
#   1. Verifies beta ~ 0 for CS2 at 800nm (below 2PA edge)
#   2. For other materials where beta != 0, you divide closed/open
#      to isolate the pure Kerr contribution to n2
#   3. Gives you the pure absorption signature independently
#
# Expected behaviour:
#   CS2 at 800nm: flat T(z) = 1.0 (no 2PA)
#   Material with beta > 0: symmetric dip at focus (z=0)
#   Dip depth increases with beta and I0
#
# Parameters match zscan_thick_closed.py exactly for direct comparison.
# =============================================================================

import numpy as np
import os

# =============================================================================
# Beam and experimental parameters — identical to zscan_thick_closed.py
# =============================================================================

LAM    = 800e-9       # wavelength (m)
W0     = 1e-6         # beam waist (m)
N0     = 1.63         # CS2 linear refractive index
L      = 100e-6       # CS2 sample thickness (m)
D_DET  = 0.04         # sample to detector distance (m)
S      = 0.25         # aperture transmittance (not used for open, kept for ref)

# Derived beam parameters
Z0     = np.pi * W0**2 / LAM
K      = 2 * np.pi / LAM

# Peak intensity — EOM reduced rep rate
# 1W avg / 80MHz * 100kHz EOM = 1.25mW after EOM
# E_pulse = 1.25e-3 / 100e3 = 12.5 pJ
# P_peak  = 12.5e-12 / 200e-15 = 62.5 W
# I0 = 2 * P_peak / (pi * W0^2)
P_PEAK = 62.5
I0     = 2 * P_PEAK / (np.pi * W0**2)


# =============================================================================
# Radial intensity profile at sample entrance
# =============================================================================

def beam_profile(r_grid, z_pos):
    """
    Radial intensity profile of Gaussian beam at position z_pos.

    Parameters
    ----------
    r_grid : radial grid (m)
    z_pos  : axial position relative to beam waist (m)

    Returns
    -------
    I_r : intensity profile (W/m^2), shape (N_r,)
    """
    wz  = W0 * np.sqrt(1.0 + (z_pos / Z0)**2)
    Iz  = I0 / (1.0 + (z_pos / Z0)**2)
    return Iz * np.exp(-2.0 * r_grid**2 / wz**2)


# =============================================================================
# 2PA propagation through thick sample — slice by slice
# =============================================================================

def propagate_2PA_thick(r_grid, z_s, beta_val, N_slices=200):
    """
    Propagate radial intensity profile through thick sample with 2PA.

    Parameters
    ----------
    r_grid   : radial grid (m), shape (N_r,)
    z_s      : sample centre position (m)
    beta_val : 2PA coefficient (m/W)
    N_slices : number of slices through sample

    Returns
    -------
    I_out : exit intensity profile (W/m^2), shape (N_r,)
    I_lin : linear reference exit profile (W/m^2), shape (N_r,)
    """
    dL    = L / N_slices
    z_ent = z_s - L / 2.0
    I     = beam_profile(r_grid, z_ent)

    for m in range(N_slices):
        if abs(beta_val) > 1e-30:
            I = I / (1.0 + beta_val * I * dL)

    I_lin = beam_profile(r_grid, z_ent)

    return I, I_lin


# =============================================================================
# Open aperture transmittance T(z, S=1)
# =============================================================================

def T_thick_open(z_arr, beta_val, N_r=300, N_slices=200):
    """
    Open aperture normalised transmittance T(z) for thick sample.

    Parameters
    ----------
    z_arr    : array of sample centre positions (m)
    beta_val : 2PA coefficient (m/W)
    N_r      : radial integration points
    N_slices : slices through sample

    Returns
    -------
    T : ndarray, normalised transmittance at each z
    """
    z_max  = np.max(np.abs(z_arr))
    r_max  = 5.0 * W0 * np.sqrt(1.0 + (z_max / Z0)**2)
    r_grid = np.linspace(0.0, r_max, N_r)

    T_out = np.empty(len(z_arr))

    for i, z_s in enumerate(z_arr):
        I_out, I_lin = propagate_2PA_thick(r_grid, z_s, beta_val, N_slices)

        P_out = 2.0 * np.pi * np.trapezoid(I_out * r_grid, r_grid)
        P_lin = 2.0 * np.pi * np.trapezoid(I_lin * r_grid, r_grid)

        T_out[i] = P_out / P_lin if P_lin > 1e-30 else 1.0

    return T_out


def q0_thick(beta_val):
    """On-axis 2PA parameter at focus: q0 = beta * I0 * L."""
    return beta_val * I0 * L


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

    print("=" * 60)
    print("CS2 Thick Medium Open Aperture Z-scan Parameters")
    print("=" * 60)
    print(f"  lambda  = {LAM*1e9:.0f} nm")
    print(f"  w0      = {W0*1e6:.1f} um")
    print(f"  z0      = {Z0*1e6:.2f} um")
    print(f"  L       = {L*1e6:.0f} um")
    print(f"  L / z0  = {L/Z0:.2f}  (thick medium)")
    print(f"  n0      = {N0}")
    print(f"  I0      = {I0:.3e} W/m^2")
    print(f"  P_peak  = {P_PEAK:.1f} W")
    print("=" * 60)

    z_range = 5.0 * max(Z0, L / 2.0)
    z_arr   = np.linspace(-z_range, z_range, 200)
    x_norm  = z_arr / Z0

    PARAM_STR = (f'lambda={LAM*1e9:.0f}nm  w0={W0*1e6:.1f}um  '
                 f'z0={Z0*1e6:.2f}um  L={L*1e6:.0f}um  L/z0={L/Z0:.1f}\n'
                 f'n0={N0}  I0={I0:.2e} W/m^2  S=1 (open)')

    SAVE_DIR = r'C:\PhD\Plots'

    def save_fig(fig, filename):
        os.makedirs(SAVE_DIR, exist_ok=True)
        filepath = os.path.join(SAVE_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f'Saved: {filepath}')

    # ------------------------------------------------------------------
    # Plot 1: CS2 at 800nm — beta ~ 0, expect flat T(z) = 1.0
    # ------------------------------------------------------------------
    BETA_CS2 = 0.0

    print('\nComputing open aperture T(z) for CS2 (beta=0)...')
    T_cs2 = T_thick_open(z_arr, BETA_CS2)
    print(f'  Max deviation from 1.0: {np.max(np.abs(T_cs2 - 1.0)):.2e}')
    print(f'  (Should be ~0 for beta=0)')

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x_norm, T_cs2, color='steelblue',
            label=f'CS2 beta=0   q0={q0_thick(BETA_CS2):.4f}')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.axvline(0.0, color='gray', linestyle=':', alpha=0.4)
    ax.set_ylim(0.95, 1.05)
    ax.set_xlabel('z / z0', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalised T(z)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Open aperture (S=1) — Thick medium, CS2 at 800nm\n'
        'beta=0: flat T(z)=1.0 confirms no 2PA response\n' + PARAM_STR,
        fontsize=11, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    save_fig(fig, 'thick_open_cs2_flat.png')
    plt.show()

    # ------------------------------------------------------------------
    # Plot 2: beta sweep
    # ------------------------------------------------------------------
    BETA_VALS_cmGW = [0.0, 0.5, 1.0, 2.0, 5.0]
    BETA_VALS_SI   = [b * 1e-11 for b in BETA_VALS_cmGW]
    colors         = plt.cm.plasma(np.linspace(0.1, 0.9, len(BETA_VALS_SI)))

    fig, ax = plt.subplots(figsize=(11, 7))
    print('\nComputing beta sweep...')

    for beta_v, col, b_cgw in zip(BETA_VALS_SI, colors, BETA_VALS_cmGW):
        T_v = T_thick_open(z_arr, beta_v)
        q0  = q0_thick(beta_v)
        lbl = (f'beta=0 (no 2PA)   q0=0' if b_cgw == 0.0
               else f'beta={b_cgw:.1f} cm/GW   q0={q0:.4f}')
        print(f'  beta={b_cgw:.1f} cm/GW   q0={q0:.4f}   '
              f'min T={np.min(T_v):.4f}')
        ax.plot(x_norm, T_v, color=col, label=lbl)

    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.axvline(0.0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('z / z0', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalised T(z)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Open aperture (S=1) — Thick medium, beta sweep\n'
        'Symmetric dip at focus — pure 2PA signature\n' + PARAM_STR,
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig(fig, 'thick_open_beta_sweep.png')
    plt.show()

    # ------------------------------------------------------------------
    # Plot 3: Open vs closed for a material WITH both n2 and beta
    # ------------------------------------------------------------------
    from zscan_thick_closed import T_thick_closed, alpha_correction

    BETA_DEMO = 1.0e-11
    N2_DEMO   = 3e-19

    beam_params_demo = {
        'w0':     W0,
        'z0':     Z0,
        'lam':    LAM,
        'n0':     N0,
        'L':      L,
        'I0':     I0,
        'a_corr': alpha_correction(S),
        'd_det':  D_DET,
    }

    w_det_lin = W0 * np.sqrt(1.0 + (D_DET / Z0)**2)
    r_a_demo  = w_det_lin * np.sqrt(-np.log(1.0 - S) / 2.0)

    print('\nComputing open vs closed for demo material (n2 + beta)...')
    T_open_demo   = T_thick_open(z_arr, BETA_DEMO)
    T_closed_demo = T_thick_closed(z_arr, N2_DEMO, beam_params_demo, r_a_demo)
    T_ratio       = T_closed_demo / T_open_demo

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(x_norm, T_closed_demo, color='red',
            label='Closed aperture (n2 + beta)')
    ax.plot(x_norm, T_open_demo, color='green',
            label=f'Open aperture (beta only, {BETA_DEMO*1e11:.1f} cm/GW)')
    ax.plot(x_norm, T_ratio, color='purple', linestyle='--',
            label='Closed / Open (pure Kerr)')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.axvline(0.0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('z / z0', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalised T(z)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Open vs Closed aperture — Why both scans are needed\n'
        'Closed/Open ratio isolates pure Kerr (n2) contribution\n' + PARAM_STR,
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig(fig, 'thick_open_vs_closed.png')
    plt.show()