# zscan_open_SA.py
# =============================================================================
# Open aperture Z-scan transmittance for Saturable Absorption (SA)
# Uses the FULL saturation model — no approximations
#
# Physical model for SA absorption coefficient:
#
#   alpha(I) = alpha_0 / (1 + I / I_sat)
#
# where:
#   alpha_0  = linear (unsaturated) absorption coefficient (m^-1)
#   I_sat    = saturation intensity — irradiance at which absorption
#              drops to half its linear value (W/m^2)
#
# This is NOT the beta_eff approximation from Eq. (9) of the review.
# The full propagation equation is integrated numerically through the
# sample thickness at each radial and z position:
#
#   dI/dz' = -alpha(I) * I = -[alpha_0 / (1 + I/I_sat)] * I
#
# The transmitted intensity at the sample exit is then integrated
# over the beam profile to give the total transmitted power, which
# is normalised by the linear (low irradiance) transmitted power
# to give T(z, S=1).
#
# Expected behaviour:
#   SA (alpha_0 > 0, I_sat finite):
#       Transmittance peak at focus (z=0) — sample bleaches at high I
#       Curve is symmetric about z=0
#       Peak increases as I0/I_sat increases (stronger saturation)
# =============================================================================

import numpy as np
from beam import GaussianBeam
from config import I0, S

# =============================================================================
# SA parameters
# Chosen so that I0/I_sat is in the range 0.1-5 for well-behaved curves
# and alpha_0 * L is in the range 0.1-1.0 so linear reference is non-zero
#
# With I0 = 0.21e13 W/m^2 and L = 2.7mm from config:
#   ALPHA_0 = 185 m^-1  =>  alpha_0 * L = 0.5   (moderate linear absorption)
#   I_SAT   = 2.1e12    =>  I0 / I_sat  = 1.0   (at saturation threshold)
# =============================================================================
ALPHA_0 = 185.0    # linear absorption coefficient (m^-1)
I_SAT   = 2.1e12   # saturation intensity (W/m^2)


# =============================================================================
# Propagation through sample — full SA model via RK4
#
# Solves: dI/dz' = -[alpha_0 / (1 + I/I_sat)] * I
#
# This is the full Beer-Lambert propagation with intensity-dependent
# absorption — no linearisation or approximation applied
# =============================================================================

def propagate_SA(I_in, alpha_0, I_sat, L, N_steps=200):
    """
    Propagate intensity through sample under full SA model via RK4.

    dI/dz' = -[alpha_0 / (1 + I/I_sat)] * I

    Parameters
    ----------
    I_in    : input intensity at sample entrance (W/m^2)
    alpha_0 : linear absorption coefficient (m^-1)
    I_sat   : saturation intensity (W/m^2)
    L       : sample thickness (m)
    N_steps : number of RK4 integration steps

    Returns
    -------
    I_out : transmitted intensity at sample exit
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


def propagate_linear(I_in, alpha_0, L):
    """Linear reference: I_out = I_in * exp(-alpha_0 * L)"""
    return I_in * np.exp(-alpha_0 * L)


# =============================================================================
# Open aperture transmittance T(z, S=1) for SA
# =============================================================================

def T_open_SA(beam, alpha_0, I_sat, N_r=300, N_steps=200):
    """
    Open aperture normalised transmittance T(z, S=1) for saturable absorption.

    Uses full SA model — dI/dz' = -[alpha_0 / (1 + I/I_sat)] * I
    No approximations applied.

    Parameters
    ----------
    beam    : GaussianBeam instance
    alpha_0 : linear absorption coefficient (m^-1)
    I_sat   : saturation intensity (W/m^2)
    N_r     : radial integration points
    N_steps : RK4 steps through sample

    Returns
    -------
    T : ndarray, shape (n_z,)
    """
    r_max  = 3.0 * beam._wz(beam.z_arr[-1])
    r_grid = np.linspace(0.0, r_max, N_r)

    out = np.empty(len(beam.z_arr))

    for i, z in enumerate(beam.z_arr):
        wz  = beam._wz(z)
        I_r = beam._Iz(z) * np.exp(-2.0 * r_grid**2 / wz**2)

        I_out_SA  = propagate_SA(I_r, alpha_0, I_sat, beam.L, N_steps)
        I_out_lin = propagate_linear(I_r, alpha_0, beam.L)

        P_SA  = 2.0 * np.pi * np.trapezoid(I_out_SA  * r_grid, r_grid)
        P_lin = 2.0 * np.pi * np.trapezoid(I_out_lin * r_grid, r_grid)

        out[i] = P_SA / P_lin if P_lin > 1e-30 else 1.0

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

    beam = GaussianBeam()
    beam.summary()

    # Print key ratios so we know what regime we're in
    print(f'\nI0          = {beam.I0:.3e} W/m^2')
    print(f'I_sat       = {I_SAT:.3e} W/m^2')
    print(f'I0 / I_sat  = {beam.I0 / I_SAT:.3f}')
    print(f'alpha_0 * L = {ALPHA_0 * beam.L:.4f}')

    PARAM_STR = (f'lambda={beam.lam*1e9:.0f}nm  w0={beam.w0*1e6:.0f}um  '
                 f'z0={beam.z0*1e3:.2f}mm  L={beam.L*1e3:.1f}mm  '
                 f'I0={beam.I0:.2e} W/m2  '
                 f'alpha0={ALPHA_0:.0f} m^-1  '
                 f'I_sat={I_SAT:.2e} W/m2  '
                 f'I0/I_sat={beam.I0/I_SAT:.2f}')

    # ------------------------------------------------------------------
    # Plot 1: single curve sanity check
    # ------------------------------------------------------------------
    print('\nComputing SA transmittance at reference parameters...')
    T_SA = T_open_SA(beam, ALPHA_0, I_SAT)
    print(f'  Peak T = {np.max(T_SA):.4f}  at z/z0 = {beam.x_norm[np.argmax(T_SA)]:.2f}')

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(beam.x_norm, T_SA, color='steelblue',
            label=f'alpha_0={ALPHA_0:.0f} m^-1   I_sat={I_SAT:.2e} W/m2'
                  f'   I0/I_sat={beam.I0/I_SAT:.2f}')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.axvline(0.0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('z / z0', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalised T(z)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Open aperture (S=1) — Saturable Absorption, full SA model\n'
        'Symmetric peak at focus — sample bleaches at high irradiance\n'
        + PARAM_STR,
        fontsize=12, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    save_fig(fig, 'open_SA_single.png')
    plt.show()

    # ------------------------------------------------------------------
    # Plot 2: I_sat sweep
    # I0/I_sat ranges from 0.1 (weak) to 5.0 (strong saturation)
    # ------------------------------------------------------------------
    ISAT_VALS = [
        beam.I0 * 10,    # I0/I_sat = 0.1  weak saturation
        beam.I0 * 2,     # I0/I_sat = 0.5
        beam.I0 * 1,     # I0/I_sat = 1.0  at threshold
        beam.I0 / 2,     # I0/I_sat = 2.0
        beam.I0 / 5,     # I0/I_sat = 5.0  strong saturation
    ]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(ISAT_VALS)))

    fig, ax = plt.subplots(figsize=(11, 7))

    for I_s, col in zip(ISAT_VALS, colors):
        print(f'  I_sat={I_s:.2e}  I0/I_sat={beam.I0/I_s:.2f}')
        T = T_open_SA(beam, ALPHA_0, I_s)
        ax.plot(beam.x_norm, T, color=col,
                label=f'I_sat={I_s:.2e} W/m2   I0/I_sat={beam.I0/I_s:.1f}')

    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.axvline(0.0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('z / z0', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalised T(z)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Open aperture (S=1) — SA, varying saturation intensity\n'
        'Higher I0/I_sat = stronger bleaching at focus\n'
        + PARAM_STR,
        fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig(fig, 'open_SA_Isat_sweep.png')
    plt.show()

    # ------------------------------------------------------------------
    # Plot 3: alpha_0 sweep
    # Keep alpha_0 * L between 0.1 and 1.0
    # ------------------------------------------------------------------
    ALPHA_VALS = [
        37.0,    # a0*L = 0.10
        100.0,   # a0*L = 0.27
        185.0,   # a0*L = 0.50
        370.0,   # a0*L = 1.00
        740.0,   # a0*L = 2.00
    ]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ALPHA_VALS)))

    fig, ax = plt.subplots(figsize=(11, 7))

    for a0, col in zip(ALPHA_VALS, colors):
        print(f'  alpha_0={a0:.0f} m^-1  a0*L={a0*beam.L:.3f}')
        T = T_open_SA(beam, a0, I_SAT)
        ax.plot(beam.x_norm, T, color=col,
                label=f'alpha_0={a0:.0f} m^-1   a0*L={a0*beam.L:.2f}')

    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.axvline(0.0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('z / z0', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalised T(z)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Open aperture (S=1) — SA, varying linear absorption coefficient\n'
        'Stronger linear absorption = larger bleaching contrast at focus\n'
        + PARAM_STR,
        fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig(fig, 'open_SA_alpha_sweep.png')
    plt.show()