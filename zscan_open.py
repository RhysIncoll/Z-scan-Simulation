# zscan_open.py
# =============================================================================
# Open aperture Z-scan transmittance — Eqs. (30)/(31) Sheik-Bahae 1990
# Imports beam definition from beam.py and parameters from config.py
# =============================================================================

import numpy as np
from beam import GaussianBeam
from config import BETA, S


def T_open(beam, beta, N_tau=400):
    """
    Open aperture normalised transmittance T(z, S=1) via Eq. (30).

    Symmetric dip at focus for two-photon absorption (beta > 0).
    Valid for |q0| < 1 where q0 = beta * I(z) * Leff.

    Parameters
    ----------
    beam  : GaussianBeam instance
    beta  : two-photon absorption coefficient (m/W)
    N_tau : number of integration points for the Gaussian pulse integral

    Returns
    -------
    T : ndarray, shape (n_z,), normalised transmittance at each z
    """
    tau = np.linspace(-6, 6, N_tau)
    out = np.empty(len(beam.z_arr))

    for i, z in enumerate(beam.z_arr):
        q = beta * beam._Iz(z) * beam.Leff
        if abs(q) < 1e-14:
            out[i] = 1.0
        else:
            out[i] = np.trapezoid(
                np.log(1.0 + q * np.exp(-tau**2)), tau
            ) / (np.sqrt(np.pi) * q)

    return out


def q0_on_axis(beam, beta):
    """
    On-axis q0 parameter at focus: q0 = beta * I0 * Leff.
    Must be < 1 for Eq. (31) series to converge.
    """
    return beta * beam.I0 * beam.Leff


# =============================================================================
# Run standalone — produces two plots:
#   1. Single beta sanity check (symmetric dip)
#   2. Beta sweep showing deepening dip with increasing absorption
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
                 f'I0={beam.I0:.2e} W/m2')
    SAVE_DIR = r'C:\PhD\Plots'

    def save_fig(fig, filename):
        import os
        os.makedirs(SAVE_DIR, exist_ok=True)
        filepath = os.path.join(SAVE_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f'Saved: {filepath}')
    # ------------------------------------------------------------------
    # Plot 1: single beta sanity check
    # ------------------------------------------------------------------
    T_ref = T_open(beam, BETA)
    q0    = q0_on_axis(beam, BETA)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(beam.x_norm, T_ref, color='steelblue',
            label=f'beta={BETA*1e11:.1f} cm/GW   q0(0)={q0:.3f}')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.axvline(0.0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('z / z0', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalised T(z)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Open aperture (S=1) — Eq. (30) sanity check\n'
        'Symmetric dip at focus confirms 2PA response\n' + PARAM_STR,
        fontsize=12, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    save_fig(fig, 'open_aperture_single_beta.png')
    plt.show()

    # ------------------------------------------------------------------
    # Plot 2: beta sweep
    # Cap at q0 < 1 to stay within convergence of Eq. (31)
    # q0 = beta * I0 * Leff < 1  =>  beta < 1/(I0*Leff)
    # ------------------------------------------------------------------
    BETA_VALS_cmGW = [0.0, 1.0, 2.9, 5.8, 8.7, 11.6]
    BETA_VALS_SI   = [b * 1e-11 for b in BETA_VALS_cmGW]
    colors         = plt.cm.plasma(np.linspace(0.1, 0.9, len(BETA_VALS_SI)))

    fig, ax = plt.subplots(figsize=(11, 7))

    for beta, col, b_cgw in zip(BETA_VALS_SI, colors, BETA_VALS_cmGW):
        T   = T_open(beam, beta)
        q0  = q0_on_axis(beam, beta)
        lbl = ('beta=0 (linear)' if b_cgw == 0.0
               else f'beta={b_cgw:.1f} cm/GW   q0={q0:.3f}')
        ax.plot(beam.x_norm, T, color=col, label=lbl)

    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.axvline(0.0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('z / z0', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalised T(z)', fontsize=14, fontweight='bold')
    ax.set_title(
        'Open aperture (S=1) — varying beta\n'
        'Symmetric 2PA dip, deeper with increasing beta\n' + PARAM_STR,
        fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig(fig, 'open_aperture_beta_sweep.png')
    plt.show()