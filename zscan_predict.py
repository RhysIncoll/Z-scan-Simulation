# zscan_predict.py
# =============================================================================
# Z-scan forward simulator — pre-lab prediction and literature resolvability
#
# MODE 1 — Material prediction
#   Pick a material, get the T(z) curve your setup would produce in the lab.
#
# MODE 2 — Literature resolvability
#   For a material, simulate Tpv using YOUR setup parameters for each reported
#   n2 in the literature. If Tpv > TPV_MIN the signal is detectable.
#   DPhi0 = 1 rad is a model validity guide only, not a detectability limit.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from zscan_thick_closed import T_thick_closed, alpha_correction

# =============================================================================
# Run control
# =============================================================================

MODE         = 2
MATERIAL_KEY = 'graphene'

# =============================================================================
# Fixed laser setup parameters
# =============================================================================

LAM    = 800e-9
W0     = 1e-6
L_SAMP = 100e-6
S      = 0.25
P_PEAK = 62.5
D_DET  = 0.04

Z0 = np.pi * W0**2 / LAM
K  = 2 * np.pi / LAM
I0 = 2 * P_PEAK / (np.pi * W0**2)

w_det_lin = W0 * np.sqrt(1.0 + (D_DET / Z0)**2)
r_a       = w_det_lin * np.sqrt(-np.log(1.0 - S) / 2.0)
A_CORR    = alpha_correction(S)

TPV_MIN  = 0.01
SAVE_DIR = r'C:\PhD\Plots'

SETUP_STR = (f'lambda={LAM*1e9:.0f} nm  w0={W0*1e6:.1f} um  '
             f'z0={Z0*1e6:.2f} um  I0={I0:.2e} W/m2  '
             f'P_peak={P_PEAK:.1f} W  S={S}  L={L_SAMP*1e6:.0f} um')

# =============================================================================
# Material database — Mode 1
# =============================================================================

MATERIALS = {
    'CS2': {
        'n2':    3.20e-18,
        'n0':    1.63,
        'notes': 'Reference standard.',
    },
    'silicon_nitride': {
        'n2':    2.40e-19,
        'n0':    2.00,
        'notes': 'Si3N4 waveguide material.',
    },
    'graphene': {
        'n2':   -5.01e-17,
        'n0':    1.45,
        'notes': 'FLG suspension, Gonzalez 2024, 800 nm.',
    },
    'MoS2': {
        'n2':   -8.00e-18,
        'n0':    4.20,
        'notes': 'Monolayer MoS2.',
    },
}

# =============================================================================
# Literature database — Mode 2
# =============================================================================

LITERATURE = {
    'graphene': [
        {
            'label':  'Gonzalez 2024\nFLG suspension',
            'n2':     -5.01e-17,
            'n0':      1.45,
            'color':  'darkorange',
            'marker': 's',
        },
        {
            'label':  'Wang 2014\nGraphene',
            'n2':     -2.34e-16,
            'n0':      1.45,
            'color':  'mediumseagreen',
            'marker': '^',
        },
        {
            'label':  'CS2\nReference',
            'n2':      3.20e-18,
            'n0':      1.63,
            'color':  'black',
            'marker': '*',
        },
    ],
    'CS2': [
        {
            'label':  'CS2\nReference',
            'n2':      3.00e-19,
            'n0':      1.63,
            'color':  'black',
            'marker': '*',
        },
    ],
}

# =============================================================================
# Helpers
# =============================================================================

def make_beam_params(n0):
    return {
        'w0':     W0,
        'z0':     Z0,
        'lam':    LAM,
        'n0':     n0,
        'L':      L_SAMP,
        'I0':     I0,
        'a_corr': A_CORR,
        'd_det':  D_DET,
    }

def z_array(n_pts=300):
    z_range = 5.0 * max(Z0, L_SAMP / 2.0)
    return np.linspace(-z_range, z_range, n_pts)

def save_fig(fig, filename):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')

# =============================================================================
# MODE 1
# =============================================================================

def run_mode1(key):
    mat   = MATERIALS[key]
    n2    = mat['n2']
    n0    = mat['n0']
    bp    = make_beam_params(n0)
    z     = z_array()
    xnorm = z / Z0

    T     = T_thick_closed(z, n2, bp, r_a)
    Tpv   = np.max(T) - np.min(T)
    DPhi0 = K * abs(n2) * I0 * L_SAMP

    print(f"\nMode 1 — {key}")
    print(f"  n2={n2:.3e}  n0={n0}  DPhi0={DPhi0:.4f} rad  Tpv={Tpv:.5f}")
    print(f"  Detectable: {'YES' if Tpv > TPV_MIN else 'NO'}")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(xnorm, T, color='steelblue', linewidth=2.5,
            label=f'Simulated T(z)  Tpv={Tpv:.4f}')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0.0, color='gray', linestyle=':', alpha=0.4)
    ax.fill_between(xnorm,
                    1.0 - TPV_MIN / 2, 1.0 + TPV_MIN / 2,
                    color='crimson', alpha=0.10,
                    label=f'Noise floor band  Tpv_min={TPV_MIN}')
    ax.set_xlabel('z / z0', fontsize=13, fontweight='bold')
    ax.set_ylabel('Normalised T(z)', fontsize=13, fontweight='bold')
    ax.set_title(
        f'Predicted Z-scan trace — {key}\n'
        f'n2={n2:.3e} m2/W   DPhi0={DPhi0:.3f} rad   Tpv={Tpv:.4f}\n'
        + SETUP_STR,
        fontsize=10, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig(fig, f'zscan_predict_{key}.png')
    plt.show()

# =============================================================================
# MODE 2
# =============================================================================

def run_mode2(key):
    entries = LITERATURE[key]
    z_full  = z_array(n_pts=300)
    z_sweep = z_array(n_pts=40)
    xnorm   = z_full / Z0

    # Sensitivity curve — 120 points, 100 slices (fast)
    n0_rep    = entries[0]['n0']
    bp_rep    = make_beam_params(n0_rep)
    n2_sweep  = np.logspace(-23, -12, 60)
    tpv_sweep = np.empty(len(n2_sweep))

    print(f'\nMode 2 — {key}')
    print('  Computing sensitivity curve...', end=' ', flush=True)
    for i, n2_v in enumerate(n2_sweep):
        T_v          = T_thick_closed(z_sweep, n2_v, bp_rep, r_a, N_slices=60)
        tpv_sweep[i] = np.max(T_v) - np.min(T_v)
    print('done.')

    # Per-paper simulation — full resolution
    print(f"\n  {'Paper':<28} {'n2':>14}  {'DPhi0':>8}  {'Tpv':>8}  {'Detectable':>10}")
    print('  ' + '-' * 75)

    for p in entries:
        bp    = make_beam_params(p['n0'])
        T     = T_thick_closed(z_full, p['n2'], bp, r_a, N_slices=100)
        Tpv   = np.max(T) - np.min(T)
        DPhi0 = K * abs(p['n2']) * I0 * L_SAMP

        p['T']           = T
        p['Tpv']         = Tpv
        p['DPhi0']       = DPhi0
        p['detectable']  = Tpv >= TPV_MIN
        p['model_valid'] = DPhi0 <= 1.0

        lbl = p['label'].replace('\n', ' ')
        print(f"  {lbl:<28} {p['n2']:>14.3e}  {DPhi0:>8.2f}  "
              f"{Tpv:>8.5f}  {'YES' if p['detectable'] else 'NO':>10}")
        
    # ==========================================================================
    # Figure
    # ==========================================================================

    plt.rcParams.update({
        'font.family':       'DejaVu Sans',
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         True,
        'grid.alpha':        0.2,
        'grid.linestyle':    '--',
        'figure.facecolor':  'white',
        'axes.facecolor':    'white',
        'legend.framealpha': 0.95,
        'legend.edgecolor':  '#cccccc',
    })

    fig = plt.figure(figsize=(16, 7))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1], wspace=0.32)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ------------------------------------------------------------------
    # Left panel — sensitivity curve
    # ------------------------------------------------------------------

    ax1.loglog(n2_sweep, tpv_sweep,
               color='steelblue', linewidth=2.2, zorder=2,
               label='Sensitivity curve')

    n2_dphi1 = 1.0 / (K * I0 * L_SAMP)
    ax1.axvline(n2_dphi1, color='gray', linestyle=':', linewidth=1.4,
                label=f'DPhi0 = 1 rad  n2 = {n2_dphi1:.1e} m2/W')
    ax1.axhline(TPV_MIN, color='crimson', linestyle='--', linewidth=1.4,
                label=f'Noise floor  Tpv = {TPV_MIN}')
    ax1.axhspan(TPV_MIN, 20.0, color='limegreen', alpha=0.07, zorder=0,
                label='Detectable region')

    # Scatter — name as annotation on plot only, not in legend
    for p in entries:
        n2_abs = abs(p['n2'])
        edge   = 'black' if p['detectable'] else 'none'
        sign   = '+' if p['n2'] > 0 else '-'
        ax1.scatter(n2_abs, p['Tpv'],
                    color=p['color'], marker=p['marker'],
                    s=180, zorder=5, edgecolors=edge, linewidths=1.5,
                    label=f"n2={sign}{n2_abs:.1e}  Tpv={p['Tpv']:.4f}")
        ax1.annotate(p['label'],
                     xy=(n2_abs, p['Tpv']),
                     xytext=(7, 4), textcoords='offset points',
                     fontsize=8, color=p['color'], fontweight='bold')

    ax1.legend(fontsize=8, loc='upper left', framealpha=0.95,
               markerscale=0.5, handlelength=1.5)
    ax1.set_xlim(n2_sweep[0] * 0.5, n2_sweep[-1] * 2)
    ax1.set_ylim(1e-6, 10.0)
    ax1.set_xlabel('|n2|  (m2/W)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Tpv', fontsize=12, fontweight='bold')
    ax1.set_title(f'Sensitivity curve — {key}\n'
                  f'Black edge = detectable  |  Green = above noise floor',
                  fontsize=10, fontweight='bold')

    # ------------------------------------------------------------------
    # Right panel — T(z) all entries; dashed if below noise floor
    # ------------------------------------------------------------------

    # Right panel — T(z) ALL entries; dashed if below noise floor
    for p in entries:
        sign = '+' if p['n2'] > 0 else '-'
        name = p['label'].replace('\n', ' ')
        det  = '  below noise' if not p['detectable'] else ''
        lbl  = f"{name}  n2={sign}{abs(p['n2']):.1e}  Tpv={p['Tpv']:.4f}{det}"

        ls = '-'  if p['detectable'] else '--'
        lw = 2.2  if p['detectable'] else 1.5
        ax2.plot(xnorm, p['T'],
                 color=p['color'], linewidth=lw, linestyle=ls, label=lbl)

    ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(0.0, color='gray', linestyle=':', alpha=0.4)
    ax2.set_xlabel('z / z0', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Normalised T(z)', fontsize=12, fontweight='bold')
    ax2.set_title('Predicted T(z) — all entries\n'
                  'Solid = detectable  |  Dashed = below noise floor',
                  fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right', framealpha=0.95)

    fig.suptitle(
        f'Z-scan resolvability — {key} — literature vs our setup\n' + SETUP_STR,
        fontsize=10, fontweight='bold', y=1.01)

    plt.tight_layout()
    save_fig(fig, f'zscan_resolvability_{key}.png')
    plt.show()

# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    if MODE == 1:
        run_mode1(MATERIAL_KEY)
    elif MODE == 2:
        run_mode2(MATERIAL_KEY)
    else:
        raise ValueError(f"MODE must be 1 or 2, got {MODE}.")