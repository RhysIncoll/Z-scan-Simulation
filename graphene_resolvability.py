# graphene_resolvability.py
# =============================================================================
# Resolvability of graphene n2 values from literature using our Z-scan setup.
#
# Core question:
#   Given our fixed laser parameters, which literature n2 values fall within
#   our measurable window — i.e. above our noise floor (Tpv_min) and within
#   the perturbative regime (DPhi0 <= 1 rad)?
#
# Approach:
#   1. Compute our setup's Tpv as a function of |n2| — the sensitivity curve.
#      This uses the thick-medium distributed lens model from zscan_thick_closed.
#   2. Mark the detection window:
#        Lower bound: Tpv_min = 0.05 (high noise floor assumption)
#        Upper bound: |n2| where DPhi0 = k * n2 * I0 * L = 1 rad
#   3. Plot each literature entry on the sensitivity curve.
#      Entries inside the window are resolvable. Outside are not.
#
# Literature entries (800 nm only):
#   Thakur 2019  : n2 = +9.07e-13 m^2/W  SLG monolayer CVD
#   Gonzalez 2024: n2 = -5.01e-17 m^2/W  FLG suspension
#   Wang 2014    : n2 = -2.34e-16 m^2/W  graphene
#   CS2          : n2 = +3.00e-19 m^2/W  reference standard
#
# All parameters fixed to our actual setup — no tuning to make the model behave.
# =============================================================================

import numpy as np
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

from zscan_thick_closed import T_thick_closed, alpha_correction

# =============================================================================
# Fixed setup parameters — these are not tuned
# =============================================================================

LAM    = 800e-9       # wavelength (m)
W0     = 1e-6         # beam waist (m)
L_SAMP = 100e-6       # sample thickness (m)
S      = 0.25         # aperture linear transmittance
P_PEAK = 62.5         # peak power (W)
D_DET  = 0.04         # sample to detector distance (m)
N0     = 1.45         # assumed linear refractive index for graphene samples

Z0     = np.pi * W0**2 / LAM
K      = 2 * np.pi / LAM
I0     = 2 * P_PEAK / (np.pi * W0**2)

w_det_lin = W0 * np.sqrt(1.0 + (D_DET / Z0)**2)
r_a       = w_det_lin * np.sqrt(-np.log(1.0 - S) / 2.0)
A_CORR    = alpha_correction(S)

BEAM_PARAMS = {
    'w0':     W0,
    'z0':     Z0,
    'lam':    LAM,
    'n0':     N0,
    'L':      L_SAMP,
    'I0':     I0,
    'a_corr': A_CORR,
    'd_det':  D_DET,
}

# Upper bound of measurable window: |n2| where DPhi0 = 1 rad
# DPhi0 = K * n2 * I0 * L_SAMP  =>  n2_max = 1 / (K * I0 * L_SAMP)
N2_MAX_WINDOW = 1.0 / (K * I0 * L_SAMP)

# Lower bound: noise floor
TPV_MIN = 0.05

SAVE_DIR = r'C:\PhD\Plots'

SETUP_STR = (f'lambda={LAM*1e9:.0f} nm  w0={W0*1e6:.1f} um  '
             f'z0={Z0*1e6:.2f} um  I0={I0:.2e} W/m2  '
             f'P_peak={P_PEAK:.1f} W  S={S}  L={L_SAMP*1e6:.0f} um')


def save_fig(fig, filename):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved: {path}')


# =============================================================================
# Literature entries
# =============================================================================

PAPERS = [
    {
        'label':  'Thakur 2019\nSLG 800 nm',
        'n2':      9.07e-13,
        'color':  'royalblue',
        'marker': 'o',
    },
    {
        'label':  'Gonzalez 2024\nFLG 800 nm',
        'n2':     -5.01e-17,
        'color':  'darkorange',
        'marker': 's',
    },
    {
        'label':  'Wang 2014\nGraphene 800 nm',
        'n2':     -2.34e-16,
        'color':  'mediumseagreen',
        'marker': '^',
    },
    {
        'label':  'CS2\nReference',
        'n2':      3.00e-19,
        'color':  'black',
        'marker': '*',
    },
]

# =============================================================================
# Compute Tpv sensitivity curve over |n2| sweep
#
# We sweep |n2| and pass the positive value to T_thick_closed.
# The sign of n2 only determines valley-peak vs peak-valley orientation;
# Tpv magnitude is symmetric. Literature signs are recorded separately
# and shown in the legend.
# =============================================================================

print('Computing sensitivity curve...')
print(f'  I0          = {I0:.3e} W/m2')
print(f'  n2_max      = {N2_MAX_WINDOW:.3e} m^2/W  (DPhi0 = 1 rad boundary)')
print(f'  Tpv_min     = {TPV_MIN}  (noise floor)')
print()

# z array for T(z) computation — 5 Rayleigh ranges either side of focus
z_range = 5.0 * max(Z0, L_SAMP / 2.0)
z_arr   = np.linspace(-z_range, z_range, 200)

# Sweep |n2| from well below CS2 to above Thakur
n2_sweep  = np.logspace(-21, -12, 300)
tpv_sweep = np.empty(len(n2_sweep))

for i, n2_v in enumerate(n2_sweep):
    T_v          = T_thick_closed(z_arr, n2_v, BEAM_PARAMS, r_a)
    tpv_sweep[i] = np.max(T_v) - np.min(T_v)

# Interpolate predicted Tpv for each paper and classify
print(f"{'Paper':<32} {'n2':>14}  {'sign':>5}  {'DPhi0':>10}  "
      f"{'Tpv_pred':>10}  {'In window?':>12}")
print('-' * 90)

for p in PAPERS:
    n2_abs   = abs(p['n2'])
    DPhi0    = K * n2_abs * I0 * L_SAMP
    n2_clip  = np.clip(n2_abs, n2_sweep[0], n2_sweep[-1])
    tpv_pred = float(np.interp(n2_clip, n2_sweep, tpv_sweep))
    sign_str = '+' if p['n2'] > 0 else '-'

    in_window = (tpv_pred >= TPV_MIN) and (n2_abs <= N2_MAX_WINDOW)

    p['tpv_pred']  = tpv_pred
    p['DPhi0']     = DPhi0
    p['in_window'] = in_window

    lbl = p['label'].replace('\n', ' ')
    print(f"{lbl:<32} {p['n2']:>14.3e}  {sign_str:>5}  {DPhi0:>10.3f}  "
          f"{tpv_pred:>10.5f}  {'YES' if in_window else 'NO':>12}")

print()

# =============================================================================
# Plot: Sensitivity curve with detection window
# =============================================================================

fig, ax = plt.subplots(figsize=(13, 7))

# --- Sensitivity curve ---
ax.loglog(n2_sweep, tpv_sweep,
          color='steelblue', linewidth=2.5, zorder=2,
          label='Setup sensitivity curve  Tpv(|n2|)')

# --- Detection window shading ---
# Lower bound: horizontal line at Tpv_min
ax.axhline(TPV_MIN, color='crimson', linestyle='--', linewidth=1.5,
           label=f'Noise floor  Tpv_min = {TPV_MIN}')

# Upper bound: vertical line at n2_max (DPhi0 = 1 rad)
ax.axvline(N2_MAX_WINDOW, color='darkorange', linestyle='--', linewidth=1.5,
           label=f'DPhi0 = 1 rad boundary  n2 = {N2_MAX_WINDOW:.2e} m2/W')

# Shade the resolvable region
tpv_lo = ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1e-10
ax.axhspan(TPV_MIN, 10.0,
           xmin=0, xmax=1,
           color='limegreen', alpha=0.07, zorder=0)
ax.fill_betweenx([TPV_MIN, 10.0],
                 n2_sweep[0], N2_MAX_WINDOW,
                 color='limegreen', alpha=0.13, zorder=0,
                 label='Resolvable window')

# Label window bounds on axes
ax.annotate('Detection\nwindow', xy=(1e-20, TPV_MIN * 2.5),
            fontsize=10, color='forestgreen', fontweight='bold')

# --- Literature points ---
for p in PAPERS:
    n2_abs   = abs(p['n2'])
    sign_str = '+' if p['n2'] > 0 else '-'
    lbl_flat = p['label'].replace('\n', ' ')
    edge     = 'black' if p['in_window'] else 'none'
    marker_s = 220

    ax.scatter(n2_abs, p['tpv_pred'],
               color=p['color'], marker=p['marker'],
               s=marker_s, zorder=5, edgecolors=edge, linewidths=1.5,
               label=(f"{lbl_flat}  "
                      f"n2={sign_str}{n2_abs:.2e} m2/W  "
                      f"Tpv={p['tpv_pred']:.4f}  "
                      f"DPhi0={p['DPhi0']:.2f} rad  "
                      f"{'IN' if p['in_window'] else 'OUT'}"))

    # Vertical dashed line from x-axis to point
    ax.axvline(n2_abs, color=p['color'], linestyle=':',
               alpha=0.4, linewidth=1.2)

    # Annotation
    ax.annotate(p['label'],
                xy=(n2_abs, p['tpv_pred']),
                xytext=(7, 6), textcoords='offset points',
                fontsize=8.5, color=p['color'], fontweight='bold')

# --- Axes and labels ---
ax.set_xlim(n2_sweep[0] * 0.5, n2_sweep[-1] * 2)
ax.set_ylim(1e-6, 5.0)

ax.set_xlabel('|n2|  (m2/W)', fontsize=13, fontweight='bold')
ax.set_ylabel('Predicted Tpv', fontsize=13, fontweight='bold')
ax.set_title(
    'Graphene n2 resolvability — our Z-scan setup vs 800 nm literature\n'
    'Black edge = within detection window  |  Points outside window are not measurable\n'
    + SETUP_STR,
    fontsize=10.5, fontweight='bold')

ax.legend(fontsize=8.5, loc='upper left', framealpha=0.95)
plt.tight_layout()
save_fig(fig, 'graphene_resolvability.png')
plt.show()