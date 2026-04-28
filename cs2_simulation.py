# cs2_simulation.py
# =============================================================================
# Z-scan simulation — CS2 and material comparison
#
# Purpose:
#   Take any material's reported n2 and beta values from the literature,
#   plug them into YOUR laser setup parameters, and determine:
#     1. What T(z) curve your setup would produce for that material
#     2. What Tpv you would measure
#     3. Whether your setup can resolve that Tpv above the noise floor
#
# Workflow:
#   - Closed aperture T(z) via zscan_thick_closed.py (distributed lens)
#   - Open aperture T(z) via zscan_thick_open.py (2PA slice propagation)
#   - Closed/Open ratio to isolate pure Kerr contribution
#   - Resolvability verdict: RESOLVABLE if Tpv > NOISE_FLOOR
#
# Materials database (literature values at 800nm where available):
#   CS2       : n2 = 3.0e-19 m^2/W,  beta = 0       (Kerr reference)
#   Fused SiO2: n2 = 2.5e-20 m^2/W,  beta = 0
#   BK7 glass : n2 = 3.4e-20 m^2/W,  beta = 0
#   Water     : n2 = 2.7e-20 m^2/W,  beta = 0
#   Toluene   : n2 = 2.8e-19 m^2/W,  beta = 0
#
# Add your own materials to the MATERIALS dict below.
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

from zscan_thick_closed import (T_thick_closed, alpha_correction,
                                 propagate_thick_sample, aperture_transmittance)
from zscan_thick_open   import T_thick_open

# =============================================================================
# YOUR LASER SETUP PARAMETERS
# Modify these to match your actual experimental setup
# =============================================================================

LAM    = 800e-9       # wavelength (m)
W0     = 1e-6         # beam waist (m)
N0_CS2 = 1.63         # CS2 linear refractive index
L_CS2  = 100e-6       # CS2 layer thickness (m)
D_DET  = 0.04         # sample to detector distance (m)
S      = 0.25         # aperture linear transmittance

# Peak intensity — EOM reduced rep rate
# 1W avg / 80MHz * 100kHz EOM = 1.25mW after EOM
# E_pulse = 12.5 pJ, P_peak = 62.5 W
P_PEAK = 2 * 1 / 80e6 / 200e-15
Z0     = np.pi * W0**2 / LAM
K      = 2 * np.pi / LAM
I0     = 2 * P_PEAK / (np.pi * W0**2)

# Aperture radius from S
w_det_lin = W0 * np.sqrt(1.0 + (D_DET / Z0)**2)
r_a       = w_det_lin * np.sqrt(-np.log(1.0 - S) / 2.0)

# Correction factor a (Sheik-Bahae 1991 Eq. 12)
A_CORR = alpha_correction(S)

# Noise floor — minimum Tpv your detector can resolve
# Typical value: 1% (0.01). Conservative: 0.5% (0.005)
NOISE_FLOOR = 0.01

# =============================================================================
# Z scan array
# =============================================================================

z_range = 5.0 * max(Z0, L_CS2 / 2.0)
z_arr   = np.linspace(-z_range, z_range, 200)
x_norm  = z_arr / Z0

# =============================================================================
# Materials database
# Format: 'Name': {'n2': value (m^2/W), 'beta': value (m/W),
#                  'n0': linear index, 'L': thickness (m),
#                  'ref': citation string}
#
# ADD YOUR OWN MATERIALS HERE — copy any entry and change the values
# =============================================================================

MATERIALS = {
    'CS2': {
        'n2':   3.0e-19,
        'beta': 0.0,
        'n0':   1.63,
        'L':    100e-6,
        'ref':  'Sheik-Bahae 1991, gamma~3e-14 cm2/W at 800nm',
        'color': 'steelblue',
    },
    'Fused SiO2': {
        'n2':   2.5e-20,
        'beta': 0.0,
        'n0':   1.45,
        'L':    100e-6,
        'ref':  'Boyd 2008, n2~2.5e-20 m2/W at 800nm',
        'color': 'tomato',
    },
    'BK7 glass': {
        'n2':   3.4e-20,
        'beta': 0.0,
        'n0':   1.52,
        'L':    100e-6,
        'ref':  'Milam 1998, n2~3.4e-20 m2/W',
        'color': 'mediumseagreen',
    },
    'Water': {
        'n2':   2.7e-20,
        'beta': 0.0,
        'n0':   1.33,
        'L':    100e-6,
        'ref':  'Nibbering 1997, n2~2.7e-20 m2/W at 800nm',
        'color': 'mediumpurple',
    },
    'Toluene': {
        'n2':   2.8e-19,
        'beta': 0.0,
        'n0':   1.50,
        'L':    100e-6,
        'ref':  'Couris 2003, n2~2.8e-19 m2/W at 800nm',
        'color': 'darkorange',
    },
}

# =============================================================================
# Helper: build beam_params dict for a given material
# =============================================================================

def make_beam_params(mat):
    """Build beam_params dict for a given material entry."""
    return {
        'w0':     W0,
        'z0':     Z0,
        'lam':    LAM,
        'n0':     mat['n0'],
        'L':      mat['L'],
        'I0':     I0,
        'a_corr': alpha_correction(S),
        'd_det':  D_DET,
    }

# =============================================================================
# Run simulation for all materials
# =============================================================================

SAVE_DIR = r'C:\PhD\Plots'

def save_fig(fig, filename):
    os.makedirs(SAVE_DIR, exist_ok=True)
    filepath = os.path.join(SAVE_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f'Saved: {filepath}')

SETUP_STR = (f'SETUP: lambda={LAM*1e9:.0f}nm  w0={W0*1e6:.1f}um  '
             f'z0={Z0*1e6:.2f}um  I0={I0:.2e} W/m^2\n'
             f'P_peak={P_PEAK:.1f}W  d_det={D_DET*100:.0f}cm  '
             f'S={S}  Noise floor={NOISE_FLOOR*100:.1f}%')

print("=" * 65)
print("CS2 Z-SCAN SIMULATION — MATERIAL RESOLVABILITY CHECK")
print("=" * 65)
print(SETUP_STR)
print(f"\n{'Material':<16} {'n2 (m2/W)':<14} {'DPhi0':<10} "
      f"{'Tpv_closed':<13} {'Tpv_open':<12} {'VERDICT'}")
print("-" * 85)

results = {}

for name, mat in MATERIALS.items():

    bp   = make_beam_params(mat)
    DPhi0 = K * mat['n2'] * I0 * mat['L']
    perturbative = '(perturbative ✓)' if abs(DPhi0) < 1.0 else '(NON-PERTURBATIVE ✗)'    
    # Closed aperture
    T_cl  = T_thick_closed(z_arr, mat['n2'], bp, r_a)
    Tpv_cl = np.max(T_cl) - np.min(T_cl)

    # Open aperture
    T_op   = T_thick_open(z_arr, mat['beta'])
    Tpv_op = 1.0 - np.min(T_op)   # dip depth for open aperture

    # Closed / Open ratio — pure Kerr isolation
    T_ratio    = T_cl / T_op
    Tpv_ratio  = np.max(T_ratio) - np.min(T_ratio)

    # Resolvability verdict
    resolvable = Tpv_cl > NOISE_FLOOR
    verdict    = 'RESOLVABLE ✓' if resolvable else 'BELOW NOISE ✗'

 
    
    results[name] = {
        'T_cl':      T_cl,
        'T_op':      T_op,
        'T_ratio':   T_ratio,
        'Tpv_cl':    Tpv_cl,
        'Tpv_op':    Tpv_op,
        'Tpv_ratio': Tpv_ratio,
        'DPhi0':     DPhi0,
        'resolvable': resolvable,
    }

    print(f'{name:<16} {mat["n2"]:<14.2e} {DPhi0:<10.4f} '
          f'{Tpv_cl:<13.5f} {Tpv_op:<12.5f} {verdict}  {perturbative}')   

print("=" * 85)
print(f'Noise floor = {NOISE_FLOOR*100:.1f}%  |  '
      f'Tpv must exceed this to be experimentally resolvable')
print()

# =============================================================================
# Plot 1: Closed aperture T(z) for all materials
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

for name, mat in MATERIALS.items():
    r = results[name]
    lbl = (f'{name}   n2={mat["n2"]:.1e}   '
           f'Tpv={r["Tpv_cl"]:.5f}  '
           f'{"✓" if r["resolvable"] else "✗"}')
    ax.plot(x_norm, r['T_cl'], color=mat['color'], label=lbl)

ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
ax.axhline(1.0 + NOISE_FLOOR/2, color='red', linestyle='--',
           alpha=0.5, label=f'Noise floor ±{NOISE_FLOOR*100/2:.1f}%')
ax.axhline(1.0 - NOISE_FLOOR/2, color='red', linestyle='--', alpha=0.5)
ax.axvline(0.0, color='gray', linestyle=':', alpha=0.4)
ax.set_xlabel('z / z0', fontsize=14, fontweight='bold')
ax.set_ylabel('Normalised T(z)', fontsize=14, fontweight='bold')
ax.set_title(
    'Closed aperture T(z) — Material comparison\n'
    'Can setup resolve these literature n2 values?\n' + SETUP_STR,
    fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
save_fig(fig, 'cs2_sim_closed_comparison.png')
plt.show()

# =============================================================================
# Plot 2: Tpv bar chart — resolvability at a glance
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

names  = list(results.keys())
tpvs   = [results[n]['Tpv_cl'] for n in names]
colors = [MATERIALS[n]['color'] for n in names]
bars   = ax.bar(names, tpvs, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)

ax.axhline(NOISE_FLOOR, color='red', linestyle='--', linewidth=2,
           label=f'Noise floor = {NOISE_FLOOR*100:.1f}%')

# Label each bar with value and verdict
for bar, tpv, name in zip(bars, tpvs, names):
    verdict = '✓' if tpv > NOISE_FLOOR else '✗'
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + NOISE_FLOOR * 0.05,
            f'{tpv:.4f}\n{verdict}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Peak-Valley Tpv', fontsize=13, fontweight='bold')
ax.set_title(
    'Resolvability check — Tpv vs noise floor\n'
    'Red line = minimum detectable signal\n' + SETUP_STR,
    fontsize=11, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
save_fig(fig, 'cs2_sim_tpv_barchart.png')
plt.show()

# =============================================================================
# Plot 3: Closed / Open ratio for CS2 — pure Kerr isolation
# Demonstrates how dividing out the open aperture removes beta contribution
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

cs2 = MATERIALS['CS2']
r   = results['CS2']

axes[0].plot(x_norm, r['T_cl'], color='steelblue', label='Closed aperture')
axes[0].axhline(1.0, color='gray', linestyle=':', alpha=0.6)
axes[0].set_title('Closed aperture\n(Kerr + any beta)', fontweight='bold')
axes[0].set_xlabel('z / z0', fontweight='bold')
axes[0].set_ylabel('Normalised T(z)', fontweight='bold')
axes[0].legend()

axes[1].plot(x_norm, r['T_op'], color='green', label='Open aperture')
axes[1].axhline(1.0, color='gray', linestyle=':', alpha=0.6)
axes[1].set_title('Open aperture\n(beta only — S=1)', fontweight='bold')
axes[1].set_xlabel('z / z0', fontweight='bold')
axes[1].legend()

axes[2].plot(x_norm, r['T_ratio'], color='purple', linestyle='--',
             label=f'Closed/Open   Tpv={r["Tpv_ratio"]:.5f}')
axes[2].axhline(1.0, color='gray', linestyle=':', alpha=0.6)
axes[2].set_title('Closed / Open\n(pure Kerr — n2 only)', fontweight='bold')
axes[2].set_xlabel('z / z0', fontweight='bold')
axes[2].legend()

plt.suptitle('CS2: Open vs Closed vs Ratio — isolating pure n2\n' + SETUP_STR,
             fontsize=11, fontweight='bold')
plt.tight_layout()
save_fig(fig, 'cs2_sim_open_closed_ratio.png')
plt.show()

# =============================================================================
# Plot 4: What w0 would you need to resolve each material?
# Sweeps w0 and finds minimum n2 resolvable at each beam waist
# Directly answers: "what beam waist do I need for CS2 to be resolvable?"
# =============================================================================

print('\nComputing w0 sensitivity sweep...')

W0_VALS   = np.array([1, 2, 5, 10, 20, 50]) * 1e-6   # beam waists to test (m)
N2_TARGET = [mat['n2'] for mat in MATERIALS.values()]
N2_NAMES  = list(MATERIALS.keys())
N2_COLORS = [mat['color'] for mat in MATERIALS.values()]

fig, ax = plt.subplots(figsize=(11, 7))

for n2_t, nm, col in zip(N2_TARGET, N2_NAMES, N2_COLORS):
    tpvs_w0 = []
    for w0_v in W0_VALS:
        z0_v  = np.pi * w0_v**2 / LAM
        I0_v  = 2 * P_PEAK / (np.pi * w0_v**2)
        w_det = w0_v * np.sqrt(1.0 + (D_DET / z0_v)**2)
        r_a_v = w_det * np.sqrt(-np.log(1.0 - S) / 2.0)
        z_r   = 5.0 * max(z0_v, L_CS2 / 2.0)
        z_v   = np.linspace(-z_r, z_r, 100)

        bp_v = {
            'w0':     w0_v,
            'z0':     z0_v,
            'lam':    LAM,
            'n0':     MATERIALS[nm]['n0'],
            'L':      MATERIALS[nm]['L'],
            'I0':     I0_v,
            'a_corr': alpha_correction(S),
            'd_det':  D_DET,
        }
        T_v  = T_thick_closed(z_v, n2_t, bp_v, r_a_v)
        tpvs_w0.append(np.max(T_v) - np.min(T_v))
        print(f'  {nm}  w0={w0_v*1e6:.0f}um  Tpv={tpvs_w0[-1]:.5f}')

    ax.plot(W0_VALS * 1e6, tpvs_w0, marker='o', color=col,
            label=f'{nm}  n2={n2_t:.1e}')

ax.axhline(NOISE_FLOOR, color='red', linestyle='--', linewidth=2,
           label=f'Noise floor {NOISE_FLOOR*100:.0f}%')
ax.set_xlabel('Beam waist w0 (μm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Peak-Valley Tpv', fontsize=14, fontweight='bold')
ax.set_title(
    'Tpv vs beam waist — finding the optimal w0\n'
    'Above red line = resolvable with our setup\n' + SETUP_STR,
    fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.set_yscale('log')
plt.tight_layout()
save_fig(fig, 'cs2_sim_w0_sweep.png')
plt.show()