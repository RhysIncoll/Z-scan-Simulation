# zscan_thick_closed.py
# =============================================================================
# Closed aperture Z-scan — Thick medium, distributed lens method
# Sheik-Bahae et al., Optical Engineering 30(8), 1228-1235 (1991)
#
# Physical model:
#   The thick nonlinear sample (L >> z0 or L ~ z0) is sliced into N thin
#   lens elements. Each slice m has:
#     - A local beam radius w_m tracked via ABCD matrix propagation
#     - A local on-axis intensity I_m = I(z_sample) * (w0/w_m)^2
#     - A nonlinear index change Dn_m = n2 * I_m
#     - An effective thin lens focal length f_m (Eq. 3)
#     - An ABCD matrix M_m (Eq. 5)
#
#   The full propagation through the sample is the product of all M_m matrices
#   followed by free-space propagation to the detector at distance d_det.
#
#   The final beam radius at the aperture w_a is extracted from the
#   output q parameter, and the normalised transmittance is computed
#   via Eqs. (7) and (11).
#
# Key equations from Sheik-Bahae 1991:
#
#   Eq. (3):  f_m = a * w_m^2 / (4 * Dn_m * dL)
#   Eq. (5):  M_m = [[1 - dL/(n0*f_m),  dL/n0],
#                    [-1/f_m,             1    ]]
#   Eq. (7):  P_T = P_a * [1 - exp(-2*r_a^2 / w_a^2)]
#   Eq. (11): T(z) = [1 - exp(-2*r_a^2 / w_a^2)] / S
#   Eq. (12): a = 6.4 * (1 - S)^0.35   for S <= 0.7, DPhi0 <= pi/2
#
# Free-space beam tracking:
#   Complex q parameter: q = z' + i*z0_local
#   After ABCD matrix [A,B;C,D]: q_out = (A*q + B) / (C*q + D)
#   Free space propagation distance d: q -> q + d
#   Beam radius: w^2 = -lambda / (pi * Im(1/q))
#
# Expected behaviour:
#   Positive n2 (CS2): valley-peak configuration (peak after focus)
#   Peak-valley separation dominated by L/n0 for thick samples
#   Flat region near centre of sample where pre/post focal effects cancel
#
# Parameters (CS2 at 800nm):
#   n2    = 3e-19 m^2/W   (Kerr nonlinear index, literature value)
#   n0    = 1.63          (linear refractive index of CS2)
#   L     = 100e-6 m      (sample thickness)
#   w0    = 1e-6 m        (beam waist)
#   lam   = 800e-9 m      (wavelength)
#   I0    = 3.98e13 W/m^2 (peak on-axis intensity after EOM)
#   S     = 0.25          (aperture linear transmittance)
#   d_det = 0.04 m        (sample to detector distance)
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

# =============================================================================
# Beam and experimental parameters
# =============================================================================

LAM    = 800e-9       # wavelength (m)
W0     = 1e-6         # beam waist (m)
N0     = 1.63         # CS2 linear refractive index
N2     = 3e-19        # CS2 nonlinear refractive index (m^2/W)
L      = 100e-6       # CS2 sample thickness (m)
D_DET  = 0.04         # sample centre to detector distance (m)
S      = 0.25         # aperture linear transmittance

# Derived beam parameters
Z0     = np.pi * W0**2 / LAM          # Rayleigh range (m)
K      = 2 * np.pi / LAM              # wave vector (rad/m)

# Peak on-axis intensity at focus
# From EOM-reduced rep rate: 80MHz -> ~100kHz
# P_avg_after_EOM = 1W * (100e3/80e6) = 1.25 mW
# E_pulse = 1.25e-3 / 100e3 = 12.5 pJ
# P_peak  = 12.5e-12 / 200e-15 = 62.5 W
# I0 = 2 * P_peak / (pi * W0^2)
P_PEAK = 62.5         # peak power (W)
I0     = 2 * P_PEAK / (np.pi * W0**2)  # ~3.98e13 W/m^2

# Aperture radius from S = 1 - exp(-2*r_a^2 / w_det^2)
# At large d_det, beam radius at detector in linear regime:
# w_det = W0 * sqrt(1 + (D_DET/Z0)^2) ... but for far field w_det >> W0
# We define r_a directly from S using far-field beam radius
# w_det_lin = W0 * D_DET / Z0  (far field approximation)
w_det_lin = W0 * np.sqrt(1.0 + (D_DET / Z0)**2)
r_a = w_det_lin * np.sqrt(-np.log(1.0 - S) / 2.0)

print("=" * 60)
print("CS2 Thick Medium Z-scan Parameters")
print("=" * 60)
print(f"  lambda    = {LAM*1e9:.0f} nm")
print(f"  w0        = {W0*1e6:.1f} um")
print(f"  z0        = {Z0*1e6:.2f} um")
print(f"  L         = {L*1e6:.0f} um")
print(f"  L / z0    = {L/Z0:.2f}  (> 1 => thick medium)")
print(f"  n0        = {N0}")
print(f"  n2        = {N2:.1e} m^2/W")
print(f"  I0        = {I0:.3e} W/m^2")
print(f"  P_peak    = {P_PEAK:.1f} W")
print(f"  S         = {S}")
print(f"  d_det     = {D_DET*100:.0f} cm")
print(f"  r_a       = {r_a*1e6:.2f} um")
print(f"  DPhi0_max = {K*N2*I0*L:.4f} rad  (on-axis, at focus)")
print("=" * 60)

# =============================================================================
# Correction factor a (Sheik-Bahae 1991, Eq. 12)
#
#   a = 6.4 * (1 - S)^0.35    for S <= 0.7 and DPhi0 <= pi/2
#
# For large phase distortions in thick media, a -> 3.77
# We use Eq. (12) here; for large DPhi0 the user should note this limit.
# =============================================================================

def alpha_correction(S_val):
    """
    Eq. (12): correction factor a for parabolic approximation.
    Accounts for higher-order terms omitted in exp(-2r^2/w^2) expansion.
    """
    return 6.4 * (1.0 - S_val)**0.35


A_CORR = alpha_correction(S)
print(f"  a (Eq.12) = {A_CORR:.3f}")
print("=" * 60)


# =============================================================================
# Complex q parameter utilities
# =============================================================================

def q_at_z(z, w0, z0, lam):
    """
    Complex q parameter of free-space Gaussian beam at position z.
    q = z + i*z0  (beam waist at z=0)
    1/q = 1/R - i*lam/(pi*w^2)
    """
    return complex(z, z0)


def w_from_q(q, lam):
    """Extract beam radius w from complex q parameter."""
    q_inv = 1.0 / q
    w_sq  = -lam / (np.pi * np.imag(q_inv))
    if w_sq <= 0:
        return np.nan
    return np.sqrt(w_sq)


def propagate_free_space(q, d):
    """Free-space propagation: q -> q + d."""
    return q + d


def apply_abcd(q, M):
    """Apply ABCD matrix M = [[A,B],[C,D]] to q parameter."""
    A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]
    return (A * q + B) / (C * q + D)


# =============================================================================
# Distributed lens propagation through thick sample
# Sheik-Bahae 1991, Section 2
#
# For sample centred at position z_s (relative to beam waist in air):
#   - Sample runs from z_s - L/2 to z_s + L/2
#   - Divided into N_slices thin lens elements, each of thickness dL = L/N
#   - At each slice m, local beam radius w_m is tracked via q parameter
#   - Local intensity: I_m = I0 * (w0/w_m)^2  [conservation of power]
#   - Nonlinear index: Dn_m = n2 * I_m
#   - Focal length (Eq. 3): f_m = a * w_m^2 / (4 * Dn_m * dL)
#   - ABCD matrix (Eq. 5): propagation through slice + thin lens
#
# After full sample traversal, propagate q to detector via free space.
# Extract w_a from final q, compute T via Eqs. (7) and (11).
# =============================================================================

def propagate_thick_sample(z_s, n2_val, beam_params, N_slices=200):
    """
    Propagate Gaussian beam through thick nonlinear sample at position z_s.

    Uses distributed lens method (Sheik-Bahae 1991, Eq. 5).
    Tracks complex q parameter through N_slices thin lens elements.

    Parameters
    ----------
    z_s         : sample centre position relative to beam waist (m)
    n2_val      : nonlinear refractive index (m^2/W)
    beam_params : dict with keys: w0, z0, lam, n0, L, I0, a_corr
    N_slices    : number of thin lens slices (should satisfy Eq. 4)

    Returns
    -------
    w_a : beam radius at detector aperture (m)
    """
    w0     = beam_params['w0']
    z0     = beam_params['z0']
    lam    = beam_params['lam']
    n0     = beam_params['n0']
    L_samp = beam_params['L']
    I0_val = beam_params['I0']
    a_c    = beam_params['a_corr']
    d_det  = beam_params['d_det']

    dL = L_samp / N_slices   # slice thickness (m)

    # Starting q at entrance face of sample: z_s - L/2
    z_entrance = z_s - L_samp / 2.0
    q = q_at_z(z_entrance, w0, z0, lam)

    # Propagate through each slice
    for m in range(N_slices):

        # Position of slice centre in air coordinates
        z_slice = z_entrance + (m + 0.5) * dL

        # Local beam radius at this slice from q parameter
        w_m = w_from_q(q, lam)
        if np.isnan(w_m) or w_m <= 0:
            return np.nan

        # Local on-axis intensity: I_m = I0 * (w0/w_m)^2
        # This follows from Gaussian beam power conservation
        I_m = I0_val * (w0 / w_m)**2

        # Nonlinear index change at this slice
        Dn_m = n2_val * I_m

        # Eq. (3): focal length of m-th thin lens
        # f_m = a * w_m^2 / (4 * Dn_m * dL)
        # Guard against zero nonlinearity
        if abs(Dn_m) < 1e-30:
            f_m = np.inf
        else:
            f_m = a_c * w_m**2 / (4.0 * Dn_m * dL)

        # Eq. (5): ABCD matrix for slice m
        # [[1 - dL/(n0*f_m),  dL/n0],
        #  [-1/f_m,           1    ]]
        if np.isfinite(f_m):
            M_m = np.array([
                [1.0 - dL / (n0 * f_m),  dL / n0],
                [-1.0 / f_m,              1.0    ]
            ])
        else:
            # No nonlinearity: pure free-space propagation in medium
            M_m = np.array([
                [1.0,       dL / n0],
                [0.0,       1.0    ]
            ])

        # Apply ABCD matrix to q
        q = apply_abcd(q, M_m)

    # Free-space propagation from sample exit to detector
    # Distance: d_det - L/(2*n0) accounts for optical path inside sample
    # Here we use geometric distance from sample centre to detector
    q = propagate_free_space(q, d_det)

    # Extract beam radius at aperture
    w_a = w_from_q(q, lam)
    return w_a


# =============================================================================
# Normalised transmittance via Eqs. (7) and (11)
#
#   Eq. (7):  P_T = P_a * [1 - exp(-2*r_a^2 / w_a^2)]
#   Eq. (11): T(z) = [1 - exp(-2*r_a^2 / w_a^2)] / S
#
# Linear reference uses n2=0, giving w_a_lin at each z.
# T(z) = P_NL / P_lin = [1 - exp(-2ra^2/wa_NL^2)] / [1 - exp(-2ra^2/wa_lin^2)]
# =============================================================================

def aperture_transmittance(r_a_val, w_a_val):
    """Fraction of Gaussian beam power through aperture of radius r_a."""
    if np.isnan(w_a_val) or w_a_val <= 0:
        return 0.0
    return 1.0 - np.exp(-2.0 * r_a_val**2 / w_a_val**2)


def T_thick_closed(z_arr, n2_val, beam_params, r_a_val, N_slices=200):
    """
    Normalised closed aperture transmittance T(z) for thick sample.

    Sheik-Bahae 1991 distributed lens method + Eqs. (7), (11).

    Parameters
    ----------
    z_arr       : array of sample positions (m)
    n2_val      : nonlinear refractive index (m^2/W)
    beam_params : dict of beam/sample parameters
    r_a_val     : aperture radius (m)
    N_slices    : number of distributed lens slices

    Returns
    -------
    T : ndarray, normalised transmittance at each z
    """
    T_out = np.empty(len(z_arr))

    for i, z_s in enumerate(z_arr):

        # Nonlinear propagation
        w_NL  = propagate_thick_sample(z_s, n2_val, beam_params, N_slices)

        # Linear reference (n2=0)
        w_lin = propagate_thick_sample(z_s, 0.0,    beam_params, N_slices)

        P_NL  = aperture_transmittance(r_a_val, w_NL)
        P_lin = aperture_transmittance(r_a_val, w_lin)

        T_out[i] = P_NL / P_lin if P_lin > 1e-30 else 1.0

    return T_out


# =============================================================================
# Run simulation
# =============================================================================

if __name__ == '__main__':

    beam_params = {
        'w0':     W0,
        'z0':     Z0,
        'lam':    LAM,
        'n0':     N0,
        'L':      L,
        'I0':     I0,
        'a_corr': A_CORR,
        'd_det':  D_DET,
    }

    z_range = 5.0 * max(Z0, L / 2.0)
    z_arr   = np.linspace(-z_range, z_range, 200)
    x_norm  = z_arr / Z0

    PARAM_STR = (f'lambda={LAM*1e9:.0f}nm  w0={W0*1e6:.1f}um  '
                 f'z0={Z0*1e6:.2f}um  L={L*1e6:.0f}um  L/z0={L/Z0:.1f}\n'
                 f'n2={N2:.1e} m^2/W  n0={N0}  I0={I0:.2e} W/m^2  '
                 f'S={S}  a={A_CORR:.2f}')

    SAVE_DIR = r'C:\PhD\Plots'

    def save_fig(fig, filename):
        import os
        os.makedirs(SAVE_DIR, exist_ok=True)
        filepath = os.path.join(SAVE_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f'Saved: {filepath}')

    # Plot 1: Single n2 curve
    print('Computing thick closed aperture T(z) for CS2 n2...')
    T_cs2 = T_thick_closed(z_arr, N2, beam_params, r_a)
    Tpv   = np.max(T_cs2) - np.min(T_cs2)
    DPhi0 = K * N2 * I0 * L
    print(f'  DPhi0     = {DPhi0:.4f} rad')
    print(f'  Peak T    = {np.max(T_cs2):.4f}')
    print(f'  Valley T  = {np.min(T_cs2):.4f}')
    print(f'  Tpv       = {Tpv:.4f}')

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x_norm, T_cs2, color='steelblue',
            label=f'CS2 n2={N2:.1e} m^2/W   Tpv={Tpv:.4f}   DPhi0={DPhi0:.2f} rad')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.axvline(0.0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('z / z0', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalised T(z)', fontsize=14, fontweight='bold')
    ax.set_title('Closed aperture — Thick medium, distributed lens methodSheik-Bahae 1991 Eqs. (3)(5)(7)(11)'+ PARAM_STR, fontsize=11, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    save_fig(fig, 'thick_closed_cs2_single.png')
    plt.show()

    # Plot 2: n2 sweep
    N2_VALS = [1e-20, 1e-19, 3e-19, 1e-18, 3e-18]
    colors  = plt.cm.plasma(np.linspace(0.1, 0.9, len(N2_VALS)))
    fig, ax = plt.subplots(figsize=(11, 7))
    print('Computing n2 sweep...')
    for n2_v, col in zip(N2_VALS, colors):
        T_v  = T_thick_closed(z_arr, n2_v, beam_params, r_a)
        tpv  = np.max(T_v) - np.min(T_v)
        dphi = K * n2_v * I0 * L
        print(f'  n2={n2_v:.1e}  DPhi0={dphi:.3f} rad  Tpv={tpv:.4f}')
        ax.plot(x_norm, T_v, color=col, label=f'n2={n2_v:.1e} m^2/W   Tpv={tpv:.4f}')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.axvline(0.0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('z / z0', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalised T(z)', fontsize=14, fontweight='bold')
    ax.set_title('Closed aperture — Thick medium n2 sweepSensitivity check: minimum resolvable n2' + PARAM_STR, fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig(fig, 'thick_closed_n2_sweep.png')
    plt.show()

    # Plot 3: Intensity sweep
    I0_VALS   = [I0, I0/10, I0/100, I0/1000]
    I0_LABELS = ['Full power', '/10', '/100', '/1000']
    colors    = plt.cm.viridis(np.linspace(0.1, 0.9, len(I0_VALS)))
    fig, ax = plt.subplots(figsize=(11, 7))
    print('Computing intensity sweep...')
    for i0_v, col, lbl in zip(I0_VALS, colors, I0_LABELS):
        bp = dict(beam_params)
        bp['I0'] = i0_v
        T_v  = T_thick_closed(z_arr, N2, bp, r_a)
        tpv  = np.max(T_v) - np.min(T_v)
        dphi = K * N2 * i0_v * L
        print(f'  I0={i0_v:.2e}  DPhi0={dphi:.4f} rad  Tpv={tpv:.4f}')
        ax.plot(x_norm, T_v, color=col, label=f'{lbl}: I0={i0_v:.1e}   DPhi0={dphi:.3f} rad   Tpv={tpv:.4f}')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)
    ax.axvline(0.0, color='gray', linestyle=':', alpha=0.4)
    ax.set_xlabel('z / z0', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalised T(z)', fontsize=14, fontweight='bold')
    ax.set_title('Closed aperture — Thick medium, intensity sweep Finding the right power regime' + PARAM_STR, fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_fig(fig, 'thick_closed_intensity_sweep.png')
    plt.show()