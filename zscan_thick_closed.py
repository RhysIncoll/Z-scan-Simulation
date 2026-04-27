# zscan_thick_closed.py
# =============================================================================
# Closed aperture Z-scan — Thick medium, distributed lens method
# Sheik-Bahae et al., Optical Engineering 30(8), 1228-1235 (1991)
#
# Extended to include:
#   - Two-photon absorption (TPA): beta (m/W)
#       dI/dz = -beta * I^2  per slice (analytic step)
#   - Saturable absorption (SA): alpha_0 (m^-1), I_sat (W/m^2)
#       dI/dz = -[alpha_0 / (1 + I/I_sat)] * I  per slice (RK4)
#
# Both absorptive effects are tracked as a scalar amplitude factor
# alongside the q-parameter beam propagation. At each slice:
#   1. q parameter updated via ABCD matrix (Kerr lensing, unchanged)
#   2. Intensity attenuated by TPA and/or SA (amplitude factor updated)
#
# The final aperture power is scaled by amp_factor^2 before normalisation.
# =============================================================================

import numpy as np


# =============================================================================
# Correction factor (Sheik-Bahae 1991, Eq. 12)
# =============================================================================

def alpha_correction(S_val):
    """a = 6.4 * (1 - S)^0.35  for S <= 0.7, DPhi0 <= pi/2"""
    return 6.4 * (1.0 - S_val) ** 0.35


# =============================================================================
# Complex q parameter utilities
# =============================================================================

def q_at_z(z, w0, z0):
    """Complex q at position z: q = z + i*z0"""
    return complex(z, z0)


def w_from_q(q, lam):
    """Beam radius from q parameter."""
    q_inv = 1.0 / q
    w_sq  = -lam / (np.pi * np.imag(q_inv))
    return np.sqrt(w_sq) if w_sq > 0 else np.nan


def apply_abcd(q, M):
    """Apply ABCD matrix to q parameter."""
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    return (A * q + B) / (C * q + D)


# =============================================================================
# Per-slice absorptive attenuation
# =============================================================================

def _tpa_step(I_m, beta, dL):
    """
    Analytic TPA attenuation over one slice.
    I_out = I_in / (1 + beta * I_in * dL)
    Returns amplitude ratio sqrt(I_out / I_in).
    """
    if beta == 0.0 or I_m <= 0:
        return 1.0
    I_out = I_m / (1.0 + beta * I_m * dL)
    return np.sqrt(I_out / I_m)


def _sa_step(I_m, alpha_0, I_sat, dL, N_rk=4):
    """
    SA attenuation over one slice via RK4.
    dI/dz = -[alpha_0 / (1 + I/I_sat)] * I
    Returns amplitude ratio sqrt(I_out / I_in).
    """
    if alpha_0 == 0.0 or I_m <= 0:
        return 1.0

    def dIdz(I):
        return -(alpha_0 / (1.0 + I / I_sat)) * I

    I = float(I_m)
    h = dL / N_rk
    for _ in range(N_rk):
        k1 = dIdz(I)
        k2 = dIdz(I + 0.5 * h * k1)
        k3 = dIdz(I + 0.5 * h * k2)
        k4 = dIdz(I + h * k3)
        I  = I + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        if I < 0:
            I = 0.0
            break

    return np.sqrt(I / I_m) if I_m > 0 else 1.0


# =============================================================================
# Core propagation
# =============================================================================

def propagate_thick_sample(z_s, n2_val, beam_params, N_slices=200,
                           beta=0.0, alpha_0=0.0, I_sat=1e20,
                           use_tpa=False, use_sa=False):
    """
    Propagate Gaussian beam through thick nonlinear sample at position z_s.

    Tracks:
      - Complex q parameter for beam size / Kerr lensing (ABCD)
      - Scalar amplitude factor for absorptive losses (TPA, SA)

    Parameters
    ----------
    z_s         : sample centre position (m)
    n2_val      : nonlinear refractive index (m^2/W)
    beam_params : dict — w0, z0, lam, n0, L, I0, a_corr, d_det
    N_slices    : number of distributed lens slices
    beta        : TPA coefficient (m/W), used if use_tpa=True
    alpha_0     : SA linear absorption coefficient (m^-1), used if use_sa=True
    I_sat       : SA saturation intensity (W/m^2), used if use_sa=True
    use_tpa     : enable TPA attenuation per slice
    use_sa      : enable SA attenuation per slice

    Returns
    -------
    w_a         : beam radius at aperture (m), or nan on failure
    amp_factor  : cumulative field amplitude factor from absorption (0-1)
    """
    w0     = beam_params['w0']
    z0     = beam_params['z0']
    lam    = beam_params['lam']
    n0     = beam_params['n0']
    L_samp = beam_params['L']
    I0_val = beam_params['I0']
    a_c    = beam_params['a_corr']
    d_det  = beam_params['d_det']

    dL         = L_samp / N_slices
    z_entrance = z_s - L_samp / 2.0
    q          = q_at_z(z_entrance, w0, z0)
    amp_factor = 1.0          # cumulative field amplitude (not intensity)

    for m in range(N_slices):

        # Local beam radius from current q
        w_m = w_from_q(q, lam)
        if np.isnan(w_m) or w_m <= 0:
            return np.nan, 0.0

        # Local on-axis intensity (power conservation)
        I_m = I0_val * (w0 / w_m) ** 2

        # ------------------------------------------------------------------
        # Absorptive attenuation (amplitude)
        # Order: TPA then SA (or whichever are enabled)
        # ------------------------------------------------------------------
        if use_tpa and beta != 0.0:
            amp_factor *= _tpa_step(I_m, beta, dL)
            # Update effective intensity seen by subsequent effects
            I_m *= (_tpa_step(I_m, beta, dL)) ** 2

        if use_sa and alpha_0 != 0.0:
            amp_factor *= _sa_step(I_m, alpha_0, I_sat, dL)

        # ------------------------------------------------------------------
        # Kerr lensing — ABCD matrix (unchanged from original)
        # ------------------------------------------------------------------
        Dn_m = n2_val * I_m

        if abs(Dn_m) < 1e-30:
            f_m = np.inf
        else:
            f_m = a_c * w_m ** 2 / (4.0 * Dn_m * dL)

        if np.isfinite(f_m):
            M_m = np.array([
                [1.0 - dL / (n0 * f_m),  dL / n0],
                [-1.0 / f_m,              1.0    ]
            ])
        else:
            M_m = np.array([
                [1.0,  dL / n0],
                [0.0,  1.0    ]
            ])

        q = apply_abcd(q, M_m)

    # Free-space propagation to detector
    q += d_det

    w_a = w_from_q(q, lam)
    return w_a, amp_factor


# =============================================================================
# Aperture transmittance
# =============================================================================

def aperture_transmittance(r_a_val, w_a_val):
    """Fraction of Gaussian beam power through circular aperture."""
    if np.isnan(w_a_val) or w_a_val <= 0:
        return 0.0
    return 1.0 - np.exp(-2.0 * r_a_val ** 2 / w_a_val ** 2)


# =============================================================================
# Main T(z) function
# =============================================================================

def T_thick_closed(z_arr, n2_val, beam_params, r_a_val, N_slices=200,
                   beta=0.0, alpha_0=0.0, I_sat=1e20,
                   use_tpa=False, use_sa=False):
    """
    Normalised closed aperture transmittance T(z) for thick sample.

    Sheik-Bahae 1991 distributed lens + optional TPA/SA per slice.

    Parameters
    ----------
    z_arr       : sample centre positions (m)
    n2_val      : nonlinear refractive index (m^2/W)
    beam_params : dict of beam/sample parameters
    r_a_val     : aperture radius (m)
    N_slices    : distributed lens slices
    beta        : TPA coefficient (m/W)
    alpha_0     : SA linear absorption (m^-1)
    I_sat       : SA saturation intensity (W/m^2)
    use_tpa     : enable TPA
    use_sa      : enable SA

    Returns
    -------
    T : ndarray, normalised transmittance
    """
    T_out = np.empty(len(z_arr))

    for i, z_s in enumerate(z_arr):

        # Nonlinear propagation (Kerr + enabled absorptive effects)
        w_NL,  amp_NL  = propagate_thick_sample(
            z_s, n2_val, beam_params, N_slices,
            beta, alpha_0, I_sat, use_tpa, use_sa
        )

        # Linear reference — n2=0, no absorption
        w_lin, amp_lin = propagate_thick_sample(
            z_s, 0.0, beam_params, N_slices,
            0.0, 0.0, 1e20, False, False
        )

        P_NL  = aperture_transmittance(r_a_val, w_NL)  * amp_NL  ** 2
        P_lin = aperture_transmittance(r_a_val, w_lin) * amp_lin ** 2

        T_out[i] = P_NL / P_lin if P_lin > 1e-30 else 1.0

    return T_out