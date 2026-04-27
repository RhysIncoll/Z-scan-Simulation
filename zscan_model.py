# zscan_model.py
# =============================================================================
# Bridge layer between Streamlit app and zscan_thick_closed.py
# Handles setup, per-material simulation dispatch, and result packaging.
# =============================================================================

import numpy as np
from zscan_thick_closed import T_thick_closed, alpha_correction


def setup_parameters(
    lam=800e-9,
    w0=1e-6,
    L=100e-6,
    S=0.25,
    P_peak=62.5,
    d_det=0.04,
    n0=1.45
):
    """
    Compute beam and setup parameters from sidebar inputs.

    Returns
    -------
    beam_params : dict
    r_a         : aperture radius (m)
    k           : wave vector (rad/m)
    I0          : peak on-axis intensity (W/m^2)
    """
    z0 = np.pi * w0 ** 2 / lam
    k  = 2 * np.pi / lam
    I0 = 2 * P_peak / (np.pi * w0 ** 2)

    w_det_lin = w0 * np.sqrt(1.0 + (d_det / z0) ** 2)
    r_a       = w_det_lin * np.sqrt(-np.log(1.0 - S) / 2.0)
    a_corr    = alpha_correction(S)

    beam_params = {
        "w0":     w0,
        "z0":     z0,
        "lam":    lam,
        "n0":     n0,
        "L":      L,
        "I0":     I0,
        "a_corr": a_corr,
        "d_det":  d_det,
    }

    return beam_params, r_a, k, I0


def simulate_closed_aperture(n2, beam_params, r_a,
                             beta=0.0, alpha_0=0.0, I_sat=1e20,
                             use_tpa=False, use_sa=False):
    """
    Run closed-aperture Z-scan simulation for one material configuration.

    Parameters
    ----------
    n2          : nonlinear refractive index (m^2/W), signed
    beam_params : dict from setup_parameters()
    r_a         : aperture radius (m)
    beta        : TPA coefficient (m/W)
    alpha_0     : SA linear absorption coefficient (m^-1)
    I_sat       : SA saturation intensity (W/m^2)
    use_tpa     : enable TPA in simulation
    use_sa      : enable SA in simulation

    Returns
    -------
    z_arr : ndarray, sample positions (m)
    Tz    : ndarray, normalised transmittance
    tpv   : float, peak-to-valley value
    """
    z0 = beam_params["z0"]
    L  = beam_params["L"]

    z_range = 5.0 * max(z0, L / 2.0)
    z_arr   = np.linspace(-z_range, z_range, 400)

    Tz = T_thick_closed(
        z_arr, n2, beam_params, r_a,
        N_slices=200,
        beta=beta,
        alpha_0=alpha_0,
        I_sat=I_sat,
        use_tpa=use_tpa,
        use_sa=use_sa
    )

    # Clean any numerical artefacts
    mask = np.isfinite(Tz)
    if mask.sum() < 2:
        Tz = np.ones_like(z_arr)

    tpv = float(np.max(Tz) - np.min(Tz))

    return z_arr, Tz, tpv


def calculate_delta_phi(n2, k, I0, L):
    """On-axis nonlinear phase shift ΔΦ₀ = k |n2| I0 L"""
    return k * abs(n2) * I0 * L


def calculate_q0(beta, I0, L):
    """On-axis TPA parameter q0 = β I0 L"""
    return beta * I0 * L