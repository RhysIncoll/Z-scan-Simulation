import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from zscan_model import setup_parameters, simulate_closed_aperture

# =========================================================
# Standard test beam — thin-ish sample so physics is clean
# =========================================================

@pytest.fixture
def setup():
    beam_params, r_a, k, I0 = setup_parameters(
        lam=800e-9, w0=5e-6, L=1e-6, S=0.25, P_peak=62.5, d_det=0.04
    )
    return beam_params, r_a, k, I0


# =========================================================
# TEST 1 — n2=0 should give flat T(z)=1
# =========================================================

def test_flat_trace_no_n2(setup):
    beam_params, r_a, k, I0 = setup
    z, Tz, tpv = simulate_closed_aperture(0.0, beam_params, r_a)
    assert tpv < 1e-6, f"Expected flat trace for n2=0, got Tpv={tpv}"


# =========================================================
# TEST 2 — positive n2 → valley before peak
# =========================================================

def test_positive_n2_valley_then_peak(setup):
    beam_params, r_a, k, I0 = setup
    z, Tz, tpv = simulate_closed_aperture(1e-18, beam_params, r_a)
    valley_z = z[np.argmin(Tz)]
    peak_z   = z[np.argmax(Tz)]
    assert valley_z < peak_z, \
        f"Positive n2 should have valley before peak, got valley={valley_z:.2e} peak={peak_z:.2e}"


# =========================================================
# TEST 3 — negative n2 → peak before valley
# =========================================================

def test_negative_n2_peak_then_valley(setup):
    beam_params, r_a, k, I0 = setup
    z, Tz, tpv = simulate_closed_aperture(-1e-18, beam_params, r_a)
    valley_z = z[np.argmin(Tz)]
    peak_z   = z[np.argmax(Tz)]
    assert peak_z < valley_z, \
        f"Negative n2 should have peak before valley, got peak={peak_z:.2e} valley={valley_z:.2e}"


# =========================================================
# TEST 4 — larger |n2| → larger Tpv
# =========================================================

def test_larger_n2_larger_tpv(setup):
    beam_params, r_a, k, I0 = setup
    n2_vals = [1e-20, 1e-19, 1e-18]
    tpvs = []
    for n2 in n2_vals:
        _, _, tpv = simulate_closed_aperture(n2, beam_params, r_a)
        tpvs.append(tpv)
    assert tpvs[0] < tpvs[1] < tpvs[2], \
        f"Tpv should increase with |n2|, got {tpvs}"


# =========================================================
# TEST 5 — TPA reduces the peak of T(z)
# =========================================================

def test_tpa_reduces_peak(setup):
    beam_params, r_a, k, I0 = setup
    _, Tz_kerr, _ = simulate_closed_aperture(
        1e-18, beam_params, r_a, use_tpa=False
    )
    _, Tz_tpa, _ = simulate_closed_aperture(
        1e-18, beam_params, r_a, beta=1e-11, use_tpa=True
    )
    assert np.max(Tz_tpa) < np.max(Tz_kerr), \
        "TPA should reduce the peak of T(z)"


# =========================================================
# TEST 6 — SA increases transmission at focus
# =========================================================
def test_sa_increases_transmission_at_focus(setup):
    """SA: transmission at focus should be higher than far from focus
    because bleaching is stronger at high intensity."""
    beam_params, r_a, k, I0 = setup
    z, Tz_sa, _ = simulate_closed_aperture(
        0.0, beam_params, r_a,
        alpha_0=100.0, I_sat=1e12, use_sa=True
    )
    idx_focus = np.argmin(np.abs(z))
    idx_far   = 0
    assert Tz_sa[idx_focus] > Tz_sa[idx_far], \
        "SA: transmission at focus should exceed transmission far from focus"