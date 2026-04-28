import numpy as np
from zscan_model import (
    setup_parameters,
    simulate_closed_aperture,
    calculate_delta_phi
)

beam_params, r_a, k, I0 = setup_parameters()

n2 = 3.00e-19  # CS2

z, Tz, tpv = simulate_closed_aperture(
    n2,
    beam_params,
    r_a
)

DPhi0 = calculate_delta_phi(
    n2,
    k,
    I0,
    beam_params["L"]
)

print("CS2 test")
print("DPhi0 =", DPhi0)
print("Tpv =", tpv)
print("Tmax =", np.max(Tz))
print("Tmin =", np.min(Tz))
print(beam_params["L"])
print(I0)
print(k)
print(n2)