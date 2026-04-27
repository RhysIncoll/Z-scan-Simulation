# config.py
# =============================================================================
# Central parameter store for Z-scan simulations
# All values in SI units unless noted in the comment
# Change values here — all other scripts adapt automatically
# =============================================================================

# --- Beam ---
LAM   = 532e-9    # wavelength (m)
W0    = 27e-6     # beam waist radius at focus (m)
I0    = 0.21e13   # peak on-axis irradiance at focus (W/m^2)

# --- Sample ---
L      = 2.7e-3   # physical sample length (m)
ALPHA  = 0.0      # linear absorption coefficient (m^-1), 0 = transparent

# --- Material nonlinear coefficients ---
BETA   = 5.8e-11  # two-photon absorption coefficient (m/W)
GAMMA  = 6.8e-18  # nonlinear refractive index (m^2/W), positive = self-focusing
N0     = 2.7      # linear refractive index

# --- Experiment ---
S      = 0.40     # aperture linear transmittance (dimensionless, 0 < S < 1)

# --- Scan grid ---
X_MIN  = -8.0     # scan range in units of z0
X_MAX  =  8.0
N_Z    =  600     # number of z points

# --- Detector distance (far-field condition: d >> z0) ---
D_FACTOR = 500.0  # detector placed at D_FACTOR * z0


