# beam.py
# =============================================================================
# GaussianBeam class — defines the beam and spatial grid once.
# All physics scripts import this and work from the same arrays.
# =============================================================================

import numpy as np
from config import LAM, W0, I0, L, ALPHA, D_FACTOR, X_MIN, X_MAX, N_Z


class GaussianBeam:
    """
    Defines a focused TEM00 Gaussian beam and its derived quantities.

    All quantities are in SI units throughout.

    Parameters
    ----------
    lam      : wavelength (m)
    w0       : beam waist radius at focus (m)
    I0       : peak on-axis irradiance at focus (W/m^2)
    L        : sample length (m)
    alpha    : linear absorption coefficient (m^-1)
    d_factor : detector distance in units of z0
    x_min, x_max, n_z : scan grid in units of z0
    """

    def __init__(
        self,
        lam      = LAM,
        w0       = W0,
        I0       = I0,
        L        = L,
        alpha    = ALPHA,
        d_factor = D_FACTOR,
        x_min    = X_MIN,
        x_max    = X_MAX,
        n_z      = N_Z,
    ):
        # --- Fundamental beam parameters ---
        self.lam   = lam
        self.w0    = w0
        self.I0    = I0
        self.k     = 2.0 * np.pi / lam
        self.z0    = np.pi * w0**2 / lam   # Rayleigh range (m)

        # --- Sample ---
        self.L     = L
        self.alpha = alpha
        self.Leff  = (1.0 - np.exp(-alpha * L)) / alpha if alpha > 1e-10 else L

        # --- Scan grid ---
        self.x_norm = np.linspace(x_min, x_max, n_z)   # z / z0
        self.z_arr  = self.x_norm * self.z0             # physical z (m)

        # --- Detector and aperture geometry ---
        # Detector placed far enough to satisfy d >> z0
        self.d_det     = d_factor * self.z0
        self.w_det_lin = w0 * np.sqrt(1.0 + (self.d_det / self.z0)**2)

        # --- Pre-computed beam profiles on z_arr ---
        self.wz  = self._wz(self.z_arr)    # beam radius profile (m)
        self.Rz  = self._Rz(self.z_arr)    # wavefront radius profile (m)
        self.Iz  = self._Iz(self.z_arr)    # on-axis irradiance profile (W/m^2)
        self.qz  = self._qz(self.z_arr)    # complex q parameter profile

    # -------------------------------------------------------------------------
    # Beam profile methods — accept scalar or array z
    # -------------------------------------------------------------------------

    def _wz(self, z):
        """Beam radius w(z) = w0 * sqrt(1 + (z/z0)^2)"""
        return self.w0 * np.sqrt(1.0 + (z / self.z0)**2)

    def _Rz(self, z):
        """
        Wavefront radius of curvature R(z) = z * (1 + (z0/z)^2).
        Returns np.inf at z=0 (planar wavefront at focus).
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        with np.errstate(divide='ignore', invalid='ignore'):
            R = np.where(np.abs(z) < 1e-20, np.inf, z * (1.0 + (self.z0 / z)**2))
        return R if R.size > 1 else float(R)

    def _Iz(self, z):
        """On-axis irradiance I(z) = I0 / (1 + (z/z0)^2)"""
        return self.I0 / (1.0 + (z / self.z0)**2)

    def _qz(self, z):
        """
        Complex beam parameter q(z).
        1/q = 1/R - i*lam/(pi*w^2)
        At focus (z=0): q = i*z0 (pure imaginary, planar wavefront)
        """
        z   = np.atleast_1d(np.asarray(z, dtype=float))
        w2  = self._wz(z)**2
        R   = np.atleast_1d(np.asarray(self._Rz(z), dtype=float))

        inv_q = np.where(
            np.isinf(R),
            -1j * self.lam / (np.pi * w2),
            1.0 / R - 1j * self.lam / (np.pi * w2)
        )
        q = 1.0 / inv_q
        return q if q.size > 1 else complex(q)

    # -------------------------------------------------------------------------
    # Aperture helpers
    # -------------------------------------------------------------------------

    def aperture_radius(self, S):
        """
        Physical aperture radius r_a for a given linear transmittance S.
        Derived from S = 1 - exp(-2 * r_a^2 / w_det_lin^2).
        """
        if S <= 0.0 or S >= 1.0:
            raise ValueError(f"S must be in (0, 1), got {S}")
        return self.w_det_lin * np.sqrt(-0.5 * np.log(1.0 - S))

    def S_from_aperture_radius(self, r_a):
        """Linear transmittance S for a given physical aperture radius r_a."""
        return 1.0 - np.exp(-2.0 * (r_a / self.w_det_lin)**2)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(self):
        print("=" * 60)
        print("GaussianBeam summary")
        print("=" * 60)
        print(f"  lambda     = {self.lam*1e9:.1f} nm")
        print(f"  w0         = {self.w0*1e6:.2f} um")
        print(f"  z0         = {self.z0*1e3:.3f} mm")
        print(f"  k          = {self.k:.4e} m^-1")
        print(f"  I0         = {self.I0:.3e} W/m^2")
        print(f"  L          = {self.L*1e3:.2f} mm")
        print(f"  alpha      = {self.alpha:.3e} m^-1")
        print(f"  Leff       = {self.Leff*1e3:.4f} mm")
        print(f"  d_det      = {self.d_det*1e3:.1f} mm  ({self.d_det/self.z0:.0f} * z0)")
        print(f"  w_det_lin  = {self.w_det_lin*1e3:.3f} mm")
        print(f"  z_arr      : {len(self.z_arr)} points from"
              f" {self.x_norm[0]:.1f} to {self.x_norm[-1]:.1f} z0")
        print("=" * 60)