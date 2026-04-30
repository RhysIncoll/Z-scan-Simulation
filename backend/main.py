from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from beam import GaussianBeam
from zscan_closed_GD import T_closed_GD
from zscan_open import T_open, q0_on_axis
from zscan_open_SA import T_open_SA
from zscan_thick_open import T_thick_open
from config import BETA, GAMMA, S

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# MODELS
# =============================================================================

class BeamConfig(BaseModel):
    lam: float = 800e-9
    w0: float = 1e-6
    L: float = 100e-6
    S: float = 0.25
    P_peak: float = 62.5
    d_det: float = 0.04
    tpv_min: float = 0.01


class Material(BaseModel):
    name: str
    n2: float
    n0: float
    beta: float = 0.0
    alpha_0: float = 0.0
    I_sat: float = 1e20
    color: str = "steelblue"
    marker: str = "o"


class SimulateRequest(BaseModel):
    beam: BeamConfig
    materials: list[Material]
    use_tpa: bool = False
    use_sa: bool = False


# =============================================================================
# HELPERS
# =============================================================================
def build_beam(cfg: BeamConfig):
    return GaussianBeam(
        lam=cfg.lam,
        w0=cfg.w0,
        L=cfg.L
    )


# =============================================================================
# HEALTH
# =============================================================================

@app.get("/health")
def health():
    return {"status": "ok"}


# =============================================================================
# THIN SAMPLE — CLOSED APERTURE
# =============================================================================

@app.post("/simulate_thin_closed")
def simulate_thin_closed(req: SimulateRequest):
    beam = build_beam(req.beam)

    results = []

    for mat in req.materials:
        try:
            T = T_closed_GD(
                beam=beam,
                beta=mat.beta,
                gamma=mat.n2,
                S=req.beam.S
            )

            T = np.array(T, dtype=float)
            z = np.array(beam.x_norm, dtype=float)

            mask = np.isfinite(z) & np.isfinite(T)
            z = z[mask]
            T = T[mask]

            tpv = float(np.max(T) - np.min(T))
            q0 = float(q0_on_axis(beam, mat.beta))

            results.append({
                "name": mat.name,
                "z": z.tolist(),
                "Tz": T.tolist(),
                "tpv": tpv,
                "q0": q0,
                "n2": mat.n2,
                "beta": mat.beta,
                "color": mat.color,
                "marker": mat.marker,
            })

        except Exception as e:
            results.append({
                "name": mat.name,
                "error": str(e)
            })

    return {"results": results}


# =============================================================================
# THIN SAMPLE — OPEN APERTURE (TPA)
# =============================================================================

@app.post("/simulate_thin_open_tpa")
def simulate_thin_open_tpa(req: SimulateRequest):
    beam = build_beam(req.beam)

    results = []

    for mat in req.materials:
        try:
            T = T_open(
                beam=beam,
                beta=mat.beta
            )

            T = np.array(T, dtype=float)
            z = np.array(beam.x_norm, dtype=float)

            mask = np.isfinite(z) & np.isfinite(T)
            z = z[mask]
            T = T[mask]

            results.append({
                "name": mat.name,
                "z": z.tolist(),
                "Tz": T.tolist(),
                "q0": float(q0_on_axis(beam, mat.beta)),
                "beta": mat.beta,
                "color": mat.color,
                "marker": mat.marker,
            })

        except Exception as e:
            results.append({
                "name": mat.name,
                "error": str(e)
            })

    return {"results": results}


# =============================================================================
# THIN SAMPLE — OPEN APERTURE (SA)
# =============================================================================

@app.post("/simulate_thin_open_sa")
def simulate_thin_open_sa(req: SimulateRequest):
    beam = build_beam(req.beam)

    results = []

    for mat in req.materials:
        try:
            T = T_open_SA(
                beam=beam,
                alpha_0=mat.alpha_0,
                I_sat=mat.I_sat
            )

            T = np.array(T, dtype=float)
            z = np.array(beam.x_norm, dtype=float)

            mask = np.isfinite(z) & np.isfinite(T)
            z = z[mask]
            T = T[mask]

            results.append({
                "name": mat.name,
                "z": z.tolist(),
                "Tz": T.tolist(),
                "alpha_0": mat.alpha_0,
                "I_sat": mat.I_sat,
                "color": mat.color,
                "marker": mat.marker,
            })

        except Exception as e:
            results.append({
                "name": mat.name,
                "error": str(e)
            })

    return {"results": results}