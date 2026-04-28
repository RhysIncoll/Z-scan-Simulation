# --- local overrides for this run ---
# config.py stays unchanged, these only apply here
from config import GAMMA, S   # keep these from config
LOCAL_GAMMA = 3.4e-18         # override just gamma
LOCAL_S     = 0.20            # override just S
LOCAL_BEAM  = GaussianBeam(w0=15e-6, I0=0.5e13)  # override beam params

T = T_closed_ABCD(LOCAL_BEAM, LOCAL_GAMMA, LOCAL_S)