import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from zscan_model import (
    setup_parameters,
    simulate_closed_aperture,
    calculate_delta_phi,
    calculate_q0
)

# =========================================================
# GLOBAL PLOT STYLE
# =========================================================

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.2,
    'grid.linestyle':    '--',
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'legend.framealpha': 0.95,
    'legend.edgecolor':  '#cccccc',
})

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(page_title="Z-scan Research Dashboard", layout="wide")
st.title("Closed-Aperture Z-scan Research Dashboard")

# =========================================================
# DEFAULT MATERIAL DATABASE
# =========================================================

if "materials" not in st.session_state:
    st.session_state.materials = [
        {
            "name":    "CS2 Reference",
            "n2":      3.2e-18,
            "n0":      1.63,
            "beta":    0.0,
            "alpha_0": 0.0,
            "I_sat":   1e20,
            "color":   "black",
            "marker":  "*",
        },
        {
            "name":    "Graphene FLG",
            "n2":      -5.01e-17,
            "n0":      1.45,
            "beta":    0.0,
            "alpha_0": 0.0,
            "I_sat":   1e20,
            "color":   "darkorange",
            "marker":  "s",
        },
        {
            "name":    "Wang 2014 Graphene",
            "n2":      -2.34e-16,
            "n0":      1.45,
            "beta":    0.0,
            "alpha_0": 0.0,
            "I_sat":   1e20,
            "color":   "mediumseagreen",
            "marker":  "^",
        },
    ]

# =========================================================
# SIDEBAR — BEAM PARAMETERS
# =========================================================

st.sidebar.header("Beam / Experiment Parameters")

lam    = st.sidebar.number_input("Wavelength λ (m)",         value=800e-9, format="%.3e")
w0     = st.sidebar.number_input("Beam waist w0 (m)",        value=1e-6,   format="%.3e")
L      = st.sidebar.number_input("Sample thickness L (m)",   value=100e-6, format="%.3e")
S      = st.sidebar.number_input("Aperture transmittance S", value=0.25)
P_peak = st.sidebar.number_input("Peak power (W)",           value=62.5)
d_det  = st.sidebar.number_input("Detector distance (m)",    value=0.04)
TPV_MIN= st.sidebar.number_input("Noise floor Tpv_min",      value=0.01)

st.sidebar.divider()
st.sidebar.header("Simulation Toggles")
enable_tpa = st.sidebar.toggle("Enable TPA (β)", value=False)
enable_sa  = st.sidebar.toggle("Enable SA (α₀, I_sat)", value=False)

if enable_tpa:
    st.sidebar.caption("TPA active — amplitude attenuated per slice via β")
if enable_sa:
    st.sidebar.caption("SA active — amplitude attenuated per slice via α₀ / I_sat")

# =========================================================
# SETUP PARAMETERS
# =========================================================

beam_params, r_a, k, I0 = setup_parameters(
    lam=lam, w0=w0, L=L, S=S, P_peak=P_peak, d_det=d_det
)
z0 = beam_params["z0"]

SETUP_STR = (
    f"λ={lam*1e9:.0f} nm  w0={w0*1e6:.1f} µm  "
    f"z0={z0*1e6:.2f} µm  I0={I0:.2e} W/m²  "
    f"P_peak={P_peak:.1f} W  S={S}  L={L*1e6:.0f} µm"
)

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Material Manager",
    "Sensitivity Plot",
    "T(z) Plot",
    "Results Table",
])

# =========================================================
# TAB 1 — MATERIAL MANAGER
# =========================================================

with tab1:
    st.header("Add New Material")

    name = st.text_input("Material Name")

    c1, c2 = st.columns(2)
    with c1:
        n2_new = st.number_input("n2 (m²/W)", value=-1e-17, format="%.3e")
    with c2:
        n0_new = st.number_input("n0", value=1.45)

    st.markdown("**Nonlinear Absorption Parameters**")
    c3, c4 = st.columns(2)
    with c3:
        beta_new = st.number_input(
            "β — TPA coefficient (m/W)",
            value=0.0, format="%.3e",
            help="Two-photon absorption. 0 = disabled."
        )
    with c4:
        alpha0_new = st.number_input(
            "α₀ — SA linear absorption (m⁻¹)",
            value=0.0, format="%.3e",
            help="Linear absorption for saturable absorber. 0 = disabled."
        )

    c5, c6 = st.columns(2)
    with c5:
        Isat_new = st.number_input(
            "I_sat — SA saturation intensity (W/m²)",
            value=1e13, format="%.3e",
            help="Only used if α₀ > 0."
        )
    with c6:
        color_new = st.text_input("Plot Color", value="royalblue")

    marker_new = st.text_input("Marker", value="o")

    if st.button("Add Material"):
        if name.strip():
            st.session_state.materials.append({
                "name":    name,
                "n2":      n2_new,
                "n0":      n0_new,
                "beta":    beta_new,
                "alpha_0": alpha0_new,
                "I_sat":   Isat_new,
                "color":   color_new,
                "marker":  marker_new,
            })
            st.success(f"{name} added.")

    st.divider()
    st.subheader("Current Materials")

    for i, mat in enumerate(st.session_state.materials):
        with st.expander(f"{mat['name']}"):
            ec1, ec2 = st.columns([4, 1])
            with ec1:
                st.write(
                    f"**n2** = {mat['n2']:.3e} m²/W  |  "
                    f"**n0** = {mat['n0']}  |  "
                    f"**β** = {mat['beta']:.2e} m/W  |  "
                    f"**α₀** = {mat['alpha_0']:.2e} m⁻¹  |  "
                    f"**I_sat** = {mat['I_sat']:.2e} W/m²"
                )
            with ec2:
                if st.button("Remove", key=f"del_{i}"):
                    st.session_state.materials.pop(i)
                    st.rerun()

# =========================================================
# PRECOMPUTE RESULTS
# =========================================================

results = []

for mat in st.session_state.materials:
    try:
        z, Tz, tpv = simulate_closed_aperture(
            n2          = mat["n2"],
            beam_params = beam_params,
            r_a         = r_a,
            beta        = mat["beta"],
            alpha_0     = mat["alpha_0"],
            I_sat       = mat["I_sat"],
            use_tpa     = enable_tpa,
            use_sa      = enable_sa,
        )

        z  = np.array(z,  dtype=float)
        Tz = np.array(Tz, dtype=float)
        mask = np.isfinite(z) & np.isfinite(Tz)
        z  = z[mask]
        Tz = Tz[mask]
        if len(z) == 0:
            continue

        tpv        = float(np.max(Tz) - np.min(Tz))
        dphi       = calculate_delta_phi(mat["n2"], k, I0, L)
        q0         = calculate_q0(mat["beta"], I0, L)
        detectable = tpv >= TPV_MIN

        results.append({
            "name":       mat["name"],
            "z":          z,
            "Tz":         Tz,
            "n2":         mat["n2"],
            "n0":         mat["n0"],
            "beta":       mat["beta"],
            "alpha_0":    mat["alpha_0"],
            "I_sat":      mat["I_sat"],
            "color":      mat["color"],
            "marker":     mat["marker"],
            "tpv":        tpv,
            "dphi":       dphi,
            "q0":         q0,
            "detectable": detectable,
        })

    except Exception as e:
        st.error(f"{mat['name']} failed: {str(e)}")

# =========================================================
# TAB 2 — SENSITIVITY PLOT
# =========================================================

with tab2:
    st.header("Sensitivity Plot")

    if not results:
        st.error("No valid simulation results.")
    else:
        n2_dphi1  = 1.0 / (k * I0 * L)
        n2_sweep  = np.logspace(-23, -12, 80)
        tpv_sweep = []

        for n2_val in n2_sweep:
            try:
                _, Tz_tmp, _ = simulate_closed_aperture(
                    n2=n2_val, beam_params=beam_params, r_a=r_a,
                    use_tpa=False, use_sa=False
                )
                Tz_tmp  = np.array(Tz_tmp, dtype=float)
                Tz_tmp  = Tz_tmp[np.isfinite(Tz_tmp)]
                tpv_val = float(np.max(Tz_tmp) - np.min(Tz_tmp)) if len(Tz_tmp) else 1e-12
                tpv_val = max(tpv_val, 1e-12)
            except Exception:
                tpv_val = 1e-12
            tpv_sweep.append(tpv_val)

        toggle_parts = []
        if enable_tpa: toggle_parts.append("TPA on")
        if enable_sa:  toggle_parts.append("SA on")
        toggle_label = ("  |  " + "  ".join(toggle_parts)) if toggle_parts else ""

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.loglog(n2_sweep, tpv_sweep,
                  color='steelblue', linewidth=2.2, zorder=2,
                  label='Sensitivity curve')
        ax.axvline(n2_dphi1, color='gray', linestyle=':', linewidth=1.4,
                   label=f'ΔΦ₀ = 1 rad  n2 = {n2_dphi1:.1e} m²/W')
        ax.axhline(TPV_MIN, color='crimson', linestyle='--', linewidth=1.4,
                   label=f'Noise floor  Tpv = {TPV_MIN}')
        ax.axhspan(TPV_MIN, 20.0, color='limegreen', alpha=0.07, zorder=0,
                   label='Detectable region')

        for r in results:
            n2_abs   = abs(r["n2"])
            tpv_plot = max(r["tpv"], 1e-12)
            edge     = 'black' if r["detectable"] else 'none'
            sign     = '+' if r["n2"] > 0 else '-'
            ax.scatter(n2_abs, tpv_plot,
                       s=180, color=r["color"], marker=r["marker"],
                       zorder=5, edgecolors=edge, linewidths=1.5,
                       label=f"n2={sign}{n2_abs:.1e}  Tpv={r['tpv']:.4f}")
            ax.annotate(r["name"], xy=(n2_abs, tpv_plot),
                        xytext=(7, 4), textcoords='offset points',
                        fontsize=8, color=r["color"], fontweight='bold')

        ax.set_xlim(n2_sweep[0] * 0.5, n2_sweep[-1] * 2)
        ax.set_ylim(1e-6, 10.0)
        ax.set_xlabel('|n2|  (m²/W)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted Tpv', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Sensitivity curve{toggle_label}\n'
            'Black edge = detectable  |  Green = above noise floor\n' + SETUP_STR,
            fontsize=10, fontweight='bold'
        )
        ax.legend(fontsize=8, loc='upper left', framealpha=0.95,
                  markerscale=0.5, handlelength=1.5)
        st.pyplot(fig)
        plt.close(fig)

# =========================================================
# TAB 3 — T(z) PLOT
# =========================================================

with tab3:
    st.header("T(z) Plot")

    if not results:
        st.error("No valid simulation results.")
    else:
        toggle_parts = []
        if enable_tpa: toggle_parts.append("TPA on")
        if enable_sa:  toggle_parts.append("SA on")
        toggle_label = ("  |  " + "  ".join(toggle_parts)) if toggle_parts else ""

        fig, ax = plt.subplots(figsize=(11, 6))

        for r in results:
            sign = '+' if r["n2"] > 0 else '-'
            det  = '  below noise' if not r["detectable"] else ''

            extra = []
            if enable_tpa and r["beta"] != 0.0:
                extra.append(f"β={r['beta']:.1e}")
            if enable_sa and r["alpha_0"] != 0.0:
                extra.append(f"α₀={r['alpha_0']:.1e}")
            extra_str = ("  " + "  ".join(extra)) if extra else ""

            lbl = (f"{r['name']}  n2={sign}{abs(r['n2']):.1e}"
                   f"{extra_str}  Tpv={r['tpv']:.4f}{det}")

            ls = '-'  if r["detectable"] else '--'
            lw = 2.2  if r["detectable"] else 1.5

            ax.plot(r["z"] / z0, r["Tz"],
                    linestyle=ls, linewidth=lw,
                    color=r["color"], label=lbl)

        ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0.0, color='gray', linestyle=':', alpha=0.4)
        ax.set_xlabel('z / z0', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalised T(z)', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Predicted T(z) — all materials{toggle_label}\n'
            'Solid = detectable  |  Dashed = below noise floor\n' + SETUP_STR,
            fontsize=10, fontweight='bold'
        )
        ax.legend(fontsize=8, loc='upper right', framealpha=0.95)
        st.pyplot(fig)
        plt.close(fig)

# =========================================================
# TAB 4 — RESULTS TABLE
# =========================================================

with tab4:
    st.header("Results Table")

    if not results:
        st.error("No valid results.")
    else:
        # Validity warnings
        for r in results:
            if r["dphi"] > 1.0:
                st.warning(
                    f"⚠ {r['name']}: ΔΦ₀ = {r['dphi']:.2f} rad > 1 — "
                    "outside parabolic approximation validity range."
                )
            if enable_tpa and r["q0"] > 1.0:
                st.warning(
                    f"⚠ {r['name']}: q0 = {r['q0']:.2f} > 1 — "
                    "TPA is in the strong absorption regime."
                )

        rows = []
        for r in results:
            row = {
                "Material":      r["name"],
                "n2 (m²/W)":    r["n2"],
                "ΔΦ₀ (rad)":    round(r["dphi"], 6),
                "β (m/W)":      r["beta"]    if enable_tpa else "—",
                "q0":           round(r["q0"], 4) if enable_tpa else "—",
                "α₀ (m⁻¹)":    r["alpha_0"] if enable_sa  else "—",
                "I_sat (W/m²)": r["I_sat"]   if enable_sa  else "—",
                "Tpv":          round(r["tpv"], 6),
                "Detectable":   "YES" if r["detectable"] else "NO",
            }
            rows.append(row)

        st.dataframe(pd.DataFrame(rows), use_container_width=True)