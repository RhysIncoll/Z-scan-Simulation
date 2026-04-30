"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, ReferenceLine,
} from "recharts";
import {
  Plus, Trash2, FlaskConical, Activity, Table2, BarChart3,
  ChevronDown, ChevronRight, Pencil, Check, X, Layers,
} from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Material {
  name: string; n2: number; n0: number;
  beta: number; alpha_0: number; I_sat: number;
  color: string; marker: string;
}

interface BeamConfig {
  lam: number; w0: number; L: number; S: number;
  P_avg: number; f_rep: number; tau_pulse: number;
  d_det: number; tpv_min: number;
}

interface SimResult {
  name: string; z: number[]; Tz: number[];
  n2: number; tpv: number; dphi: number; q0: number;
  detectable: boolean; color: string; error?: string;
}

interface ThinResult {
  name: string; z: number[]; Tz: number[];
  tpv: number; detectable: boolean; color: string;
  q0?: number; dphi?: number; ratio?: number; error?: string;
}

// ─── Defaults ─────────────────────────────────────────────────────────────────

const DEFAULT_MATERIALS: Material[] = [
  { name: "CS2 Reference",      n2: 3.2e-18,   n0: 1.63, beta: 0, alpha_0: 0,   I_sat: 1e20, color: "#e2e8f0", marker: "star"     },
  { name: "Graphene FLG",       n2: -5.01e-17,  n0: 1.45, beta: 0, alpha_0: 0,   I_sat: 1e20, color: "#fb923c", marker: "square"   },
  { name: "Wang 2014 Graphene", n2: -2.34e-16,  n0: 1.45, beta: 0, alpha_0: 0,   I_sat: 1e20, color: "#34d399", marker: "triangle" },
];

const DEFAULT_BEAM: BeamConfig = {
  lam: 800e-9, w0: 1e-6, L: 100e-6, S: 0.25,
  P_avg: 0.5, f_rep: 80e6, tau_pulse: 1e-15,
  d_det: 0.04, tpv_min: 0.01,
};

const COLORS = ["#e2e8f0","#fb923c","#34d399","#60a5fa","#f472b6","#a78bfa","#facc15","#f87171"];

// ─── Helpers ──────────────────────────────────────────────────────────────────

const fmtSci = (v: number) => v === 0 ? "0" : v.toExponential(2);

function SciField({ label, value, onChange, help }: {
  label: string; value: number; onChange: (v: number) => void; help?: string;
}) {
  const [raw, setRaw]         = useState(value.toExponential(2));
  const [focused, setFocused] = useState(false);
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-medium text-slate-400 uppercase tracking-wider">{label}</label>
      <input type="text"
        value={focused ? raw : value.toExponential(2)}
        onFocus={() => { setRaw(value.toExponential(2)); setFocused(true); }}
        onChange={e => setRaw(e.target.value)}
        onBlur={() => { setFocused(false); const p = parseFloat(raw); if (!isNaN(p)) onChange(p); }}
        className="bg-slate-800 border border-slate-700 rounded px-3 py-1.5 text-sm text-slate-100 focus:outline-none focus:border-cyan-500 font-mono"
      />
      {help && <span className="text-xs text-slate-600">{help}</span>}
    </div>
  );
}

function NumField({ label, value, onChange, help }: {
  label: string; value: number; onChange: (v: number) => void; help?: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs font-medium text-slate-400 uppercase tracking-wider">{label}</label>
      <input type="number" step="any" value={value}
        onChange={e => onChange(parseFloat(e.target.value) || 0)}
        className="bg-slate-800 border border-slate-700 rounded px-3 py-1.5 text-sm text-slate-100 focus:outline-none focus:border-cyan-500 font-mono"
      />
      {help && <span className="text-xs text-slate-600">{help}</span>}
    </div>
  );
}

function ReadOnlyField({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-xs text-slate-500">{label}</span>
      <span className={`text-xs font-mono ${highlight ? "text-cyan-400" : "text-slate-300"}`}>{value}</span>
    </div>
  );
}

function CollapsibleSection({ title, children, defaultOpen = true }: {
  title: string; children: React.ReactNode; defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div>
      <button onClick={() => setOpen(o => !o)} className="flex items-center gap-2 w-full text-left py-1.5 group">
        {open ? <ChevronDown size={12} className="text-cyan-400" /> : <ChevronRight size={12} className="text-slate-500" />}
        <span className="text-xs font-bold text-slate-500 uppercase tracking-widest group-hover:text-slate-300 transition-colors">{title}</span>
      </button>
      {open && <div className="flex flex-col gap-3 mt-2 pl-2">{children}</div>}
    </div>
  );
}

function LogSlider({ label, value, onChange, min, max, unit, color }: {
  label: string; value: number; onChange: (v: number) => void;
  min: number; max: number; unit: string; color: string;
}) {
  const logVal = value > 0 ? Math.log10(value) : min;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-400">{label}</span>
        <span className="text-xs font-mono font-bold" style={{ color }}>
          {value.toExponential(2)} <span className="text-slate-500 font-normal">{unit}</span>
        </span>
      </div>
      <input type="range" min={min} max={max} step={0.05} value={logVal}
        onChange={e => onChange(Math.pow(10, parseFloat(e.target.value)))}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer bg-slate-700"
        style={{ accentColor: color }}
      />
      <div className="flex justify-between text-xs text-slate-700">
        <span>10^{min}</span><span>10^{max}</span>
      </div>
    </div>
  );
}

interface ZoomState { left: number|"auto"; right: number|"auto"; top: number|"auto"; bottom: number|"auto"; }
const ZOOM_INITIAL: ZoomState = { left: "auto", right: "auto", top: "auto", bottom: "auto" };

function PlotHeader({ title, onReset }: { title: string; onReset: () => void }) {
  return (
    <div className="flex items-center justify-between mb-4">
      <h2 className="text-sm font-bold text-slate-400 uppercase tracking-widest">{title}</h2>
      <div className="flex items-center gap-3">
        <span className="text-xs text-slate-600">Click-drag to zoom</span>
        <button onClick={onReset}
          className="text-xs text-slate-500 hover:text-cyan-400 border border-slate-700 hover:border-cyan-600 px-3 py-1 rounded transition-colors">
          Reset zoom
        </button>
      </div>
    </div>
  );
}

function EmptyPlot() {
  return (
    <div className="flex items-center justify-center h-64 text-slate-600 text-sm">
      Press "Run Simulation" to generate plots
    </div>
  );
}

// ─── Main ─────────────────────────────────────────────────────────────────────

export default function ZScanDashboard() {
  const [beam, setBeam]           = useState<BeamConfig>(DEFAULT_BEAM);
  const [materials, setMaterials] = useState<Material[]>(DEFAULT_MATERIALS);
  const [useTPA, setUseTPA]       = useState(false);
  const [useSA,  setUseSA]        = useState(false);

  // Thick medium state
  const [results, setResults]         = useState<SimResult[]>([]);
  const [sensitivity, setSensitivity] = useState<{ n2: number; tpv: number }[]>([]);
  const [n2Dphi1, setN2Dphi1]         = useState<number | null>(null);
  const [loadingThick, setLoadingThick] = useState(false);

  // Thin sample state
  const [thinClosed, setThinClosed]     = useState<ThinResult[]>([]);
  const [thinOpenTPA, setThinOpenTPA]   = useState<ThinResult[]>([]);
  const [thinOpenSA,  setThinOpenSA]    = useState<ThinResult[]>([]);
  const [loadingThin, setLoadingThin]   = useState(false);

  // Thin sample slider overrides (per-material)
  const [thinBeta,    setThinBeta]    = useState<Record<string, number>>({});
  const [thinAlpha,   setThinAlpha]   = useState<Record<string, number>>({});
  const [thinISat,    setThinISat]    = useState<Record<string, number>>({});

  const [activeTab, setActiveTab] = useState<"tz"|"sensitivity"|"thin"|"table"|"materials">("tz");

  // Zoom
  const [tzZoom,   setTzZoom]   = useState<ZoomState>(ZOOM_INITIAL);
  const [sensZoom, setSensZoom] = useState<ZoomState>(ZOOM_INITIAL);
  const [thinZoom, setThinZoom] = useState<ZoomState>(ZOOM_INITIAL);
  const tzRef   = useRef<{x1:number|null;x2:number|null}>({x1:null,x2:null});
  const sensRef = useRef<{x1:number|null;x2:number|null}>({x1:null,x2:null});
  const thinRef = useRef<{x1:number|null;x2:number|null}>({x1:null,x2:null});
  const [tzSel,   setTzSel]   = useState(false);
  const [sensSel, setSensSel] = useState(false);
  const [thinSel, setThinSel] = useState(false);

  // Material editing
  const [editIdx, setEditIdx] = useState<number|null>(null);
  const [editBuf, setEditBuf] = useState<Material|null>(null);
  const [newMat, setNewMat]   = useState<Material>({
    name:"", n2:-1e-17, n0:1.45, beta:0, alpha_0:0, I_sat:1e13, color:"#60a5fa", marker:"circle",
  });

  const setBeamField = (k: keyof BeamConfig) => (v: number) => setBeam(b => ({...b,[k]:v}));

  // Derived
  const P_peak  = beam.P_avg / (beam.f_rep * beam.tau_pulse);
  const E_pulse = beam.P_avg / beam.f_rep;
  const z0      = Math.PI * beam.w0**2 / beam.lam;
  const I0      = 2 * P_peak / (Math.PI * beam.w0**2);
  const apiBeam = { lam:beam.lam, w0:beam.w0, L:beam.L, S:beam.S, P_peak, d_det:beam.d_det, tpv_min:beam.tpv_min };

  // Seed thin sliders when tabs first visited
  useEffect(() => {
    setThinBeta(prev  => { const n={...prev}; materials.forEach(m=>{ if(!n[m.name]) n[m.name]=m.beta||1e-11; }); return n; });
    setThinAlpha(prev => { const n={...prev}; materials.forEach(m=>{ if(!n[m.name]) n[m.name]=m.alpha_0||1e4; }); return n; });
    setThinISat(prev  => { const n={...prev}; materials.forEach(m=>{ if(!n[m.name]) n[m.name]=(m.I_sat<1e19?m.I_sat:1e12); }); return n; });
  }, [materials]);

  // ── Thick simulation ────────────────────────────────────────────────────────
  const runThick = useCallback(async () => {
    setLoadingThick(true);
    try {
      const [simRes, sensRes] = await Promise.all([
        axios.post(`${API}/simulate`, { beam: apiBeam, materials, use_tpa: useTPA, use_sa: useSA }),
        axios.post(`${API}/sensitivity`, apiBeam),
      ]);
      setResults(simRes.data.results.filter((r: SimResult) => !r.error));
      setSensitivity(sensRes.data.n2_sweep.map((n2: number, i: number) => ({ n2, tpv: sensRes.data.tpv_sweep[i] })));
      setN2Dphi1(sensRes.data.n2_dphi1);
      setTzZoom(ZOOM_INITIAL); setSensZoom(ZOOM_INITIAL);
    } catch(e) { console.error(e); }
    finally { setLoadingThick(false); }
  }, [beam, materials, useTPA, useSA]);

  // ── Thin simulation ─────────────────────────────────────────────────────────
  const runThin = useCallback(async () => {
    setLoadingThin(true);
    try {
      const matsWithOverrides = materials.map(m => ({
        ...m,
        beta:    thinBeta[m.name]  ?? m.beta,
        alpha_0: thinAlpha[m.name] ?? m.alpha_0,
        I_sat:   thinISat[m.name]  ?? m.I_sat,
      }));
      const base = { beam: apiBeam, materials: matsWithOverrides, use_tpa: useTPA, use_sa: useSA };
      const [closedRes, openTpaRes, openSaRes] = await Promise.all([
        axios.post(`${API}/simulate_thin_closed`,   base),
        axios.post(`${API}/simulate_thin_open_tpa`, base),
        axios.post(`${API}/simulate_thin_open_sa`,  base),
      ]);
      setThinClosed(closedRes.data.results.filter((r: ThinResult) => !r.error));
      setThinOpenTPA(openTpaRes.data.results.filter((r: ThinResult) => !r.error));
      setThinOpenSA(openSaRes.data.results.filter((r: ThinResult) => !r.error));
      setThinZoom(ZOOM_INITIAL);
    } catch(e) { console.error(e); }
    finally { setLoadingThin(false); }
  }, [beam, materials, useTPA, useSA, thinBeta, thinAlpha, thinISat]);

  // ── Zoom helpers ─────────────────────────────────────────────────────────────
  const makeZoomHandlers = (
    ref: React.MutableRefObject<{x1:number|null;x2:number|null}>,
    setSel: (b: boolean) => void,
    setZoom: (z: ZoomState) => void,
    selecting: boolean,
  ) => ({
    onMouseDown: (e: any) => { if(!e?.activeLabel) return; ref.current.x1=parseFloat(e.activeLabel); setSel(true); },
    onMouseMove: (e: any) => { if(!selecting||!e?.activeLabel) return; ref.current.x2=parseFloat(e.activeLabel); },
    onMouseUp:   () => {
      setSel(false);
      const {x1,x2}=ref.current;
      if(x1===null||x2===null||x1===x2) return;
      const [l,r]=x1<x2?[x1,x2]:[x2,x1];
      setZoom(z=>({...z,left:l,right:r}));
      ref.current={x1:null,x2:null};
    },
  });

  const tzHandlers   = makeZoomHandlers(tzRef,   setTzSel,   setTzZoom,   tzSel);
  const sensHandlers = makeZoomHandlers(sensRef,  setSensSel, setSensZoom, sensSel);
  const thinHandlers = makeZoomHandlers(thinRef,  setThinSel, setThinZoom, thinSel);

  const addMaterial  = () => {
    if(!newMat.name.trim()) return;
    setMaterials(m=>[...m,{...newMat}]);
    setNewMat({name:"",n2:-1e-17,n0:1.45,beta:0,alpha_0:0,I_sat:1e13,color:COLORS[materials.length%COLORS.length],marker:"circle"});
  };
  const removeMaterial = (i: number) => { setMaterials(m=>m.filter((_,j)=>j!==i)); if(editIdx===i) setEditIdx(null); };
  const startEdit  = (i: number) => { setEditIdx(i); setEditBuf({...materials[i]}); };
  const saveEdit   = () => { if(editIdx===null||!editBuf) return; setMaterials(m=>m.map((mat,i)=>i===editIdx?editBuf:mat)); setEditIdx(null); setEditBuf(null); };
  const cancelEdit = () => { setEditIdx(null); setEditBuf(null); };

  const tabs = [
    { id:"tz",          label:"T(z) Thick",      icon: Activity     },
    { id:"sensitivity", label:"Sensitivity",      icon: BarChart3    },
    { id:"thin",        label:"Thin Sample",      icon: Layers       },
    { id:"table",       label:"Results",          icon: Table2       },
    { id:"materials",   label:"Materials",        icon: FlaskConical },
  ] as const;

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-mono flex flex-col">

      {/* Header */}
      <header className="border-b border-slate-800 px-6 py-3 flex items-center gap-4">
        <div className="w-2 h-7 bg-cyan-400 rounded-sm" />
        <div>
          <h1 className="text-base font-bold tracking-tight text-white">Z-scan Nonlinear Optics Predictor</h1>
          <p className="text-xs text-slate-500">Thick medium (Sheik-Bahae 1991) · Thin sample GD (Sheik-Bahae 1990) · 2D Materials</p>
        </div>
        <div className="ml-auto flex items-center gap-4">
          <label className="flex items-center gap-2 text-xs text-slate-400 cursor-pointer select-none">
            <input type="checkbox" checked={useTPA} onChange={e=>setUseTPA(e.target.checked)} className="accent-cyan-400" /> TPA (β)
          </label>
          <label className="flex items-center gap-2 text-xs text-slate-400 cursor-pointer select-none">
            <input type="checkbox" checked={useSA} onChange={e=>setUseSA(e.target.checked)} className="accent-cyan-400" /> SA (α₀)
          </label>
          <button onClick={runThick} disabled={loadingThick}
            className="bg-cyan-500 hover:bg-cyan-400 disabled:opacity-40 text-slate-950 font-bold text-sm px-4 py-2 rounded transition-colors">
            {loadingThick ? "Running…" : "Run Thick"}
          </button>
          <button onClick={runThin} disabled={loadingThin}
            className="bg-violet-500 hover:bg-violet-400 disabled:opacity-40 text-white font-bold text-sm px-4 py-2 rounded transition-colors">
            {loadingThin ? "Running…" : "Run Thin"}
          </button>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">

        {/* Sidebar */}
        <aside className="w-64 border-r border-slate-800 p-4 flex flex-col gap-4 overflow-y-auto shrink-0">

          <CollapsibleSection title="Laser Parameters">
            <SciField label="Wavelength λ (m)"     value={beam.lam}       onChange={setBeamField("lam")}       help="e.g. 800e-9" />
            <SciField label="Avg power P_avg (W)"  value={beam.P_avg}     onChange={setBeamField("P_avg")}     help="Power meter reading" />
            <SciField label="Rep rate f_rep (Hz)"  value={beam.f_rep}     onChange={setBeamField("f_rep")}     help="e.g. 80e6 for 80 MHz" />
            <SciField label="Pulse duration τ (s)" value={beam.tau_pulse} onChange={setBeamField("tau_pulse")} help="e.g. 100e-15 for 100 fs" />
          </CollapsibleSection>

          <CollapsibleSection title="Beam & Sample">
            <SciField label="Beam waist w₀ (m)"     value={beam.w0} onChange={setBeamField("w0")} help="At focus" />
            <SciField label="Sample thickness L (m)" value={beam.L}  onChange={setBeamField("L")} />
            <NumField label="Aperture S"              value={beam.S}  onChange={setBeamField("S")} help="Linear transmittance 0–1" />
          </CollapsibleSection>

          <CollapsibleSection title="Setup" defaultOpen={false}>
            <SciField label="Detector dist (m)"   value={beam.d_det}   onChange={setBeamField("d_det")} />
            <NumField label="Noise floor Tpv_min" value={beam.tpv_min} onChange={setBeamField("tpv_min")} />
          </CollapsibleSection>

          <CollapsibleSection title="Derived" defaultOpen={true}>
            <ReadOnlyField label="P_peak"  value={`${P_peak.toExponential(2)} W`} />
            <ReadOnlyField label="E_pulse" value={`${E_pulse.toExponential(2)} J`} />
            <ReadOnlyField label="z₀"      value={`${z0.toExponential(2)} m`} />
            <ReadOnlyField label="I₀"      value={`${I0.toExponential(2)} W/m²`} />
            <ReadOnlyField label="L / z₀"  value={`${(beam.L/z0).toFixed(2)}${beam.L/z0>1?" (thick)":" (thin)"}`} highlight={beam.L/z0>1} />
          </CollapsibleSection>

          <CollapsibleSection title="Materials" defaultOpen={true}>
            {materials.map((m,i)=>(
              <div key={i} className="flex items-center gap-2 group">
                <div className="w-2 h-2 rounded-full shrink-0" style={{background:m.color}} />
                <span className="text-xs text-slate-300 truncate flex-1">{m.name}</span>
                <button onClick={()=>startEdit(i)} className="opacity-0 group-hover:opacity-100 text-slate-600 hover:text-cyan-400 transition-all"><Pencil size={11}/></button>
                <button onClick={()=>removeMaterial(i)} className="opacity-0 group-hover:opacity-100 text-slate-600 hover:text-red-400 transition-all"><Trash2 size={11}/></button>
              </div>
            ))}
          </CollapsibleSection>

        </aside>

        {/* Main */}
        <main className="flex-1 flex flex-col overflow-hidden">

          {/* Tabs */}
          <div className="border-b border-slate-800 flex overflow-x-auto">
            {tabs.map(({id,label,icon:Icon})=>(
              <button key={id} onClick={()=>setActiveTab(id)}
                className={`flex items-center gap-2 px-5 py-3 text-sm transition-colors border-b-2 whitespace-nowrap ${
                  activeTab===id ? "border-cyan-400 text-cyan-400" : "border-transparent text-slate-500 hover:text-slate-300"
                }`}>
                <Icon size={14}/>{label}
              </button>
            ))}
          </div>

          <div className="flex-1 overflow-auto p-6">

            {/* ── Thick T(z) ── */}
            {activeTab==="tz" && (
              <div>
                <PlotHeader title="Normalised T(z) — Thick Medium Closed Aperture" onReset={()=>setTzZoom(ZOOM_INITIAL)} />
                {results.length===0 ? <EmptyPlot/> : (
                  <ResponsiveContainer width="100%" height={440}>
                    <LineChart
  data={results.flatMap(r =>
    r.z.map((z, i) => ({
      z,
      [r.name]: r.Tz[i]
    }))
  )}
  margin={{ top: 10, right: 30, left: 10, bottom: 30 }}
  {...tzHandlers}
>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                      <XAxis dataKey="z" type="number" domain={[tzZoom.left,tzZoom.right]} allowDataOverflow
                        label={{value:"z / z₀",position:"insideBottom",offset:-15,fill:"#94a3b8",fontSize:12}}
                        stroke="#334155" tick={{fill:"#64748b",fontSize:11}} tickFormatter={v=>Number(v).toFixed(1)}/>
                      <YAxis domain={[tzZoom.bottom,tzZoom.top]} allowDataOverflow stroke="#334155" tick={{fill:"#64748b",fontSize:11}}
                        label={{value:"Normalised T(z)",angle:-90,position:"insideLeft",offset:15,fill:"#94a3b8",fontSize:12}}
                        tickFormatter={v=>Number(v).toFixed(3)}/>
                      <Tooltip contentStyle={{background:"#0f172a",border:"1px solid #334155",fontSize:11,fontFamily:"monospace"}}
                        labelFormatter={v=>`z/z₀ = ${Number(v).toFixed(3)}`}
                        formatter={(v:number,n:string)=>[v.toFixed(6),n]}/>
                      <Legend wrapperStyle={{fontSize:11,color:"#94a3b8",paddingTop:8}}/>
                      <ReferenceLine y={1} stroke="#334155" strokeDasharray="4 4"/>
                      <ReferenceLine x={0} stroke="#334155" strokeDasharray="4 4"/>
                      {results.map(r=>(
                        <Line key={r.name} data={r.z.map((z,i)=>({z,Tz:r.Tz[i]}))} dataKey="Tz"
                          name={`${r.name} | Tpv=${r.tpv.toFixed(4)}`} stroke={r.color}
                          strokeWidth={r.detectable?2.2:1.5} strokeDasharray={r.detectable?undefined:"5 5"}
                          dot={false} isAnimationActive={false}/>
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            )}

            {/* ── Sensitivity ── */}
            {activeTab==="sensitivity" && (
              <div>
                <PlotHeader title="Sensitivity / Literature Resolvability" onReset={()=>setSensZoom(ZOOM_INITIAL)}/>
                {sensitivity.length===0 ? <EmptyPlot/> : (
                  <ResponsiveContainer width="100%" height={440}>
                    <LineChart data={sensitivity} margin={{top:10,right:30,left:10,bottom:50}} {...sensHandlers}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                      <XAxis dataKey="n2" type="number" scale="log" domain={[sensZoom.left,sensZoom.right]} allowDataOverflow
                        tickFormatter={v=>v.toExponential(0)} interval="preserveStartEnd"
                        label={{value:"|n₂| (m²/W)",position:"insideBottom",offset:-35,fill:"#94a3b8",fontSize:12}}
                        stroke="#334155" tick={{fill:"#64748b",fontSize:10,angle:-45,textAnchor:"end"}} height={60}/>
                      <YAxis scale="log" domain={[sensZoom.bottom,sensZoom.top]} allowDataOverflow
                        tickFormatter={v=>v.toExponential(0)}
                        label={{value:"Predicted Tpv",angle:-90,position:"insideLeft",offset:15,fill:"#94a3b8",fontSize:12}}
                        stroke="#334155" tick={{fill:"#64748b",fontSize:10}}/>
                      <Tooltip contentStyle={{background:"#0f172a",border:"1px solid #334155",fontSize:11,fontFamily:"monospace"}}
                        formatter={(v:number)=>[v.toExponential(3),"Tpv"]}
                        labelFormatter={v=>`|n₂| = ${Number(v).toExponential(2)} m²/W`}/>
                      <Legend wrapperStyle={{fontSize:11,color:"#94a3b8",paddingTop:8}}/>
                      <ReferenceLine y={beam.tpv_min} stroke="#f87171" strokeDasharray="4 4"
                        label={{value:"Noise floor",fill:"#f87171",fontSize:10,position:"right"}}/>
                      {n2Dphi1&&<ReferenceLine x={n2Dphi1} stroke="#64748b" strokeDasharray="4 4"
                        label={{value:"ΔΦ₀=1",fill:"#64748b",fontSize:10,position:"top"}}/>}
                      <Line dataKey="tpv" stroke="#22d3ee" strokeWidth={2} dot={false} name="Sensitivity curve" isAnimationActive={false}/>
                      {results.map(r=>(
                        <ReferenceLine key={r.name} x={Math.abs(r.n2)} stroke={r.color} strokeOpacity={0.7}
                          label={{value:r.name,fill:r.color,fontSize:9,position:"insideTopRight"}}/>
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            )}

            {/* ── Thin Sample ── */}
            {activeTab==="thin" && (
              <div className="flex flex-col gap-8">

                <div className="flex items-start justify-between">
                  <div>
                    <h2 className="text-sm font-bold text-slate-300 uppercase tracking-widest">Thin Sample Approximation</h2>
                    <p className="text-xs text-slate-500 mt-1">
                      Gaussian decomposition method — Sheik-Bahae 1990. Valid when L &lt; z₀ ({(beam.L/z0).toFixed(2)} currently).
                      {beam.L/z0>1 && <span className="text-amber-400"> ⚠ Your sample is thick — use thick model for quantitative results.</span>}
                    </p>
                  </div>
                  <button onClick={runThin} disabled={loadingThin}
                    className="bg-violet-500 hover:bg-violet-400 disabled:opacity-40 text-white font-bold text-sm px-4 py-2 rounded transition-colors shrink-0">
                    {loadingThin ? "Running…" : "Run Thin"}
                  </button>
                </div>

                {/* Per-material sliders */}
                <div className="border border-slate-800 rounded-xl bg-slate-900/50 p-5 flex flex-col gap-5">
                  <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Absorption Parameters</p>
                  <div className="grid gap-4" style={{gridTemplateColumns:`repeat(${Math.min(materials.length,3)},1fr)`}}>
                    {materials.map(m=>(
                      <div key={m.name} className="border border-slate-800 rounded-lg p-4 flex flex-col gap-4 bg-slate-900">
                        <div className="flex items-center gap-2">
                          <div className="w-2.5 h-2.5 rounded-full" style={{background:m.color}}/>
                          <span className="text-xs font-bold text-slate-200 truncate">{m.name}</span>
                        </div>
                        {useTPA && (
                          <LogSlider label="β (TPA)" value={thinBeta[m.name]??1e-11}
                            onChange={v=>setThinBeta(p=>({...p,[m.name]:v}))}
                            min={-14} max={-7} unit="m/W" color={m.color}/>
                        )}
                        {useSA && (
                          <>
                            <LogSlider label="α₀ (SA)" value={thinAlpha[m.name]??1e4}
                              onChange={v=>setThinAlpha(p=>({...p,[m.name]:v}))}
                              min={0} max={8} unit="m⁻¹" color={m.color}/>
                            <LogSlider label="I_sat" value={thinISat[m.name]??1e12}
                              onChange={v=>setThinISat(p=>({...p,[m.name]:v}))}
                              min={8} max={16} unit="W/m²" color={m.color}/>
                          </>
                        )}
                        {!useTPA && !useSA && (
                          <p className="text-xs text-slate-600">Enable TPA or SA above to show sliders.</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {thinClosed.length===0 ? <EmptyPlot/> : (
                  <>
                    {/* Closed aperture */}
                    <div>
                      <PlotHeader title="Closed Aperture — Thin Sample GD" onReset={()=>setThinZoom(ZOOM_INITIAL)}/>
                      <ResponsiveContainer width="100%" height={360}>
                        <LineChart margin={{top:10,right:30,left:10,bottom:30}} {...thinHandlers}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                          <XAxis dataKey="z" type="number" domain={[thinZoom.left,thinZoom.right]} allowDataOverflow
                            label={{value:"z / z₀",position:"insideBottom",offset:-15,fill:"#94a3b8",fontSize:12}}
                            stroke="#334155" tick={{fill:"#64748b",fontSize:11}} tickFormatter={v=>Number(v).toFixed(1)}/>
                          <YAxis domain={[thinZoom.bottom,thinZoom.top]} allowDataOverflow stroke="#334155" tick={{fill:"#64748b",fontSize:11}}
                            label={{value:"Normalised T(z)",angle:-90,position:"insideLeft",offset:15,fill:"#94a3b8",fontSize:12}}
                            tickFormatter={v=>Number(v).toFixed(3)}/>
                          <Tooltip contentStyle={{background:"#0f172a",border:"1px solid #334155",fontSize:11,fontFamily:"monospace"}}
                            labelFormatter={v=>`z/z₀ = ${Number(v).toFixed(3)}`}
                            formatter={(v:number,n:string)=>[v.toFixed(6),n]}/>
                          <Legend wrapperStyle={{fontSize:11,color:"#94a3b8",paddingTop:8}}/>
                          <ReferenceLine y={1} stroke="#334155" strokeDasharray="4 4"/>
                          <ReferenceLine x={0} stroke="#334155" strokeDasharray="4 4"/>
                          {thinClosed.map(r=>(
                            <Line key={r.name} data={r.z.map((z,i)=>({z,Tz:r.Tz[i]}))} dataKey="Tz"
                              name={`${r.name} | Tpv=${r.tpv.toFixed(4)}`} stroke={r.color}
                              strokeWidth={r.detectable?2.2:1.5} strokeDasharray={r.detectable?undefined:"5 5"}
                              dot={false} isAnimationActive={false}/>
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Open aperture TPA */}
                    {useTPA && thinOpenTPA.length>0 && (
                      <div>
                        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">
                          Open Aperture — TPA (Eq. 30, symmetric dip at focus)
                        </h3>
                        <ResponsiveContainer width="100%" height={300}>
                          <LineChart margin={{top:10,right:30,left:10,bottom:30}}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                            <XAxis dataKey="z" type="number" domain={["auto","auto"]}
                              label={{value:"z / z₀",position:"insideBottom",offset:-15,fill:"#94a3b8",fontSize:12}}
                              stroke="#334155" tick={{fill:"#64748b",fontSize:11}} tickFormatter={v=>Number(v).toFixed(1)}/>
                            <YAxis stroke="#334155" tick={{fill:"#64748b",fontSize:11}}
                              label={{value:"Normalised T(z)",angle:-90,position:"insideLeft",offset:15,fill:"#94a3b8",fontSize:12}}
                              tickFormatter={v=>Number(v).toFixed(3)}/>
                            <Tooltip contentStyle={{background:"#0f172a",border:"1px solid #334155",fontSize:11,fontFamily:"monospace"}}
                              labelFormatter={v=>`z/z₀ = ${Number(v).toFixed(3)}`}
                              formatter={(v:number,n:string)=>[v.toFixed(6),n]}/>
                            <Legend wrapperStyle={{fontSize:11,color:"#94a3b8",paddingTop:8}}/>
                            <ReferenceLine y={1} stroke="#334155" strokeDasharray="4 4"/>
                            <ReferenceLine x={0} stroke="#334155" strokeDasharray="4 4"/>
                            {thinOpenTPA.map(r=>(
                              <Line key={r.name} data={r.z.map((z,i)=>({z,Tz:r.Tz[i]}))} dataKey="Tz"
                                name={`${r.name} | q₀=${r.q0?.toFixed(3)}`} stroke={r.color}
                                strokeWidth={2} dot={false} isAnimationActive={false}/>
                            ))}
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    )}

                    {/* Open aperture SA */}
                    {useSA && thinOpenSA.length>0 && (
                      <div>
                        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">
                          Open Aperture — SA (symmetric peak at focus — bleaching)
                        </h3>
                        <ResponsiveContainer width="100%" height={300}>
                          <LineChart margin={{top:10,right:30,left:10,bottom:30}}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b"/>
                            <XAxis dataKey="z" type="number" domain={["auto","auto"]}
                              label={{value:"z / z₀",position:"insideBottom",offset:-15,fill:"#94a3b8",fontSize:12}}
                              stroke="#334155" tick={{fill:"#64748b",fontSize:11}} tickFormatter={v=>Number(v).toFixed(1)}/>
                            <YAxis stroke="#334155" tick={{fill:"#64748b",fontSize:11}}
                              label={{value:"Normalised T(z)",angle:-90,position:"insideLeft",offset:15,fill:"#94a3b8",fontSize:12}}
                              tickFormatter={v=>Number(v).toFixed(3)}/>
                            <Tooltip contentStyle={{background:"#0f172a",border:"1px solid #334155",fontSize:11,fontFamily:"monospace"}}
                              labelFormatter={v=>`z/z₀ = ${Number(v).toFixed(3)}`}
                              formatter={(v:number,n:string)=>[v.toFixed(6),n]}/>
                            <Legend wrapperStyle={{fontSize:11,color:"#94a3b8",paddingTop:8}}/>
                            <ReferenceLine y={1} stroke="#334155" strokeDasharray="4 4"/>
                            <ReferenceLine x={0} stroke="#334155" strokeDasharray="4 4"/>
                            {thinOpenSA.map(r=>(
                              <Line key={r.name} data={r.z.map((z,i)=>({z,Tz:r.Tz[i]}))} dataKey="Tz"
                                name={`${r.name} | I₀/I_sat=${r.ratio?.toFixed(2)}`} stroke={r.color}
                                strokeWidth={2} dot={false} isAnimationActive={false}/>
                            ))}
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}

            {/* ── Results Table ── */}
            {activeTab==="table" && (
              <div>
                <h2 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">Results — Thick Medium</h2>
                {results.length===0 ? <EmptyPlot/> : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-slate-800 text-left">
                          {["Material","n₂ (m²/W)","ΔΦ₀ (rad)","β (m/W)","q₀","α₀ (m⁻¹)","Tpv","Detectable"].map(h=>(
                            <th key={h} className="pb-3 pr-6 text-xs text-slate-500 uppercase tracking-wider font-medium whitespace-nowrap">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {results.map((r,i)=>{
                          const mat=materials.find(m=>m.name===r.name);
                          return (
                            <tr key={i} className="border-b border-slate-900 hover:bg-slate-900 transition-colors">
                              <td className="py-3 pr-6">
                                <div className="flex items-center gap-2">
                                  <div className="w-2 h-2 rounded-full shrink-0" style={{background:r.color}}/>{r.name}
                                </div>
                              </td>
                              <td className="py-3 pr-6 text-slate-400 font-mono">{fmtSci(r.n2)}</td>
                              <td className="py-3 pr-6 text-slate-400 font-mono">{r.dphi.toFixed(4)}</td>
                              <td className="py-3 pr-6 text-slate-400 font-mono">{mat?fmtSci(mat.beta):"—"}</td>
                              <td className="py-3 pr-6 text-slate-400 font-mono">{r.q0.toFixed(4)}</td>
                              <td className="py-3 pr-6 text-slate-400 font-mono">{mat?fmtSci(mat.alpha_0):"—"}</td>
                              <td className="py-3 pr-6 text-slate-400 font-mono">{r.tpv.toFixed(6)}</td>
                              <td className="py-3 pr-6">
                                <span className={`px-2 py-0.5 rounded text-xs font-bold ${r.detectable?"bg-emerald-900 text-emerald-300":"bg-slate-800 text-slate-500"}`}>
                                  {r.detectable?"YES":"NO"}
                                </span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}

            {/* ── Material Manager ── */}
            {activeTab==="materials" && (
              <div className="max-w-2xl flex flex-col gap-8">
                <div>
                  <h2 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">Add New Material</h2>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="col-span-2 flex flex-col gap-1">
                      <label className="text-xs font-medium text-slate-400 uppercase tracking-wider">Name</label>
                      <input type="text" placeholder="e.g. MoS2 monolayer" value={newMat.name}
                        onChange={e=>setNewMat(m=>({...m,name:e.target.value}))}
                        className="bg-slate-800 border border-slate-700 rounded px-3 py-1.5 text-sm text-slate-100 focus:outline-none focus:border-cyan-500 font-mono"/>
                    </div>
                    <SciField label="n₂ (m²/W)"   value={newMat.n2}      onChange={v=>setNewMat(m=>({...m,n2:v}))}/>
                    <NumField label="n₀"           value={newMat.n0}      onChange={v=>setNewMat(m=>({...m,n0:v}))}/>
                    <SciField label="β (m/W)"      value={newMat.beta}    onChange={v=>setNewMat(m=>({...m,beta:v}))}      help="TPA — 0 if unknown"/>
                    <SciField label="α₀ (m⁻¹)"    value={newMat.alpha_0} onChange={v=>setNewMat(m=>({...m,alpha_0:v}))}   help="SA — 0 if unknown"/>
                    <SciField label="I_sat (W/m²)" value={newMat.I_sat}   onChange={v=>setNewMat(m=>({...m,I_sat:v}))}    help="SA saturation intensity"/>
                    <div className="flex flex-col gap-1">
                      <label className="text-xs font-medium text-slate-400 uppercase tracking-wider">Color</label>
                      <input type="color" value={newMat.color} onChange={e=>setNewMat(m=>({...m,color:e.target.value}))}
                        className="h-9 w-full rounded cursor-pointer bg-slate-800 border border-slate-700"/>
                    </div>
                  </div>
                  <button onClick={addMaterial}
                    className="mt-4 flex items-center gap-2 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-sm px-4 py-2 rounded transition-colors">
                    <Plus size={14}/> Add Material
                  </button>
                </div>

                <div>
                  <h2 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-4">Current Materials</h2>
                  <div className="flex flex-col gap-3">
                    {materials.map((m,i)=>(
                      <div key={i} className="border border-slate-800 rounded-lg overflow-hidden">
                        <div className="flex items-center gap-3 px-4 py-3 bg-slate-900">
                          <div className="w-3 h-3 rounded-full shrink-0" style={{background:m.color}}/>
                          <div className="flex-1">
                            <p className="text-sm text-slate-200 font-medium">{m.name}</p>
                            <p className="text-xs text-slate-500 font-mono">
                              n₂ = {fmtSci(m.n2)} · n₀ = {m.n0}
                              {m.beta!==0&&` · β = ${fmtSci(m.beta)}`}
                              {m.alpha_0!==0&&` · α₀ = ${fmtSci(m.alpha_0)}`}
                            </p>
                          </div>
                          <div className="flex items-center gap-2">
                            {editIdx===i?(
                              <>
                                <button onClick={saveEdit}   className="text-emerald-400 hover:text-emerald-300"><Check  size={14}/></button>
                                <button onClick={cancelEdit} className="text-slate-500   hover:text-slate-300"><X      size={14}/></button>
                              </>
                            ):(
                              <button onClick={()=>startEdit(i)} className="text-slate-500 hover:text-cyan-400"><Pencil size={14}/></button>
                            )}
                            <button onClick={()=>removeMaterial(i)} className="text-slate-500 hover:text-red-400"><Trash2 size={14}/></button>
                          </div>
                        </div>
                        {editIdx===i&&editBuf&&(
                          <div className="grid grid-cols-2 gap-4 p-4 bg-slate-950 border-t border-slate-800">
                            <div className="col-span-2 flex flex-col gap-1">
                              <label className="text-xs font-medium text-slate-400 uppercase tracking-wider">Name</label>
                              <input type="text" value={editBuf.name}
                                onChange={e=>setEditBuf(b=>b?{...b,name:e.target.value}:b)}
                                className="bg-slate-800 border border-slate-700 rounded px-3 py-1.5 text-sm text-slate-100 focus:outline-none focus:border-cyan-500 font-mono"/>
                            </div>
                            <SciField label="n₂ (m²/W)"   value={editBuf.n2}      onChange={v=>setEditBuf(b=>b?{...b,n2:v}:b)}/>
                            <NumField label="n₀"           value={editBuf.n0}      onChange={v=>setEditBuf(b=>b?{...b,n0:v}:b)}/>
                            <SciField label="β (m/W)"      value={editBuf.beta}    onChange={v=>setEditBuf(b=>b?{...b,beta:v}:b)}     help="TPA — 0 if unknown"/>
                            <SciField label="α₀ (m⁻¹)"    value={editBuf.alpha_0} onChange={v=>setEditBuf(b=>b?{...b,alpha_0:v}:b)}  help="SA — 0 if unknown"/>
                            <SciField label="I_sat (W/m²)" value={editBuf.I_sat}   onChange={v=>setEditBuf(b=>b?{...b,I_sat:v}:b)}/>
                            <div className="flex flex-col gap-1">
                              <label className="text-xs font-medium text-slate-400 uppercase tracking-wider">Color</label>
                              <input type="color" value={editBuf.color}
                                onChange={e=>setEditBuf(b=>b?{...b,color:e.target.value}:b)}
                                className="h-9 w-full rounded cursor-pointer bg-slate-800 border border-slate-700"/>
                            </div>
                            <div className="col-span-2 flex justify-end gap-2 pt-2">
                              <button onClick={cancelEdit}
                                className="text-xs text-slate-500 hover:text-slate-300 border border-slate-700 px-3 py-1.5 rounded">Cancel</button>
                              <button onClick={saveEdit}
                                className="text-xs bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold px-4 py-1.5 rounded">Save changes</button>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

          </div>
        </main>
      </div>
    </div>
  );
}