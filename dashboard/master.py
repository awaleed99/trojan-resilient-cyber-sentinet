"""
dashboard/master.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trojan-Resilient Cyber-Sentinet — Master Control Dashboard

Run:  streamlit run dashboard/master.py   (from project root)
"""

import os, sys, json, pickle, subprocess, threading, time, csv, textwrap
from pathlib import Path
from datetime import datetime
from collections import deque

import streamlit as st
import numpy as np

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cyber-Sentinet | Control Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# PREMIUM CSS — dark cybersecurity theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ─── Reset & base ─── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp {
  font-family: 'Inter', -apple-system, sans-serif !important;
  background: #070b14 !important;
  color: #dde4f0 !important;
}
.main .block-container {
  padding: 1.5rem 2rem !important;
  max-width: 1500px !important;
  background: #070b14 !important;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #060a13 0%, #0a1020 100%) !important;
  border-right: 1px solid rgba(0,200,255,0.12) !important;
}
[data-testid="stSidebar"] * { color: #dde4f0 !important; }
[data-testid="stSidebar"] .stRadio > label { display: none !important; }
[data-testid="stSidebar"] .stRadio > div {
  display: flex !important;
  flex-direction: column !important;
  gap: 4px !important;
}
[data-testid="stSidebar"] .stRadio > div > label {
  display: block !important;
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(0,200,255,0.08) !important;
  border-radius: 10px !important;
  padding: 10px 14px !important;
  cursor: pointer !important;
  font-size: 0.88rem !important;
  font-weight: 500 !important;
  transition: all 0.2s !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
  background: rgba(0,200,255,0.07) !important;
  border-color: rgba(0,200,255,0.25) !important;
}

/* ─── Hero header ─── */
.hero {
  position: relative;
  background: linear-gradient(135deg, #0d1e3d 0%, #0a1528 60%, #0f1e3a 100%);
  border: 1px solid rgba(0,200,255,0.18);
  border-radius: 20px;
  padding: 2.2rem 2.8rem;
  margin-bottom: 1.8rem;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent 0%, #00c8ff 30%, #8b5cf6 70%, transparent 100%);
}
.hero::after {
  content: '';
  position: absolute; bottom: -60px; right: -60px;
  width: 200px; height: 200px;
  background: radial-gradient(circle, rgba(0,200,255,0.06) 0%, transparent 70%);
  border-radius: 50%;
}
.hero h1 {
  margin: 0 0 0.4rem;
  font-size: 2.1rem;
  font-weight: 800;
  background: linear-gradient(135deg, #00c8ff 0%, #8b5cf6 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero p { margin: 0; color: #6b7fa3; font-size: 0.93rem; line-height: 1.6; }
.hero .tag {
  display: inline-block;
  margin-top: 0.8rem;
  padding: 3px 12px;
  border-radius: 20px;
  background: rgba(0,200,255,0.1);
  border: 1px solid rgba(0,200,255,0.25);
  color: #00c8ff;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

/* ─── Metric cards ─── */
.metrics-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 1.8rem; }
.mc {
  background: linear-gradient(135deg, #0d1830 0%, #111c38 100%);
  border: 1px solid rgba(0,200,255,0.12);
  border-radius: 16px;
  padding: 1.4rem 1.2rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: transform 0.2s, border-color 0.2s;
}
.mc:hover { transform: translateY(-3px); border-color: rgba(0,200,255,0.3); }
.mc .val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 2.2rem;
  font-weight: 700;
  line-height: 1;
  margin-bottom: 0.4rem;
}
.mc .lbl { font-size: 0.7rem; color: #5a7090; text-transform: uppercase; letter-spacing: 0.08em; }
.mc.c1 .val { color: #00c8ff; } .mc.c1::before { background: rgba(0,200,255,0.05); }
.mc.c2 .val { color: #00e896; }
.mc.c3 .val { color: #ff7c44; }
.mc.c4 .val { color: #a78bfa; }
.mc::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(255,255,255,0.02); border-radius: 16px;
}

/* ─── Section headers ─── */
.sec-hdr {
  display: flex; align-items: center; gap: 10px;
  font-size: 0.82rem; font-weight: 700;
  color: #00c8ff;
  text-transform: uppercase; letter-spacing: 0.1em;
  border-bottom: 1px solid rgba(0,200,255,0.1);
  padding-bottom: 8px; margin-bottom: 16px;
}
.sec-hdr span { width: 6px; height: 6px; background: #00c8ff; border-radius: 50%; display: inline-block; }

/* ─── Pipeline stage cards ─── */
.stage {
  background: #0d1830;
  border: 1px solid rgba(0,200,255,0.1);
  border-radius: 14px;
  padding: 1rem 1.3rem;
  margin-bottom: 10px;
  transition: border-color 0.25s, box-shadow 0.25s;
}
.stage:hover { border-color: rgba(0,200,255,0.28); }
.stage.s-done { border-color: rgba(0,232,150,0.3); }
.stage.s-running { border-color: #00c8ff; box-shadow: 0 0 16px rgba(0,200,255,0.15); }
.stage.s-error { border-color: rgba(255,70,80,0.4); }
.stage-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
.stage-name { font-size: 0.93rem; font-weight: 600; }
.stage-desc { font-size: 0.78rem; color: #5a7090; }

/* ─── Badges ─── */
.badge {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 3px 10px; border-radius: 20px;
  font-size: 0.68rem; font-weight: 700;
  letter-spacing: 0.04em; text-transform: uppercase;
}
.bd { background: rgba(90,112,144,0.18); color: #6b7fa3; border: 1px solid rgba(90,112,144,0.25); }
.br { background: rgba(0,200,255,0.12); color: #00c8ff; border: 1px solid rgba(0,200,255,0.3); }
.bo { background: rgba(0,232,150,0.12); color: #00e896; border: 1px solid rgba(0,232,150,0.3); }
.be { background: rgba(255,70,80,0.12); color: #ff4650; border: 1px solid rgba(255,70,80,0.3); }

/* ─── Terminal ─── */
.terminal-wrap {
  background: #040810;
  border: 1px solid rgba(0,200,255,0.15);
  border-radius: 14px;
  overflow: hidden;
}
.terminal-bar {
  background: #0a1020;
  padding: 8px 16px;
  display: flex; align-items: center; gap: 6px;
  border-bottom: 1px solid rgba(0,200,255,0.08);
}
.terminal-bar .dot { width: 10px; height: 10px; border-radius: 50%; }
.terminal-bar .d1 { background: #ff5f57; }
.terminal-bar .d2 { background: #febc2e; }
.terminal-bar .d3 { background: #28c840; }
.terminal-bar .title { color: #3a5070; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; margin-left: 8px; }
.terminal-body {
  padding: 14px 18px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.76rem;
  line-height: 1.75;
  height: 380px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-all;
}
.t-ok   { color: #00e896; }
.t-info { color: #7fb9d0; }
.t-err  { color: #ff6b70; }
.t-ts   { color: #2a3a50; }
.t-dim  { color: #3a5070; }
.cursor { display: inline-block; width: 7px; height: 14px; background: #00c8ff; animation: blink 1s infinite; vertical-align: middle; }
@keyframes blink { 0%,49%{opacity:1} 50%,100%{opacity:0} }

/* ─── Buttons ─── */
.stButton > button {
  background: linear-gradient(135deg, #0e2040 0%, #102848 100%) !important;
  color: #00c8ff !important;
  border: 1px solid rgba(0,200,255,0.3) !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-size: 0.83rem !important;
  padding: 0.45rem 1rem !important;
  transition: all 0.2s !important;
  font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover:not(:disabled) {
  background: linear-gradient(135deg, #122848 0%, #14325a 100%) !important;
  box-shadow: 0 0 18px rgba(0,200,255,0.2) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:disabled { opacity: 0.4 !important; cursor: not-allowed !important; }
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #0e3560 0%, #0e4880 100%) !important;
  color: #fff !important;
  border-color: rgba(0,200,255,0.5) !important;
  font-size: 0.9rem !important;
  padding: 0.55rem 1.2rem !important;
}

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
  background: #0d1830 !important;
  border: 1px solid rgba(0,200,255,0.12) !important;
  border-radius: 12px !important; padding: 5px !important; gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important; color: #6b7fa3 !important;
  border-radius: 9px !important; font-weight: 500 !important;
  font-size: 0.85rem !important; padding: 7px 18px !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, #00c8ff, #8b5cf6) !important;
  color: #fff !important; font-weight: 600 !important;
}

/* ─── Selectbox / text inputs ─── */
.stSelectbox div[data-baseweb="select"] > div,
.stTextInput > div > div {
  background: #0d1830 !important;
  border-color: rgba(0,200,255,0.15) !important;
  color: #dde4f0 !important;
  border-radius: 10px !important;
}

/* ─── Dataframes ─── */
.stDataFrame { border: 1px solid rgba(0,200,255,0.1) !important; border-radius: 12px !important; }

/* ─── Progress bar ─── */
.stProgress > div > div { background: linear-gradient(90deg, #00c8ff, #8b5cf6) !important; border-radius: 999px !important; }

/* ─── Alerts ─── */
.stSuccess { background: rgba(0,232,150,0.08) !important; border-color: rgba(0,232,150,0.25) !important; }
.stWarning { background: rgba(255,124,68,0.08) !important; border-color: rgba(255,124,68,0.25) !important; }
.stError   { background: rgba(255,70,80,0.08) !important; border-color: rgba(255,70,80,0.25) !important; }
.stInfo    { background: rgba(0,200,255,0.07) !important; border-color: rgba(0,200,255,0.2) !important; }

/* ─── Class dist bar ─── */
.dist-row {
  display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
  margin-top: 4px;
}
.dist-item {
  background: #0d1830;
  border: 1px solid rgba(0,200,255,0.08);
  border-radius: 10px;
  padding: 8px 12px;
}
.dist-name { font-size: 0.76rem; font-weight: 500; color: #c0cce0; }
.dist-bar-wrap { height: 4px; background: #0a1525; border-radius: 2px; margin-top: 5px; }
.dist-bar { height: 100%; border-radius: 2px; }
.dist-count { font-size: 0.7rem; color: #3a5070; float: right; }

/* ─── SHAP card ─── */
.shap-card {
  background: linear-gradient(135deg, #0f1d38 0%, #1a1038 100%);
  border: 1px solid rgba(139,92,246,0.3);
  border-radius: 16px; padding: 1.6rem 2rem;
  margin-bottom: 1.4rem; position: relative; overflow: hidden;
}
.shap-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, #8b5cf6, #00c8ff);
}
.shap-card h3 { color: #a78bfa; font-size: 1rem; font-weight: 700; margin: 0 0 0.8rem; }
.shap-card code {
  display: inline-block;
  background: rgba(139,92,246,0.15);
  border: 1px solid rgba(139,92,246,0.3);
  color: #c4b5fd;
  padding: 6px 14px; border-radius: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
}

/* ─── Config editor ─── */
.stTextArea > div > div > textarea {
  background: #040810 !important;
  color: #7fb9d0 !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.78rem !important;
  border: 1px solid rgba(0,200,255,0.12) !important;
  border-radius: 12px !important;
}

/* ─── Running indicator ─── */
.running-pill {
  display: inline-flex; align-items: center; gap: 8px;
  background: rgba(0,200,255,0.08);
  border: 1px solid rgba(0,200,255,0.25);
  border-radius: 20px; padding: 6px 14px;
  font-size: 0.78rem; color: #00c8ff; font-weight: 600;
}
.pulse { width: 8px; height: 8px; border-radius: 50%; background: #00c8ff;
  animation: pulse 1.2s infinite; }
@keyframes pulse { 0%{transform:scale(1);opacity:1} 50%{transform:scale(1.4);opacity:0.5} 100%{transform:scale(1);opacity:1} }

/* ─── Hide Streamlit chrome ─── */
#MainMenu, footer, header { visibility: hidden; }
.viewerBadge_container__1QSob, .stDeployButton { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# State
# ══════════════════════════════════════════════════════════════════════════════
def _init():
    defs = {
        "logs":         deque(maxlen=600),
        "status":       {k: "idle" for k in ["preprocess","baseline","attack","defense","ablation"]},
        "active_proc":  None,
        "active_stage": None,
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

def _auto_detect():
    s = st.session_state.status
    checks = [
        ("preprocess", ROOT/"data"/"processed"/"X_train.npy"),
        ("baseline",   ROOT/"checkpoints"/"clean_baseline.pt"),
        ("attack",     ROOT/"results"/"02_attack_sweep.csv"),
        ("defense",    ROOT/"results"/"03_defense_eval.csv"),
        ("ablation",   ROOT/"results"/"04a_trigger_dim.csv"),
    ]
    for key, path in checks:
        if path.exists() and s[key] == "idle":
            s[key] = "done"
_auto_detect()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def ts(): return datetime.now().strftime("%H:%M:%S")

def log(msg, level="info"):
    st.session_state.logs.append((ts(), level, msg))

def data_ok():    return (ROOT/"data"/"processed"/"X_train.npy").exists()
def model_ok():   return (ROOT/"checkpoints"/"clean_baseline.pt").exists()
def attack_ok():  return (ROOT/"results"/"02_attack_sweep.csv").exists()
def defense_ok(): return (ROOT/"results"/"03_defense_eval.csv").exists()

def badge(s):
    cfg = {
        "idle":    ("bd", "◌ Idle"),
        "running": ("br", "⟳ Running"),
        "done":    ("bo", "✓ Done"),
        "error":   ("be", "✗ Error"),
    }
    cls, label = cfg.get(s, ("bd", s))
    return f'<span class="badge {cls}">{label}</span>'

def run_stage(key, cmd):
    if st.session_state.active_proc:
        log("Another stage is running — wait for it to finish.", "err"); return
    st.session_state.status[key] = "running"
    st.session_state.active_stage = key
    log(f"[START] {' '.join(str(c) for c in cmd)}", "info")

    def _thread():
        env = {**os.environ, "PYTHONUTF8": "1"}
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            cwd=str(ROOT), env=env,
        )
        try:
            st.session_state.active_proc = proc
        except Exception:
            pass
        for line in proc.stdout:
            line = line.rstrip()
            if not line: continue
            low = line.lower()
            lvl = ("err"  if any(w in low for w in ["error","traceback","exception","failed"]) else
                   "ok"   if any(w in low for w in ["complete","success","saved","best","done","checkpoint"]) else
                   "info")
            try:
                log(line, lvl)
            except Exception:
                pass  # Silently ignore if session_state is gone (Streamlit re-run)
        proc.wait()
        try:
            st.session_state.status[key] = "done" if proc.returncode == 0 else "error"
            log(f"[{'DONE' if proc.returncode==0 else 'FAILED'}] Exit code {proc.returncode}", "ok" if proc.returncode==0 else "err")
            st.session_state.active_proc  = None
            st.session_state.active_stage = None
        except Exception:
            pass

    threading.Thread(target=_thread, daemon=True).start()

def build_terminal():
    entries = list(st.session_state.logs)[-250:]
    if not entries:
        return '<span class="t-dim">Awaiting commands...</span>\n<span class="t-dim">Click a ▶ Run button to start.</span>'
    lines = []
    for ts_s, lvl, msg in entries:
        esc = msg.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        cls = {"ok":"t-ok","err":"t-err","info":"t-info"}.get(lvl,"t-dim")
        lines.append(f'<span class="t-ts">[{ts_s}]</span> <span class="{cls}">{esc}</span>')
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 1.2rem;">
      <div style="font-size:3rem;line-height:1;">🛡️</div>
      <div style="font-size:1.05rem;font-weight:800;color:#00c8ff;margin-top:8px;">Cyber-Sentinet</div>
      <div style="font-size:0.72rem;color:#3a5070;margin-top:4px;letter-spacing:0.06em;text-transform:uppercase;">
        MSc Thesis · Control Center
      </div>
    </div>
    <div style="height:1px;background:rgba(0,200,255,0.1);margin:0 0 16px;"></div>
    """, unsafe_allow_html=True)

    page = st.radio("nav", [
        "🚀  Pipeline",
        "📊  Results",
        "🔬  SHAP Analysis",
        "⚙️  Config & Files",
    ], label_visibility="collapsed")

    st.markdown("""<div style="height:1px;background:rgba(0,200,255,0.08);margin:16px 0;"></div>""", unsafe_allow_html=True)

    # Pipeline status mini-tracker
    st.markdown('<div style="font-size:0.7rem;color:#3a5070;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;">Pipeline Status</div>', unsafe_allow_html=True)
    icons  = {"idle":"⬜","running":"🔵","done":"✅","error":"🔴"}
    labels = {"preprocess":"0 · Preprocessing","baseline":"1 · Baseline","attack":"2 · Attack Sweep","defense":"3 · Defense Eval","ablation":"4 · Ablation"}
    for key, label in labels.items():
        s = st.session_state.status[key]
        st.markdown(f'<div style="font-size:0.82rem;padding:5px 0;display:flex;align-items:center;gap:8px;">'
                    f'{icons[s]}<span style="color:{"#dde4f0" if s!="idle" else "#3a5070"}">{label}</span></div>',
                    unsafe_allow_html=True)

    if st.session_state.active_stage:
        st.markdown(f"""
        <div style="margin-top:14px;">
          <div class="running-pill">
            <div class="pulse"></div> {labels.get(st.session_state.active_stage,"Running")}
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div style="height:1px;background:rgba(0,200,255,0.08);margin:16px 0;"></div>""", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;color:#3a5070;line-height:1.8;">157,800 samples · 15 classes · 44 features<br/>Edge-IIoTset 2022 · Seed 42</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>🛡️ Trojan-Resilient Cyber-Sentinet</h1>
  <p>Explainability-Guided Backdoor Detection &amp; Mitigation for Industry 5.0 IDS<br/>
     MSc Thesis · Based on Nandanwar &amp; Katarya (2025), <em>Comp. &amp; Electrical Eng.</em> 123: 110161</p>
  <span class="tag">★ Novel: SHAP-Scan Concentration Detector</span>
  <span class="tag" style="margin-left:8px;border-color:rgba(139,92,246,0.4);background:rgba(139,92,246,0.1);color:#a78bfa;">Edge-IIoTset 2022</span>
  <span class="tag" style="margin-left:8px;border-color:rgba(0,232,150,0.4);background:rgba(0,232,150,0.1);color:#00e896;">PyTorch · SHAP · Streamlit</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
if "Pipeline" in page:

    # ── Top metrics ──────────────────────────────────────────────────────────
    active_lbl = st.session_state.active_stage or "—"
    done_count = sum(1 for v in st.session_state.status.values() if v == "done")
    st.markdown(f"""
    <div class="metrics-row">
      <div class="mc c1"><div class="val">157,800</div><div class="lbl">Total Samples</div></div>
      <div class="mc c2"><div class="val">15</div><div class="lbl">Traffic Classes</div></div>
      <div class="mc c3"><div class="val">44</div><div class="lbl">Features</div></div>
      <div class="mc c4"><div class="val">{done_count}/5</div><div class="lbl">Stages Complete</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two columns: stages + terminal ───────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        # ── Full pipeline button ──
        st.markdown('<div class="sec-hdr"><span></span>Quick Actions</div>', unsafe_allow_html=True)
        q1, q2 = st.columns(2)
        with q1:
            if st.button("🚀 Run Full Pipeline", key="full_pipe", type="primary",
                         use_container_width=True, disabled=bool(st.session_state.active_proc)):
                def _full():
                    stages = [
                        ("preprocess", [sys.executable, "-m", "data.preprocessing", "--config", "configs/config.yaml"]),
                        ("baseline",   [sys.executable, "experiments/01_reproduce_baseline.py"]),
                        ("attack",     [sys.executable, "experiments/02_attack_sweep.py"]),
                        ("defense",    [sys.executable, "experiments/03_defense_eval.py", "--attack","feature_trigger","--rate","0.05"]),
                        ("ablation",   [sys.executable, "experiments/04_ablation.py"]),
                    ]
                    for k, cmd in stages:
                        run_stage(k, cmd)
                        while st.session_state.active_proc: time.sleep(1.5)
                threading.Thread(target=_full, daemon=True).start()
                st.rerun()
        with q2:
            if st.button("🔄 Refresh Status", key="refresh", use_container_width=True):
                _auto_detect(); st.rerun()

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr"><span></span>Pipeline Stages</div>', unsafe_allow_html=True)

        # ── Stage definitions ──────────────────────────────────────────────
        stages_def = [
            ("preprocess", "0", "Data Preprocessing",
             "Load Edge-IIoT CSV · Clean · Scale · 7×7 reshape · Save splits",
             lambda: True, [sys.executable, "-m", "data.preprocessing", "--config", "configs/config.yaml"]),
            ("baseline", "1", "Baseline Training",
             "Train Cyber-Sentinet · Target 97.46% acc · Resolve Table 9/10 inconsistency",
             data_ok, [sys.executable, "experiments/01_reproduce_baseline.py"]),
            ("attack", "2", "Attack Sweep",
             "Label-flip + Feature-trigger × 4 poison rates · Measure ASR & CAD",
             model_ok, [sys.executable, "experiments/02_attack_sweep.py"]),
            ("defense", "3", "Defense Evaluation",
             "Spectral Sigs · Activation Clustering · Fine-Pruning · ★ SHAP-Scan",
             attack_ok, None),  # uses dynamic cmd below
            ("ablation", "4", "Ablation Studies",
             "Trigger dims · Target class choice · SHAP-Scan threshold sensitivity",
             attack_ok, [sys.executable, "experiments/04_ablation.py"]),
        ]

        # Defense config
        atk_col, rate_col = st.columns(2)
        with atk_col:
            def_atk  = st.selectbox("Attack →", ["feature_trigger","label_flip"], key="def_atk", label_visibility="collapsed")
        with rate_col:
            rate_map = {"1%":0.01,"3%":0.03,"5%":0.05,"10%":0.10}
            rate_sel = st.selectbox("Rate →", list(rate_map.keys()), index=2, key="def_rate", label_visibility="collapsed")

        for key, num, name, desc, prereq_fn, cmd in stages_def:
            s = st.session_state.status[key]
            cls = {"done":"s-done","running":"s-running","error":"s-error"}.get(s,"")
            if cmd is None:  # defense — dynamic cmd
                cmd = [sys.executable, "experiments/03_defense_eval.py",
                       "--attack", def_atk, "--rate", str(rate_map[rate_sel])]

            st.markdown(f"""
            <div class="stage {cls}">
              <div class="stage-top">
                <span class="stage-name">{num} · {name}</span>
                {badge(s)}
              </div>
              <div class="stage-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

            prereq_ok = prereq_fn()
            disabled  = not prereq_ok or bool(st.session_state.active_proc)
            btn_label = f"▶  Run {name}" if s != "running" else "⟳  Running…"
            if st.button(btn_label, key=f"btn_{key}", disabled=disabled, use_container_width=True):
                run_stage(key, cmd)
                st.rerun()
            if not prereq_ok:
                st.caption(f"⚠ Complete stage {int(num)-1} first")

    # ── Right: terminal + dataset info ────────────────────────────────────────
    with right:
        st.markdown('<div class="sec-hdr"><span></span>Live Terminal</div>', unsafe_allow_html=True)

        terminal_html = build_terminal()
        st.markdown(f"""
        <div class="terminal-wrap">
          <div class="terminal-bar">
            <div class="dot d1"></div><div class="dot d2"></div><div class="dot d3"></div>
            <span class="title">trojan-resilient-cyber-sentinet — bash</span>
          </div>
          <div class="terminal-body" id="tb">{terminal_html}
{"<span class='cursor'></span>" if st.session_state.active_proc else ""}</div>
        </div>
        <script>
          const tb = document.getElementById('tb');
          if(tb) tb.scrollTop = tb.scrollHeight;
        </script>""", unsafe_allow_html=True)

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("🔄 Refresh Log", key="rlog", use_container_width=True): st.rerun()
        with b2:
            if st.button("🗑️ Clear Log", key="clog", use_container_width=True):
                st.session_state.logs.clear(); st.rerun()
        with b3:
            if st.session_state.active_proc and st.button("⛔ Stop", key="stop", use_container_width=True):
                try:
                    st.session_state.active_proc.terminate()
                    st.session_state.status[st.session_state.active_stage] = "error"
                    st.session_state.active_proc = None
                    log("Process terminated by user.", "err")
                except Exception as e:
                    log(f"Stop error: {e}", "err")
                st.rerun()

        # Auto-refresh while running
        if st.session_state.active_proc:
            st.markdown('<div style="color:#00c8ff;font-size:0.78rem;margin-top:6px;">⟳ Auto-refreshing every 3 s</div>', unsafe_allow_html=True)
            time.sleep(3); st.rerun()

        # ── Dataset class distribution ────────────────────────────────────────
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr"><span></span>Dataset Distribution</div>', unsafe_allow_html=True)
        if data_ok():
            try:
                le   = pickle.load(open(ROOT/"data"/"processed"/"label_encoder.pkl","rb"))
                y_tr = np.load(ROOT/"data"/"processed"/"y_train.npy")
                classes = list(le.classes_)
                counts  = sorted([(c, int((y_tr==i).sum())) for i,c in enumerate(classes)], key=lambda x:-x[1])
                total   = len(y_tr)
                attack_colors  = {"Normal":"#00e896"}
                st.markdown('<div class="dist-row">', unsafe_allow_html=True)
                for cls_name, cnt in counts:
                    pct  = 100*cnt/total
                    w    = min(100, int(pct*3.5))
                    col  = "#00e896" if cls_name == "Normal" else "#ff6b70"
                    st.markdown(f"""
                    <div class="dist-item">
                      <span class="dist-name">{cls_name}</span>
                      <span class="dist-count">{cnt:,} ({pct:.1f}%)</span>
                      <div class="dist-bar-wrap"><div class="dist-bar" style="width:{w}%;background:{col};"></div></div>
                    </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.caption(f"Dataset info unavailable: {e}")
        else:
            st.caption("Run preprocessing to see dataset statistics.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif "Results" in page:
    try:
        import pandas as pd
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        PLOT_STYLE = {
            "figure.facecolor":"#070b14","axes.facecolor":"#0d1830",
            "text.color":"#dde4f0","axes.labelcolor":"#9eb0c8",
            "xtick.color":"#6b7fa3","ytick.color":"#6b7fa3",
            "grid.color":"#0d1830","grid.alpha":0.4,
            "axes.spines.top":False,"axes.spines.right":False,
            "axes.edgecolor":"#1a2848","axes.grid":True,
        }
        plt.rcParams.update(PLOT_STYLE)
        HAS_PLOT = True
    except ImportError:
        HAS_PLOT = False

    tab_bl, tab_atk, tab_def, tab_abl = st.tabs([
        "🎯 Baseline", "⚡ Attack Sweep", "🛡️ Defense Results", "🔬 Ablation"
    ])

    # ── Baseline ──────────────────────────────────────────────────────────────
    with tab_bl:
        p = ROOT/"results"/"01_baseline_results.json"
        if p.exists():
            d = json.loads(p.read_text())
            acc, paper = d.get("test_accuracy",0), d.get("paper_accuracy",0.9746)
            diff = abs(acc - paper)
            params = d.get("n_parameters",0)
            c1,c2,c3,c4 = st.columns(4)
            col = "c2" if diff<=0.02 else "c3"
            c1.markdown(f'<div class="mc {col}"><div class="val">{acc:.4f}</div><div class="lbl">Our Accuracy</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="mc c1"><div class="val">{paper:.4f}</div><div class="lbl">Paper Target</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="mc {"c2" if diff<=0.02 else "be"}"><div class="val">{diff:.4f}</div><div class="lbl">Δ Difference</div></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="mc c4"><div class="val">{params:,}</div><div class="lbl">Parameters</div></div>', unsafe_allow_html=True)
            if diff <= 0.02: st.success(f"✅ Reproduction successful — within 2% tolerance of paper's {paper:.4f}")
            else: st.warning(f"⚠ Accuracy gap {diff:.4f} > 0.02 — consider more epochs")
            pcm = d.get("per_class_metrics",{})
            if pcm and HAS_PLOT:
                st.markdown("<br/>**Per-Class Metrics (Tables 9 & 10 Reconciliation)**")
                df_pcm = pd.DataFrame(pcm).T.round(4)
                st.dataframe(df_pcm.style.background_gradient(cmap="RdYlGn", subset=["recall","f1"]), use_container_width=True)
                # F1 bar chart
                fig,ax = plt.subplots(figsize=(11,3.5))
                names = list(df_pcm.index); f1s = df_pcm["f1"].values
                colors = ["#ff6b70" if n!="Normal" else "#00e896" for n in names]
                ax.bar(names, f1s, color=colors, alpha=0.85, width=0.6)
                ax.set_ylim(0,1.08); ax.set_ylabel("F1 Score")
                ax.set_title("Per-Class F1 Score — Baseline Cyber-Sentinet", color="#00c8ff", fontweight="bold")
                plt.xticks(rotation=35, ha="right", fontsize=8); plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
        else:
            st.info("Run **Stage 1 (Baseline Training)** to see results here.")

    # ── Attack Sweep ──────────────────────────────────────────────────────────
    with tab_atk:
        p = ROOT/"results"/"02_attack_sweep.csv"
        if p.exists() and HAS_PLOT:
            df = pd.read_csv(p)
            df["rate_pct"] = df["poison_rate"]*100
            st.dataframe(
                df[["attack_type","rate_pct","asr","cad","poisoned_acc","n_poisoned"]]
                .rename(columns={"attack_type":"Attack","rate_pct":"Rate (%)","asr":"ASR ↑","cad":"CAD ↓","poisoned_acc":"Poisoned Acc","n_poisoned":"# Poisoned"})
                .style.background_gradient(cmap="Reds",subset=["ASR ↑"])
                      .background_gradient(cmap="RdYlGn_r",subset=["Poisoned Acc"]),
                use_container_width=True
            )
            fig, axes = plt.subplots(1,2,figsize=(12,4))
            pal = {"label_flip":"#ff6b70","feature_trigger":"#ff7c44","clean_label":"#a78bfa"}
            for atk,grp in df.groupby("attack_type"):
                grp = grp.sort_values("rate_pct")
                c = pal.get(atk,"#00c8ff")
                axes[0].plot(grp["rate_pct"], grp["asr"]*100, marker="o", color=c, lw=2.5, ms=8, label=atk.replace("_"," ").title())
                axes[1].plot(grp["rate_pct"], grp["cad"]*100, marker="s", color=c, lw=2.5, ms=8, label=atk.replace("_"," ").title())
            axes[0].set_title("ASR vs Poison Rate", color="#00c8ff", fontweight="bold")
            axes[0].set_xlabel("Poison Rate (%)"); axes[0].set_ylabel("ASR (%)")
            axes[0].axhline(80,color="#ff4650",ls="--",alpha=0.5,lw=1); axes[0].legend(framealpha=0.3,facecolor="#0d1830")
            axes[1].set_title("CAD vs Poison Rate", color="#ff7c44", fontweight="bold")
            axes[1].set_xlabel("Poison Rate (%)"); axes[1].set_ylabel("CAD (%)")
            axes[1].legend(framealpha=0.3,facecolor="#0d1830")
            plt.tight_layout(); st.pyplot(fig, use_container_width=True)
        else:
            st.info("Run **Stage 2 (Attack Sweep)** to see results here." if not p.exists() else "Install matplotlib to see charts.")

    # ── Defense Results ───────────────────────────────────────────────────────
    with tab_def:
        p = ROOT/"results"/"03_defense_eval.csv"
        if p.exists() and HAS_PLOT:
            df = pd.read_csv(p)
            st.dataframe(df.style.background_gradient(cmap="RdYlGn",subset=[c for c in df.columns if c in ["f1","auroc","recall","precision"]]), use_container_width=True)
            if "defense" in df.columns and "f1" in df.columns:
                fig,axes = plt.subplots(1,2,figsize=(12,4))
                defs = df["defense"].tolist(); pal = {"spectral_signatures":"#3b82f6","activation_clustering":"#10b981","fine_pruning":"#f59e0b","shap_scan":"#8b5cf6"}
                cols = [pal.get(d,"#00c8ff") for d in defs]
                f1s = [float(v)*100 if not pd.isna(v) else 0 for v in df["f1"]]
                bars = axes[0].bar(defs, f1s, color=cols, alpha=0.85, width=0.55)
                for bar,v in zip(bars,f1s):
                    axes[0].text(bar.get_x()+bar.get_width()/2, v+1.5, f"{v:.1f}%", ha="center", fontsize=9, color="#dde4f0")
                axes[0].set_ylim(0,115); axes[0].set_title("Detection F1 Score", color="#00c8ff", fontweight="bold")
                axes[0].set_ylabel("F1 (%)"); plt.sca(axes[0]); plt.xticks(rotation=25, ha="right", fontsize=9)
                # Highlight SHAP-Scan
                for i,d in enumerate(defs):
                    if "shap" in d.lower():
                        bars[i].set_edgecolor("#a78bfa"); bars[i].set_linewidth(2.5)
                        axes[0].text(i, f1s[i]+7, "★ Novel", ha="center", fontsize=8, color="#a78bfa", fontweight="bold")
                if "auroc" in df.columns:
                    aurocs = [float(v)*100 if not pd.isna(v) else 0 for v in df["auroc"]]
                    bars2 = axes[1].bar(defs, aurocs, color=cols, alpha=0.85, width=0.55)
                    for bar,v in zip(bars2,aurocs):
                        axes[1].text(bar.get_x()+bar.get_width()/2, v+1.5, f"{v:.1f}%", ha="center", fontsize=9, color="#dde4f0")
                    axes[1].set_ylim(0,115); axes[1].set_title("Detection AUROC", color="#8b5cf6", fontweight="bold")
                    axes[1].set_ylabel("AUROC (%)"); plt.sca(axes[1]); plt.xticks(rotation=25, ha="right", fontsize=9)
                plt.tight_layout(); st.pyplot(fig, use_container_width=True)
        else:
            st.info("Run **Stage 3 (Defense Evaluation)** to see results here.")

    # ── Ablation ──────────────────────────────────────────────────────────────
    with tab_abl:
        abl_map = {
            "Trigger Dimensionality": ROOT/"results"/"04a_trigger_dim.csv",
            "Target Class": ROOT/"results"/"04b_target_class.csv",
            "SHAP Threshold Sensitivity": ROOT/"results"/"04d_shap_threshold.csv",
        }
        found = False
        for title, path in abl_map.items():
            if path.exists():
                found = True
                st.markdown(f"**{title}**")
                df = pd.read_csv(path) if HAS_PLOT else None
                if df is not None: st.dataframe(df, use_container_width=True)
        if not found: st.info("Run **Stage 4 (Ablation Studies)** to see results here.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SHAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "SHAP" in page:
    st.markdown("""
    <div class="shap-card">
      <h3>★ SHAP-Scan Concentration Detector — Novel Contribution</h3>
      <p style="color:#9eb0c8;font-size:0.88rem;line-height:1.9;margin-bottom:1rem;">
        <strong style="color:#dde4f0;">Core Insight:</strong> Backdoored samples cause the model to over-rely on 2–4 trigger features.<br/>
        SHAP attributions are <strong style="color:#c4b5fd;">abnormally concentrated</strong> (low entropy) instead of spread across all 44 features.
      </p>
      <div style="margin-bottom:1rem;">
        <div style="font-size:0.78rem;color:#6b7fa3;margin-bottom:6px;">Concentration Score Formula</div>
        <code>score = 1 − H(|SHAP| / Σ|SHAP|) / log(n_features)</code>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:0.82rem;">
        <div style="background:rgba(0,232,150,0.07);border:1px solid rgba(0,232,150,0.2);border-radius:10px;padding:12px;">
          <span style="color:#00e896;font-weight:700;">score ≈ 0</span><br/>
          <span style="color:#6b7fa3;">Uniform attribution → Clean sample</span>
        </div>
        <div style="background:rgba(255,70,80,0.07);border:1px solid rgba(255,70,80,0.2);border-radius:10px;padding:12px;">
          <span style="color:#ff6b70;font-weight:700;">score ≈ 1</span><br/>
          <span style="color:#6b7fa3;">Concentrated on trigger features → ⚠ Backdoor</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown('<div class="sec-hdr"><span></span>Launch Trust Dashboard</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0d1830;border:1px solid rgba(0,200,255,0.12);border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:1rem;">
          <div style="font-weight:600;color:#00c8ff;margin-bottom:6px;">Interactive SHAP Trust Dashboard</div>
          <div style="font-size:0.82rem;color:#5a7090;line-height:1.6;">
            Upload any traffic sample → get prediction + full SHAP attribution chart + SHAP-Scan verdict.<br/>
            Runs on port 8502 as a separate app.
          </div>
        </div>
        """, unsafe_allow_html=True)
        if not model_ok():
            st.warning("Train the baseline model first (Stage 1).")
        else:
            if st.button("🚀 Launch SHAP Dashboard → :8502", key="launch_shap", type="primary", use_container_width=True):
                subprocess.Popen(
                    [sys.executable, "-m", "streamlit", "run", str(ROOT/"dashboard"/"app.py"),
                     "--server.port","8502","--server.headless","true"],
                    cwd=str(ROOT), env={**os.environ,"PYTHONUTF8":"1"},
                )
                st.success("Launching at http://localhost:8502 — open that URL in your browser.")

    with c2:
        st.markdown('<div class="sec-hdr"><span></span>Saved Scan Results</div>', unsafe_allow_html=True)
        score_files = list((ROOT/"results").glob("shap_scan_*scores.npy")) if (ROOT/"results").exists() else []
        if score_files:
            try:
                import matplotlib; matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                plt.rcParams.update({"figure.facecolor":"#070b14","axes.facecolor":"#0d1830","text.color":"#dde4f0"})
                sf = score_files[0]
                scores = np.load(sf)
                fig, ax = plt.subplots(figsize=(7,3))
                ax.hist(scores, bins=60, color="#00c8ff", alpha=0.7, density=True, label="All samples")
                ax.set_xlabel("Concentration Score"); ax.set_ylabel("Density")
                ax.set_title("SHAP-Scan Score Distribution", color="#a78bfa", fontweight="bold")
                ax.legend(framealpha=0.3, facecolor="#0d1830"); plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.caption(f"Could not plot: {e}")
        else:
            st.caption("Run defense evaluation to see SHAP-Scan score distributions here.")

    # ── Why it's novel ──────────────────────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr"><span></span>Why SHAP-Scan is Novel</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    cards = [
        ("🔍", "Zero Extra Training", "Reuses the model's existing XAI layer — no extra model training, no overhead."),
        ("📖", "Human-Readable Evidence", "Shows exactly which features are over-attributed — explainable to non-experts."),
        ("🎯", "Tabular-Native", "Designed for tabular IDS data — no existing equivalent in the literature."),
    ]
    for col,(ico,title,desc) in zip(cols,cards):
        col.markdown(f"""
        <div style="background:#0d1830;border:1px solid rgba(139,92,246,0.15);border-radius:14px;padding:1.2rem;text-align:center;">
          <div style="font-size:2rem;margin-bottom:8px;">{ico}</div>
          <div style="font-weight:700;color:#a78bfa;font-size:0.9rem;margin-bottom:6px;">{title}</div>
          <div style="font-size:0.78rem;color:#5a7090;line-height:1.6;">{desc}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CONFIG & FILES
# ══════════════════════════════════════════════════════════════════════════════
elif "Config" in page:
    c1, c2 = st.columns([3,1], gap="large")
    cfg_path = ROOT/"configs"/"config.yaml"

    with c1:
        st.markdown('<div class="sec-hdr"><span></span>config.yaml — Live Editor</div>', unsafe_allow_html=True)
        raw = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else "# config not found"
        new = st.text_area("", value=raw, height=550, key="cfg_edit", label_visibility="collapsed")
        sv1, sv2 = st.columns([1,3])
        with sv1:
            if st.button("💾 Save Config", key="save_cfg", type="primary", use_container_width=True):
                try:
                    import yaml; yaml.safe_load(new)
                    cfg_path.write_text(new, encoding="utf-8")
                    st.success("Config saved successfully!")
                except Exception as e:
                    st.error(f"Invalid YAML: {e}")

    with c2:
        st.markdown('<div class="sec-hdr"><span></span>Key Settings</div>', unsafe_allow_html=True)
        settings = [
            ("seed","42"), ("epochs","50"), ("batch_size","256"),
            ("lr","0.001"), ("grad_clip","1.0"), ("poison_rates","[0.01,0.03,0.05,0.10]"),
            ("shap threshold","99th pct"), ("reshape","7×7 (44→49)"),
        ]
        for k,v in settings:
            st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:0.78rem;padding:5px 0;border-bottom:1px solid rgba(0,200,255,0.05);">'
                        f'<span style="color:#5a7090;">{k}</span>'
                        f'<span style="color:#00c8ff;font-family:JetBrains Mono,monospace;">{v}</span></div>', unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr"><span></span>Project Files</div>', unsafe_allow_html=True)
        scan_dirs = [
            ("checkpoints/", "Model checkpoints"),
            ("data/processed/", "Processed numpy splits"),
            ("results/", "Experiment results"),
            ("data/poisoned/", "Poisoned datasets"),
        ]
        for rel, desc in scan_dirs:
            full = ROOT/rel.rstrip("/")
            if full.exists():
                files = sorted(full.iterdir())
                total_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1e6
                st.markdown(f'<div style="font-size:0.78rem;font-weight:600;color:#00c8ff;margin-top:10px;">{rel}</div>'
                            f'<div style="font-size:0.7rem;color:#3a5070;">{desc} · {len(files)} files · {total_mb:.1f} MB</div>', unsafe_allow_html=True)
                for f in files[:8]:
                    sz = f"{f.stat().st_size/1024:.0f}KB" if f.is_file() else "dir"
                    st.markdown(f'<div style="font-size:0.72rem;color:#3a5070;font-family:JetBrains Mono,monospace;padding:1px 0;">  {f.name} ({sz})</div>', unsafe_allow_html=True)
