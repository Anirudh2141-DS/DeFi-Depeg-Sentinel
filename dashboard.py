from __future__ import annotations
import os, json, math
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None
try:
    import statsmodels.api as sm
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False
st.set_page_config(
    page_title="Depeg Sentinel",
    page_icon="🛡️",
    layout="wide",
)
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
.metric-card {border-radius:16px;padding:14px 16px;background:rgba(255,255,255,0.03);
  box-shadow:0 8px 22px rgba(0,0,0,0.18);border:1px solid rgba(255,255,255,0.06)}
.metric-title {font-size:0.85rem;opacity:0.8;margin-bottom:2px}
.metric-value {font-size:1.6rem;font-weight:800;line-height:1.1}
.pill {display:inline-block;padding:2px 10px;border-radius:999px;font-size:0.8rem;font-weight:600}
.pill.ok {background:#22c55e;color:#001b07}
.pill.warn {background:#f59e0b;color:#1d1300}
.pill.bad {background:#ef4444;color:#1c0000}
.section {margin-top:.75rem}
hr {margin:.9rem 0;border-color: rgba(255,255,255,.08)}
.small {font-size:.85rem; opacity:.85}
</style>
""", unsafe_allow_html=True)
OUT_DIR = Path(os.getenv("OUT", "outputs")); OUT_DIR.mkdir(parents=True, exist_ok=True)
LIVE_CSV      = Path(os.getenv("LIVE_CSV", OUT_DIR / "live_dataset.csv"))
FORECAST_PQ   = Path(os.getenv("FORECAST_10M", OUT_DIR / "forecast_10m.parquet"))
EXPLAIN_JSON  = Path(os.getenv("EXPLAIN_JSON", OUT_DIR / "explain.json"))
EVENTS_JSON   = Path(os.getenv("EVENTS_JSON", OUT_DIR / "events.json"))
ARTIFACTS_DIR = OUT_DIR / "artifacts"
NOTE_PDF      = OUT_DIR / "analyst_note.pdf"
DEFAULT_TZ = os.getenv("LOCAL_TZ", "America/Denver")
STALE_SEC  = int(os.getenv("STALE_SEC", "1200"))  # 20m
def _tz_to_local(series_utc: pd.Series, tz: str) -> pd.Series:
    s = pd.to_datetime(series_utc, utc=True, errors="coerce")
    try:
        return s.dt.tz_convert(tz)
    except Exception:
        return s
def _freshness_pill(last_ts_utc: Optional[pd.Timestamp], now_utc: datetime) -> str:
    if last_ts_utc is None or pd.isna(last_ts_utc):
        return '<span class="pill bad">No data</span>'
    age = (now_utc - last_ts_utc.to_pydatetime()).total_seconds()
    if age <= STALE_SEC:
        return f'<span class="pill ok">Fresh ({int(age)}s)</span>'
    elif age <= STALE_SEC*3:
        return f'<span class="pill warn">Stale ({int(age)}s)</span>'
    else:
        return f'<span class="pill bad">Cold ({int(age)}s)</span>'
def _metric(label: str, value: str):
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-title">{label}</div>
      <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)
def _fmt_pct(x) -> str:
    return "—" if x is None or (isinstance(x,float) and (math.isnan(x) or math.isinf(x))) else f"{x*100:,.2f}%"
def _synthesize_demo(hours=24, pools=None) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    if pools is None:
        pools = ["USDC/USDT-uni","DAI/USDC-curve","USDT/DAI-uni"]
    ts = pd.date_range(end=pd.Timestamp.utcnow(), periods=hours*60, freq="min", tz="UTC")
    rows = []
    for p in pools:
        dev = rng.normal(0, 0.0008, len(ts))
        dev[int(0.70*len(ts)):int(0.72*len(ts))] += 0.006  # a blip
        fused = np.clip((np.abs(dev) / 0.006), 0, 1) * 0.9
        for t, d, f in zip(ts, dev, fused):
            rows.append({"ts": t, "pool": p, "dev": float(d), "anom_fused": float(f),
                         "neighbor_max_dev": float(np.abs(d)*1.2), "neighbor_avg_anom": float(f*0.6),
                         "lead_lag_best": 0, "corr_best": float(rng.normal(0, .2))})
    return pd.DataFrame(rows)
@st.cache_data(ttl=20)
def load_live(path: Path, demo: bool) -> pd.DataFrame:
    if path.exists() and not demo:
        df = pd.read_csv(path, parse_dates=["ts"])
    else:
        df = _synthesize_demo(hours=48)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    for col in ["dev","anom_fused","neighbor_max_dev","neighbor_avg_anom","lead_lag_best","corr_best"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["pool","ts"]).reset_index(drop=True)
@st.cache_data(ttl=20)
def load_forecast(path: Path, demo: bool, base_df: pd.DataFrame) -> pd.DataFrame:
    if path.exists() and not demo:
        try:
            if path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path)
    else:
        g = base_df.copy()
        g["p_10m"] = np.clip(np.abs(g["dev"]) / 0.006, 0, 1) * 0.75
        g["p_30m"] = np.clip(np.abs(g["dev"]) / 0.006, 0, 1) * 0.85
        df = g[["ts","pool","p_10m","p_30m"]]
    df["ts"] = pd.to_datetime(df.get("ts", pd.NaT), utc=True, errors="coerce")
    for c in list(df.columns):
        if c.lower() == "ts": df.rename(columns={c:"ts"}, inplace=True)
    for a in ["prob_10m","risk_10m","p10","y10"]:
        if a in df.columns: df.rename(columns={a:"p_10m"}, inplace=True)
    for a in ["prob_30m","risk_30m","p30","y30"]:
        if a in df.columns: df.rename(columns={a:"p_30m"}, inplace=True)
    return df.sort_values(["pool","ts"]).reset_index(drop=True)
@st.cache_data(ttl=60)
def load_json(path: Path, demo: bool) -> Dict:
    if path.exists() and not demo:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    now = datetime.now(timezone.utc)
    return {"incidents": [
        {"id":"INC-001","level":"orange","ts": (now - timedelta(hours=2)).isoformat(),
         "pool":"USDC/USDT-uni","summary":"Transient peg deviation; liquidity rotation detected."},
        {"id":"INC-002","level":"yellow","ts": (now - timedelta(hours=18)).isoformat(),
         "pool":"DAI/USDC-curve","summary":"Oracle lag spike; fused anomaly elevated briefly."}
    ]}
@st.cache_data(ttl=20)
def list_artifacts(d: Path, demo: bool) -> Dict[str, List[str]]:
    imgs = [str(p) for p in d.glob("*.png")] if d.exists() and not demo else []
    zips = [str(p) for p in OUT_DIR.glob("*.zip")] if OUT_DIR.exists() else []
    return {"images": imgs, "zips": zips}
with st.sidebar:
    st.title("🛠️ Controls")
    demo_mode = st.toggle("Demo Mode (synthetic data if files missing)", value=not LIVE_CSV.exists())
    tz_choice = st.selectbox(
        "Timezone",
        options=["UTC","America/Denver","America/New_York","Europe/London","Asia/Kolkata","Asia/Singapore"],
        index=(["UTC","America/Denver","America/New_York","Europe/London","Asia/Kolkata","Asia/Singapore"].index(DEFAULT_TZ)
               if DEFAULT_TZ in ["UTC","America/Denver","America/New_York","Europe/London","Asia/Kolkata","Asia/Singapore"] else 1)
    )
    lookback_hours = st.slider("Lookback Window (hours)", 1, 168, 24, step=1)
    dev_thr   = st.number_input("Deviation threshold |abs(dev)|", value=0.005, step=0.001, format="%.3f")
    fused_thr = st.number_input("Fused anomaly threshold", value=0.90, step=0.05, format="%.2f")
    refresh_sec = st.slider("Auto-refresh seconds", 0, 120, 30, step=5, help="0 disables auto-refresh")
    st.caption("Paths")
    st.text(f"{OUT_DIR}")
    st.text(f"{LIVE_CSV.name} | {FORECAST_PQ.name}")
if refresh_sec > 0 and st_autorefresh:
    st_autorefresh(interval=refresh_sec*1000, key="sentinel_refresh")
live   = load_live(LIVE_CSV, demo=demo_mode)
fcst   = load_forecast(FORECAST_PQ, demo=demo_mode, base_df=live)
events = load_json(EVENTS_JSON, demo=demo_mode)
art    = list_artifacts(ARTIFACTS_DIR, demo=demo_mode)
pools = sorted(live["pool"].dropna().unique().tolist()) if not live.empty else []
sel_pools = st.multiselect("Pools", options=pools, default=pools[:min(len(pools), 3)])
st.write("")
colA, colB, colC, colD, colE = st.columns(5)
now_utc = datetime.now(timezone.utc)
if live.empty:
    _metric("Data Status", "No live data")
    _metric("Last Update", "—")
    _metric("Fused anomaly (avg now)", "—")
    _metric("10m risk (avg)", "—")
    _metric("30m risk (avg)", "—")
else:
    dfv = live.copy()
    if sel_pools: dfv = dfv[dfv["pool"].isin(sel_pools)]
    last_ts = dfv["ts"].max() if not dfv.empty else pd.NaT
    last_ts_local = _tz_to_local(pd.Series([last_ts]), tz_choice).iloc[0] if pd.notna(last_ts) else None
    with colA:
        st.markdown(f'**Last update:** {last_ts_local.strftime("%Y-%m-%d %H:%M:%S %Z") if last_ts_local else "—"}')
        st.markdown(_freshness_pill(last_ts, now_utc), unsafe_allow_html=True)
    latest_row = dfv.sort_values("ts").groupby("pool", as_index=False).tail(1)
    fused_now = float(latest_row["anom_fused"].mean()) if "anom_fused" in latest_row.columns and not latest_row.empty else np.nan
    p10_now = np.nan; p30_now = np.nan
    if not fcst.empty:
        temp = fcst if not sel_pools else fcst[fcst["pool"].isin(sel_pools)]
        tmax = temp["ts"].max()
        temp = temp[temp["ts"] == tmax]
        if "p_10m" in temp.columns: p10_now = float(np.nanmean(temp["p_10m"]))
        if "p_30m" in temp.columns: p30_now = float(np.nanmean(temp["p_30m"]))
    with colB: _metric("Fused anomaly (avg now)", f"{fused_now:.3f}" if pd.notna(fused_now) else "—")
    with colC: _metric("10m risk (avg)", _fmt_pct(p10_now) if pd.notna(p10_now) else "—")
    with colD: _metric("30m risk (avg)", _fmt_pct(p30_now) if pd.notna(p30_now) else "—")
    with colE:
        n_incidents = len(events.get("incidents", [])) if isinstance(events, dict) else 0
        _metric("Incidents (total)", f"{n_incidents:,}")
st.markdown("<hr/>", unsafe_allow_html=True)
tab_overview, tab_live, tab_fcst, tab_explain, tab_reports, tab_explore, tab_admin = st.tabs(
    ["Overview", "Live Signals", "Forecasts", "Explainability", "Reports", "Data Explorer", "Admin"]
)
with tab_overview:
    st.subheader("Pulse")
    if live.empty:
        st.info("Waiting for live data. Enable Demo Mode to preview the full dashboard.")
    else:
        df = live.copy()
        if sel_pools: df = df[df["pool"].isin(sel_pools)]
        tmin = df["ts"].max() - timedelta(hours=lookback_hours)
        df = df[df["ts"] >= tmin].copy()
        df["ts_local"] = _tz_to_local(df["ts"], tz_choice)
        df["abs_dev"] = df["dev"].abs()
        df["is_anom"] = ((df["abs_dev"] >= dev_thr) | ((df.get("anom_fused", 0) >= fused_thr))).astype(int)
        fig = go.Figure()
        for p, g in df.groupby("pool"):
            fig.add_trace(go.Scatter(x=g["ts_local"], y=g["dev"], name=f"{p} dev", mode="lines"))
            gg = g[g["is_anom"] == 1]
            if not gg.empty:
                fig.add_trace(go.Scatter(x=gg["ts_local"], y=gg["dev"], name=f"{p} anomalies",
                                         mode="markers", marker=dict(size=7, symbol="x")))
        fig.update_layout(height=380, margin=dict(l=10,r=10,t=30,b=10), yaxis_title="Deviation")
        st.plotly_chart(fig, use_container_width=True)
        agg = df.groupby("pool").agg(
            last=("ts_local","max"),
            max_abs_dev=("abs_dev","max"),
            mean_fused=("anom_fused","mean") if "anom_fused" in df.columns else ("is_anom","mean"),
            anom_hits=("is_anom","sum"),
        ).reset_index()
        st.dataframe(agg, use_container_width=True, hide_index=True)
with tab_live:
    st.subheader("Anomaly & Network Signals")
    if live.empty:
        st.warning("No live data.")
    else:
        df = live.copy()
        if sel_pools: df = df[df["pool"].isin(sel_pools)]
        tmin = df["ts"].max() - timedelta(hours=lookback_hours)
        df = df[df["ts"] >= tmin].copy()
        df["ts_local"] = _tz_to_local(df["ts"], tz_choice)
        df["abs_dev"] = df["dev"].abs()
        c1, c2 = st.columns(2)
        with c1:
            if "anom_fused" in df.columns:
                fig_fused = px.line(df, x="ts_local", y="anom_fused", color="pool", title="Fused Anomaly")
                fig_fused.update_layout(height=340, margin=dict(l=10,r=10,t=35,b=10))
                st.plotly_chart(fig_fused, use_container_width=True)
            else:
                st.info("`anom_fused` not found.")
        with c2:
            fig_hist = px.histogram(df, x="abs_dev", color="pool", nbins=40, title="Abs Deviation Distribution")
            fig_hist.update_layout(height=340, margin=dict(l=10,r=10,t=35,b=10))
            st.plotly_chart(fig_hist, use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            if "neighbor_max_dev" in df.columns:
                fig_ng = px.line(df, x="ts_local", y="neighbor_max_dev", color="pool", title="Neighbor Max Dev")
                fig_ng.update_layout(height=340, margin=dict(l=10,r=10,t=35,b=10))
                st.plotly_chart(fig_ng, use_container_width=True)
            else:
                st.info("`neighbor_max_dev` not present.")
        with c4:
            if "corr_best" in df.columns:
                trend_kw = {"trendline": "lowess"} if _HAS_STATSMODELS else {}
                fig_corr = px.scatter(df, x="ts_local", y="corr_best", color="pool",
                                      title="Best Rolling Correlation", **trend_kw)
                fig_corr.update_layout(height=340, margin=dict(l=10,r=10,t=35,b=10))
                st.plotly_chart(fig_corr, use_container_width=True)
                if not _HAS_STATSMODELS:
                    st.caption("ℹ️ Install `statsmodels` to enable LOWESS trendlines.")
            else:
                st.info("`corr_best` not present.")
with tab_fcst:
    st.subheader("Short-Horizon Risk Forecasts")
    if fcst.empty:
        st.info("No forecast file found.")
    else:
        df = fcst.copy()
        if sel_pools: df = df[df["pool"].isin(sel_pools)]
        tmin = df["ts"].max() - timedelta(hours=lookback_hours)
        df = df[df["ts"] >= tmin].copy()
        df["ts_local"] = _tz_to_local(df["ts"], tz_choice)
        c1, c2 = st.columns(2)
        with c1:
            if "p_10m" in df.columns:
                f10 = px.line(df, x="ts_local", y="p_10m", color="pool", title="10-minute Risk")
                f10.update_layout(height=340, margin=dict(l=10,r=10,t=35,b=10), yaxis_tickformat=".0%")
                st.plotly_chart(f10, use_container_width=True)
            else:
                st.info("`p_10m` not found.")
        with c2:
            if "p_30m" in df.columns:
                f30 = px.line(df, x="ts_local", y="p_30m", color="pool", title="30-minute Risk")
                f30.update_layout(height=340, margin=dict(l=10,r=10,t=35,b=10), yaxis_tickformat=".0%")
                st.plotly_chart(f30, use_container_width=True)
            else:
                st.info("`p_30m` not found.")
        if not live.empty:
            join = df[["ts","pool","p_10m"]].rename(columns={"p_10m":"risk"}) if "p_10m" in df.columns else \
                   df[["ts","pool","p_30m"]].rename(columns={"p_30m":"risk"})
            merged = pd.merge(live, join, on=["ts","pool"], how="inner")
            merged["ts_local"] = _tz_to_local(merged["ts"], tz_choice)
            merged["abs_dev"] = merged["dev"].abs()
            trend_kw = {"trendline": "ols"} if _HAS_STATSMODELS else {}
            sc = px.scatter(merged, x="abs_dev", y="risk", color="pool",
                            title="Risk vs Absolute Deviation", **trend_kw)
            sc.update_layout(height=380, margin=dict(l=10,r=10,t=35,b=10), yaxis_tickformat=".0%")
            st.plotly_chart(sc, use_container_width=True)
            if not _HAS_STATSMODELS:
                st.caption("ℹ️ Install `statsmodels` to enable OLS trendlines.")
with tab_explain:
    st.subheader("Explainability & Feature Diagnostics")
    exp = {}
    if EXPLAIN_JSON.exists() and not demo_mode:
        try:
            exp = json.loads(EXPLAIN_JSON.read_text(encoding="utf-8"))
        except Exception:
            exp = {}
    importances = []
    if isinstance(exp, dict) and exp:
        if "importances" in exp and isinstance(exp["importances"], list):
            importances = exp["importances"]
        elif "feature_importances" in exp and isinstance(exp["feature_importances"], dict):
            importances = [{"feature":k,"importance":v} for k,v in exp["feature_importances"].items()]
        elif all(isinstance(v,(int,float)) for v in exp.values()):
            importances = [{"feature":k,"importance":v} for k,v in exp.items()]
    if importances:
        imp_df = pd.DataFrame(importances).sort_values("importance", ascending=False).head(25)
        fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h", title="Top Feature Importances")
        fig_imp.update_layout(height=560, margin=dict(l=10,r=10,t=35,b=10))
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("No explainability file found or unrecognized format. Populate `outputs/explain.json` to enable this view.")
    if not live.empty:
        df = live.copy()
        if sel_pools: df = df[df["pool"].isin(sel_pools)]
        tmin = df["ts"].max() - timedelta(hours=lookback_hours)
        df = df[df["ts"] >= tmin]
        num = df.select_dtypes(include=[np.number])
        if not num.empty and num.shape[1] >= 2:
            corr = num.corr(numeric_only=True)
            hm = px.imshow(corr, aspect="auto", title="Numeric Feature Correlations")
            hm.update_layout(height=560, margin=dict(l=10,r=10,t=35,b=10))
            st.plotly_chart(hm, use_container_width=True)
        else:
            st.info("Not enough numeric columns to compute correlations.")
with tab_reports:
    st.subheader("Reports & Artifacts")
    c1, c2 = st.columns(2)
    with c1:
        if NOTE_PDF.exists() and not demo_mode:
            with open(NOTE_PDF, "rb") as f:
                st.download_button("⬇️ Download Analyst Note (PDF)", f, file_name=NOTE_PDF.name, use_container_width=True)
        else:
            st.info("`analyst_note.pdf` not found (or demo mode).")
        det = ARTIFACTS_DIR / "detector_pr_auc.png"
        if det.exists() and not demo_mode:
            st.image(str(det), caption="Detector PR-AUC", use_column_width=True)
        else:
            st.info("`artifacts/detector_pr_auc.png` not found.")
    with c2:
        c10 = ARTIFACTS_DIR / "calibration_10m.png"
        c30 = ARTIFACTS_DIR / "calibration_30m.png"
        shown = False
        if c10.exists() and not demo_mode:
            st.image(str(c10), caption="Calibration: 10m", use_column_width=True); shown = True
        if c30.exists() and not demo_mode:
            st.image(str(c30), caption="Calibration: 30m", use_column_width=True); shown = True
        if not shown:
            st.info("Calibration charts not found in `artifacts/`.")
    st.markdown("---")
    zips = [p for p in OUT_DIR.glob("*.zip")] if OUT_DIR.exists() else []
    if zips:
        z = max(zips, key=lambda p: p.stat().st_mtime)
        with open(z, "rb") as f:
            st.download_button("⬇️ Download Latest Incident Pack (ZIP)", f, file_name=z.name, use_container_width=True)
    else:
        st.info("No incident pack ZIPs detected.")
    st.markdown("---")
    inc = events.get("incidents", []) if isinstance(events, dict) else []
    if inc:
        st.write("**Incidents**")
        df_inc = pd.DataFrame(inc)
        if "ts" in df_inc.columns:
            df_inc["ts"] = pd.to_datetime(df_inc["ts"], utc=True, errors="coerce")
            df_inc["ts_local"] = _tz_to_local(df_inc["ts"], tz_choice)
        cols = [c for c in ["id","level","pool","ts_local","summary"] if c in df_inc.columns]
        st.dataframe(df_inc[cols] if cols else df_inc, use_container_width=True, hide_index=True)
    else:
        st.info("No incidents recorded.")
with tab_explore:
    st.subheader("Data Explorer")
    if live.empty:
        st.info("No live data.")
    else:
        df = live.copy()
        if sel_pools: df = df[df["pool"].isin(sel_pools)]
        tmin = df["ts"].max() - timedelta(hours=lookback_hours)
        df = df[df["ts"] >= tmin].copy()
        df["ts_local"] = _tz_to_local(df["ts"], tz_choice)
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in {"ts"}]
        c1, c2 = st.columns([2,1])
        with c1:
            st.dataframe(df, use_container_width=True)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download filtered CSV", data=csv_bytes, file_name="filtered_live.csv", mime="text/csv")
        with c2:
            if numeric_cols:
                x = st.selectbox("X (numeric)", options=numeric_cols, index=0)
                y = st.selectbox("Y (numeric)", options=numeric_cols, index=min(1,len(numeric_cols)-1))
                sc2 = px.scatter(df, x=x, y=y, color="pool", title=f"{y} vs {x}")
                sc2.update_layout(height=360, margin=dict(l=10,r=10,t=35,b=10))
                st.plotly_chart(sc2, use_container_width=True)
            else:
                st.info("Not enough numeric columns to chart.")
with tab_admin:
    st.subheader("Admin & Runtime")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Environment**")
        st.json({
            "OUT_DIR": str(OUT_DIR),
            "LIVE_CSV": str(LIVE_CSV),
            "FORECAST_10M_PARQUET": str(FORECAST_PQ),
            "EXPLAIN_JSON": str(EXPLAIN_JSON),
            "EVENTS_JSON": str(EVENTS_JSON),
            "ARTIFACTS_DIR": str(ARTIFACTS_DIR),
            "NOTE_PDF": str(NOTE_PDF),
            "STALE_SEC": STALE_SEC,
            "TZ": tz_choice,
            "DemoMode": demo_mode,
            "TrendlinesEnabled": _HAS_STATSMODELS,
        })
    with c2:
        st.write("**Hints**")
        st.markdown("""
- Backend (notebooks/services) should write to `outputs/` while the UI stays read-only.
- Timezone-safe: parse timestamps as **UTC** and convert only for display.
- For PR/ROC/calibration plots: save PNGs in `outputs/artifacts/`.
- For incidents: write `outputs/events.json` as `{"incidents":[...]}`
""")
