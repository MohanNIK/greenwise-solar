# app.py
# GreenWise · Solar PV Decision Platform
# Run:
#   streamlit run app.py
#
# Optional files in same folder (case-sensitive on Streamlit Cloud!):
# - weback1.png          (main background for app pages)
# - weback2.png          (welcome/landing background)
# - greenwise_logo.png   (logo)
# - xgb_payback.pkl      (optional; if missing, auto-trains a demo model)
# - synthetic_train_data.csv (auto-generated for SHAP background)

import os
import math
import base64
from typing import Optional
from textwrap import dedent

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

HAS_ML = True
try:
    import joblib
    import shap
    import xgboost as xgb
    import matplotlib.pyplot as plt
except Exception:
    HAS_ML = False

st.set_page_config(page_title="GreenWise · Solar PV Decision Platform", layout="wide")

# -------------------------
# Assets
# -------------------------
BG_MAIN = "weback1.png"
BG_WELCOME = "weback2.png"
LOGO_PATH = "greenwise_logo.png"

# -------------------------
# Constants
# -------------------------
SHADING_MAP = {"Low": 0.95, "Medium": 0.85, "High": 0.70}

FEATURE_COLS = [
    "annual_consumption_kwh",
    "price_y_per_kwh",
    "roof_area_m2",
    "shading_factor",
    "capex_y_per_kw",
    "self_use_ratio",
    "pv_kw",
]


# =========================
# Utilities
# =========================
def img_to_base64(path: str) -> Optional[str]:
    if (not path) or (not os.path.exists(path)):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def safe_get_query_params():
    try:
        return dict(st.query_params)
    except Exception:
        return st.experimental_get_query_params()


def clear_query_params():
    try:
        st.query_params.clear()
    except Exception:
        st.experimental_set_query_params()


def do_rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def fmt_year(x):
    if x == math.inf or (isinstance(x, float) and not np.isfinite(x)):
        return "∞"
    return f"{x:.1f}"


def money(x):
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)


def recommendation(payback_y, roi_y1):
    if payback_y == math.inf:
        return (
            "Not recommended",
            "Net cashflow is too low or negative. Consider reducing CAPEX, increasing self-consumption, or re-evaluating under higher electricity prices/better solar yield.",
        )
    if payback_y <= 4 and roi_y1 >= 0.18:
        return (
            "Strongly recommended",
            "Fast payback with attractive ROI. This scenario is highly feasible for feasibility study and rollout planning.",
        )
    if payback_y <= 7:
        return (
            "Recommended",
            "Feasible. You can further improve returns by optimizing capacity, increasing self-consumption, or lowering CAPEX.",
        )
    return (
        "Borderline",
        "Payback is relatively long. Consider adjusting the sizing or key assumptions before committing.",
    )


# =========================
# Core model (formula engine)
# =========================
def pv_estimate(
    annual_consumption_kwh: float,
    price_y_per_kwh: float,
    roof_area_m2: float,
    shading_level: str,
    capex_y_per_kw: float,
    opex_pct: float,
    self_use_ratio: float,
    co2_factor_kg_per_kwh: float,
    degradation_pct: float,
    discount_rate_pct: float,
    pv_kw: float,
    kw_per_m2: float = 0.20,
    kwh_per_kw_year: float = 1100.0,
    export_price_y_per_kwh: float = 0.0,
):
    shading_factor = SHADING_MAP.get(shading_level, 0.85)

    kw_max_by_roof = max(0.0, roof_area_m2 * kw_per_m2)
    pv_kw = float(min(max(0.0, pv_kw), kw_max_by_roof))

    gen_kwh_y1 = pv_kw * kwh_per_kw_year * shading_factor

    self_used_kwh_y1 = min(gen_kwh_y1 * self_use_ratio, annual_consumption_kwh)
    exported_kwh_y1 = max(0.0, gen_kwh_y1 - self_used_kwh_y1)

    annual_savings_y1 = self_used_kwh_y1 * price_y_per_kwh + exported_kwh_y1 * export_price_y_per_kwh

    capex = pv_kw * capex_y_per_kw
    opex_y1 = capex * (opex_pct / 100.0)

    net_cash_y1 = annual_savings_y1 - opex_y1
    simple_payback = (capex / net_cash_y1) if net_cash_y1 > 0 else math.inf
    roi_y1 = (net_cash_y1 / capex) if capex > 0 else 0.0

    disc = discount_rate_pct / 100.0
    degr = degradation_pct / 100.0

    cum_disc = -capex
    discounted_payback = math.inf
    cash_rows = []
    for year in range(1, 21):
        gen = gen_kwh_y1 * ((1 - degr) ** (year - 1))
        self_used = min(gen * self_use_ratio, annual_consumption_kwh)
        exported = max(0.0, gen - self_used)
        savings = self_used * price_y_per_kwh + exported * export_price_y_per_kwh

        opex = opex_y1
        net = savings - opex

        disc_net = net / ((1 + disc) ** year) if disc > 0 else net
        cum_disc += disc_net
        if cum_disc >= 0 and discounted_payback == math.inf:
            discounted_payback = year

        cash_rows.append(
            {
                "Year": year,
                "Generation_kWh": gen,
                "Savings_Y": savings,
                "OPEX_Y": opex,
                "NetCash_Y": net,
                "DiscountedNet_Y": disc_net,
                "CumDiscounted_Y": cum_disc,
            }
        )

    cash_df = pd.DataFrame(cash_rows)
    co2_ton_y1 = (gen_kwh_y1 * co2_factor_kg_per_kwh) / 1000.0

    return {
        "pv_kw": pv_kw,
        "kw_max_by_roof": kw_max_by_roof,
        "gen_kwh_y1": gen_kwh_y1,
        "annual_savings_y1": annual_savings_y1,
        "capex": capex,
        "opex_y1": opex_y1,
        "net_cash_y1": net_cash_y1,
        "simple_payback_y": simple_payback,
        "roi_y1": roi_y1,
        "discounted_payback_y": discounted_payback,
        "co2_ton_y1": co2_ton_y1,
        "shading_factor": shading_factor,
        "kwh_per_kw_year": kwh_per_kw_year,
        "exported_kwh_y1": exported_kwh_y1,
        "self_used_kwh_y1": self_used_kwh_y1,
        "cashflow_df": cash_df,
    }


def find_optimal_capacity(kw_max: float, step_kw: float, objective: str, **kwargs):
    if kw_max <= 0:
        return None, pd.DataFrame()

    records = []
    for kw in np.arange(step_kw, kw_max + 1e-9, step_kw):
        r = pv_estimate(pv_kw=float(kw), **kwargs)
        records.append(
            {
                "kW": float(kw),
                "Payback(y)": r["simple_payback_y"],
                "ROI(Y1)": r["roi_y1"],
                "DiscPayback(y)": r["discounted_payback_y"],
                "CO2(t/y)": r["co2_ton_y1"],
                "NetCashY1(¥)": r["net_cash_y1"],
            }
        )

    df = pd.DataFrame(records)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Payback(y)", "ROI(Y1)"])
    if df.empty:
        return None, df

    if objective == "max_roi":
        best = df.loc[df["ROI(Y1)"].idxmax()]
    elif objective == "min_discounted_payback":
        df2 = df.dropna(subset=["DiscPayback(y)"])
        best = df2.loc[df2["DiscPayback(y)"].idxmin()] if not df2.empty else df.loc[df["Payback(y)"].idxmin()]
    else:
        best = df.loc[df["Payback(y)"].idxmin()]

    return best, df


# =========================
# SHAP helpers (robust + stable)
# =========================
def try_load_model(path="xgb_payback.pkl"):
    if not HAS_ML:
        return None
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def train_demo_model_synthetic(n_samples: int = 2500, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    rows = []

    for _ in range(n_samples):
        annual = float(rng.uniform(50_000, 3_000_000))
        price = float(rng.uniform(0.35, 1.80))
        roof = float(rng.uniform(200, 20_000))
        shade_label = rng.choice(list(SHADING_MAP.keys()))
        shade_factor = SHADING_MAP[shade_label]

        capex = float(rng.uniform(2500, 8000))
        opex = float(rng.uniform(0.5, 2.0))
        self_use = float(rng.uniform(0.35, 1.0))
        co2 = float(rng.uniform(0.35, 0.85))
        degr = float(rng.uniform(0.2, 1.2))
        disc = float(rng.uniform(2.0, 10.0))
        kw_per_m2 = float(rng.uniform(0.12, 0.28))
        yield_kwh = float(rng.uniform(800, 1500))
        export_price = float(rng.uniform(0.0, 0.35))

        kw_max = roof * kw_per_m2
        pv_kw = float(rng.uniform(1.0, max(2.0, kw_max)))

        r = pv_estimate(
            annual_consumption_kwh=annual,
            price_y_per_kwh=price,
            roof_area_m2=roof,
            shading_level=shade_label,
            capex_y_per_kw=capex,
            opex_pct=opex,
            self_use_ratio=self_use,
            co2_factor_kg_per_kwh=co2,
            degradation_pct=degr,
            discount_rate_pct=disc,
            pv_kw=pv_kw,
            kw_per_m2=kw_per_m2,
            kwh_per_kw_year=yield_kwh,
            export_price_y_per_kwh=export_price,
        )

        y = r["simple_payback_y"]
        if y == math.inf or (isinstance(y, float) and not np.isfinite(y)):
            continue

        y = float(np.clip(y, 0.5, 25.0) + rng.normal(0, 0.15))

        rows.append(
            {
                "annual_consumption_kwh": annual,
                "price_y_per_kwh": price,
                "roof_area_m2": roof,
                "shading_factor": shade_factor,
                "capex_y_per_kw": capex,
                "self_use_ratio": self_use,
                "pv_kw": pv_kw,
                "payback_period": y,
            }
        )

    df = pd.DataFrame(rows)
    X = df[FEATURE_COLS].astype(float)
    y = df["payback_period"].astype(float)

    model = xgb.XGBRegressor(
        n_estimators=260,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
    )
    model.fit(X, y)
    return model, df


def ensure_demo_model():
    if not HAS_ML:
        return None

    model = try_load_model("xgb_payback.pkl")
    if model is not None:
        return model

    demo_model, demo_df = train_demo_model_synthetic(n_samples=2500, random_state=42)
    try:
        joblib.dump(demo_model, "xgb_payback.pkl")
        if not os.path.exists("synthetic_train_data.csv"):
            demo_df.to_csv("synthetic_train_data.csv", index=False)
    except Exception:
        pass
    return demo_model


@st.cache_data(show_spinner=False)
def get_shap_background(max_rows: int = 400) -> pd.DataFrame:
    if os.path.exists("synthetic_train_data.csv"):
        try:
            df = pd.read_csv("synthetic_train_data.csv")
            if all(c in df.columns for c in FEATURE_COLS):
                return df[FEATURE_COLS].head(max_rows).astype(float)
        except Exception:
            pass

    if HAS_ML:
        try:
            _, demo_df = train_demo_model_synthetic(n_samples=max(900, max_rows), random_state=7)
            return demo_df[FEATURE_COLS].head(max_rows).astype(float)
        except Exception:
            pass

    return pd.DataFrame(np.zeros((max_rows, len(FEATURE_COLS))), columns=FEATURE_COLS).astype(float)


def _predict_any_xgb(model, z_np: np.ndarray) -> np.ndarray:
    z_df = pd.DataFrame(z_np, columns=FEATURE_COLS).astype(float)

    try:
        if HAS_ML and ("xgb" in globals()) and isinstance(model, xgb.Booster):
            dm = xgb.DMatrix(z_df.values, feature_names=FEATURE_COLS)
            return np.array(model.predict(dm), dtype=float).reshape(-1)
    except Exception:
        pass

    return np.array(model.predict(z_df), dtype=float).reshape(-1)


def shap_single_plots(model, X_input: pd.DataFrame, plot_type: str):
    X = X_input.copy().astype(float)
    X_np = X.values.astype(np.float64)

    try:
        model_for_explainer = model
        try:
            if hasattr(model, "get_booster"):
                model_for_explainer = model.get_booster()
        except Exception:
            model_for_explainer = model

        explainer = shap.TreeExplainer(model_for_explainer)
        shap_vals = explainer.shap_values(X_np)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        sv_row = np.array(shap_vals[0], dtype=float).reshape(-1)

        base_val = getattr(explainer, "expected_value", 0.0)
        try:
            base_val = float(np.array(base_val).reshape(-1)[0])
        except Exception:
            base_val = 0.0

        exp = shap.Explanation(
            values=sv_row,
            base_values=base_val,
            data=X_np[0],
            feature_names=list(X.columns),
        )

        plt.figure()
        if plot_type == "Waterfall":
            shap.plots.waterfall(exp, show=False)
        else:
            shap.plots.bar(exp, show=False)

        return plt.gcf(), sv_row

    except Exception:
        bg = get_shap_background(max_rows=400)
        masker = shap.maskers.Independent(bg.values.astype(np.float64))

        def _predict(z):
            return _predict_any_xgb(model, z)

        max_evals = 256
        try:
            perm = shap.explainers.Permutation(_predict, masker, feature_names=FEATURE_COLS)
            explanation = perm(X_np, max_evals=max_evals)
        except Exception:
            explainer = shap.Explainer(_predict, masker, algorithm="permutation", feature_names=FEATURE_COLS)
            explanation = explainer(X_np, max_evals=max_evals)

        exp0 = explanation[0]
        sv_row = np.array(exp0.values, dtype=float).reshape(-1)

        plt.figure()
        if plot_type == "Waterfall":
            shap.plots.waterfall(exp0, show=False)
        else:
            shap.plots.bar(exp0, show=False)

        return plt.gcf(), sv_row


def shap_text_explanation(X_input: pd.DataFrame, shap_values_row: np.ndarray):
    pairs = list(zip(X_input.columns, shap_values_row))
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:3]

    inc = [f"{k} (↑ payback)" for k, v in pairs_sorted if v > 0]
    dec = [f"{k} (↓ payback)" for k, v in pairs_sorted if v < 0]

    lines = []
    if inc:
        lines.append("Main factors increasing predicted payback: " + ", ".join(inc) + ".")
    if dec:
        lines.append("Main factors decreasing predicted payback: " + ", ".join(dec) + ".")
    if not lines:
        lines.append("SHAP contributions are near zero (prediction close to baseline).")
    return " ".join(lines)


# =========================
# Global CSS
# =========================
def inject_global_css():
    bg64 = img_to_base64(BG_MAIN)
    bg_css = f"background-image:url('data:image/png;base64,{bg64}');" if bg64 else "background:#0a0e1c;"

    st.markdown(
        dedent(
            f"""
<style>
/* App background */
.stApp {{
  {bg_css}
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
}}

/* Dark overlay for readability */
.stApp::before {{
  content:"";
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.42);
  pointer-events:none;
  z-index: 0;
}}

section[data-testid="stSidebar"], main, header, footer {{
  position: relative;
  z-index: 1;
}}

/* Reduce top blank area (Cloud looks better) */
header[data-testid="stHeader"] {{
  background: transparent !important;
}}
div[data-testid="stToolbar"] {{
  visibility: hidden !important;
  height: 0 !important;
}}
#MainMenu {{ visibility: hidden; }}

.block-container {{
  padding-top: 0.35rem;
  padding-bottom: 1.0rem;
}}

footer {{
  visibility: hidden;
  height: 0;
}}

/* Sidebar */
section[data-testid="stSidebar"]{{
  background: rgba(0,0,0,0.30);
  border-right: 1px solid rgba(255,255,255,0.10);
  backdrop-filter: blur(10px);
}}

/* Metric cards */
div[data-testid="stMetric"] {{
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 16px;
  padding: 12px 14px;
  backdrop-filter: blur(6px);
}}
div[data-testid="stMetric"] * {{
  color: rgba(255,255,255,0.92) !important;
}}

/* Global typography (IMPORTANT: do NOT color all div; it breaks button text) */
h1, h2, h3, h4, h5, h6, p, label {{
  color: rgba(255,255,255,0.92);
}}
.stCaption, .stMarkdown small {{
  color: rgba(255,255,255,0.72) !important;
}}
div[data-testid="stMarkdownContainer"] * {{
  color: rgba(255,255,255,0.92);
}}

/* --- Selectbox visibility FIX --- */
div[data-testid="stSelectbox"] label {{
  color: rgba(255,75,75,0.98) !important;
  font-weight: 800 !important;
}}
div[data-baseweb="select"] {{
  font-weight: 800 !important;
}}
div[data-baseweb="select"] * {{
  color: rgba(255,75,75,0.98) !important;
  font-weight: 800 !important;
}}
div[data-baseweb="select"] > div {{
  background: rgba(0,0,0,0.35) !important;
  border: 1px solid rgba(255,255,255,0.22) !important;
}}
div[data-baseweb="popover"] {{
  z-index: 100000 !important;
}}
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] [role="listbox"] {{
  background: rgba(10,10,10,0.96) !important;
  border: 1px solid rgba(255,255,255,0.20) !important;
}}
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] li * {{
  color: rgba(255,75,75,0.98) !important;
  font-weight: 800 !important;
}}
div[data-baseweb="popover"] li:hover {{
  background: rgba(255,255,255,0.10) !important;
}}

/* Dataframe look */
.stDataFrame {{
  background: rgba(0,0,0,0.18) !important;
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  overflow: hidden !important;
}}

/* ===== Buttons: force readable text ===== */
div[data-testid="stButton"] button,
div[data-testid="stDownloadButton"] button,
div.stDownloadButton > button,
.stButton > button {{
  background: rgba(255,255,255,0.95) !important;
  color: #111111 !important;
  border: 1px solid rgba(255,255,255,0.75) !important;
  font-weight: 900 !important;
}}
div[data-testid="stButton"] button *,
div[data-testid="stDownloadButton"] button * {{
  color: #111111 !important;
  font-weight: 900 !important;
}}
div[data-testid="stButton"] button:hover,
div[data-testid="stDownloadButton"] button:hover {{
  background: #ffffff !important;
  color: #111111 !important;
}}

/* Key-based button styling (for dark buttons) */
.st-key-btn_shap :is(button, a),
.st-key-dl_summary :is(button, a),
.st-key-dl_capacity :is(button, a),
.st-key-dl_cashflow :is(button, a) {{
  background: rgba(0,0,0,0.92) !important;
  border: 1px solid rgba(255,255,255,0.35) !important;
  border-radius: 10px !important;
}}
.st-key-btn_shap :is(button, a, button *, a *),
.st-key-dl_summary :is(button, a, button *, a *),
.st-key-dl_capacity :is(button, a, button *, a *),
.st-key-dl_cashflow :is(button, a, button *, a *) {{
  color: #ffffff !important;
  font-weight: 900 !important;
}}
.st-key-dl_summary a,
.st-key-dl_capacity a,
.st-key-dl_cashflow a {{
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  padding: 0.45rem 0.9rem !important;
  text-decoration: none !important;
}}
.st-key-btn_shap :is(button:hover, a:hover),
.st-key-dl_summary :is(button:hover, a:hover),
.st-key-dl_capacity :is(button:hover, a:hover),
.st-key-dl_cashflow :is(button:hover, a:hover) {{
  background: #000000 !important;
}}

/* About page nav buttons (white bg, black text, full width friendly) */
.st-key-about_to_platform button,
.st-key-about_to_welcome button {{
  background: rgba(255,255,255,0.95) !important;
  border: 1px solid rgba(255,255,255,0.75) !important;
  border-radius: 12px !important;
  width: 100% !important;
}}
.st-key-about_to_platform button *,
.st-key-about_to_welcome button * {{
  color: #111111 !important;
  font-weight: 900 !important;
}}

/* Chips */
.gw-chip {{
  display:inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,0.92);
  color: #111111 !important;
  font-weight: 900;
  border: 1px solid rgba(0,0,0,0.12);
  margin: 6px 0 10px 0;
}}
</style>
"""
        ),
        unsafe_allow_html=True,
    )


# =========================
# Welcome Screen (Landing)
# =========================
def show_welcome_screen():
    bg64 = img_to_base64(BG_WELCOME)
    logo64 = img_to_base64(LOGO_PATH)

    mission = (
        "GreenWise helps enterprises make transparent, data-driven rooftop solar decisions—"
        "optimizing PV capacity, estimating payback/ROI, quantifying CO₂ reduction, and explaining key drivers with SHAP."
    )

    bg_css = f"background-image:url('data:image/png;base64,{bg64}');" if bg64 else "background:#0a0e1c;"
    logo_html = f"<img src='data:image/png;base64,{logo64}' style='height:82px;margin:0 0 14px 0;'/>" if logo64 else ""

    welcome_html = (
        "<style>"
        "header[data-testid='stHeader']{background:transparent !important;}"
        "div[data-testid='stToolbar']{visibility:hidden !important;height:0 !important;}"
        "#MainMenu{visibility:hidden;}"
        f".welcome-overlay{{position:fixed;inset:0;{bg_css}background-size:cover;background-position:center;z-index:9999;}}"
        ".welcome-overlay::before{content:'';position:absolute;inset:0;background:radial-gradient(1200px 800px at 20% 20%, rgba(0,0,0,0.10), rgba(0,0,0,0.78));}"
        ".welcome-card{position:absolute;left:50%;top:44%;transform:translate(-50%,-50%);width:min(900px,92vw);padding:30px 28px;border-radius:26px;"
        "background:rgba(0,0,0,0.35);border:1px solid rgba(255,255,255,0.18);backdrop-filter:blur(12px);color:rgba(255,255,255,0.92);text-align:center;}"
        ".welcome-title{font-size:52px;font-weight:900;letter-spacing:0.4px;margin:6px 0 10px 0;}"
        ".welcome-subtitle{font-size:15px;opacity:0.86;margin-bottom:12px;}"
        ".welcome-mission{font-size:16px;line-height:1.6;opacity:0.92;margin:0 auto 18px auto;max-width:780px;}"
        ".welcome-actions{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:10px;}"
        ".welcome-btn{display:inline-block;padding:14px 28px;border-radius:999px;background:#ffffff;border:1px solid rgba(255,255,255,0.75);"
        "color:#111111;font-weight:900;font-size:18px;text-decoration:none;}"
        ".welcome-btn.secondary{background:rgba(0,0,0,0.55);border:1px solid rgba(255,255,255,0.35);color:#ffffff;}"
        ".welcome-hint{font-size:14px;opacity:0.85;margin-top:12px;}"
        "</style>"
        "<div class='welcome-overlay' id='gw_welcome'>"
        "<div class='welcome-card'>"
        f"{logo_html}"
        "<div class='welcome-title'>GreenWise</div>"
        "<div class='welcome-subtitle'>Solar PV Decision Platform</div>"
        f"<div class='welcome-mission'>{mission}</div>"
        "<div class='welcome-actions'>"
        "<a class='welcome-btn secondary' href='?page=about'>Explore Story</a>"
        "<a class='welcome-btn' href='?page=platform'>Enter Platform</a>"
        "</div>"
        "<div class='welcome-hint'>Tip: Click anywhere or press any key to enter the Platform.</div>"
        "</div></div>"
    )

    st.markdown(welcome_html, unsafe_allow_html=True)

    st.components.v1.html(
        """
<script>
function goPlatform(){
  const url = new URL(window.location.href);
  url.searchParams.set("page","platform");
  window.location.href = url.toString();
}
document.addEventListener("keydown", function(){ goPlatform(); }, {once:true});
document.addEventListener("click", function(){ goPlatform(); }, {once:true});
</script>
        """,
        height=0,
    )


# =========================
# Sidebar Navigation (always visible after landing)
# =========================
def sidebar_navigation(current_page: str) -> str:
    page_map = {
        "Welcome / Landing": "welcome",
        "About GreenWise": "about",
        "Decision Platform": "platform",
    }
    inv_map = {v: k for k, v in page_map.items()}
    default_label = inv_map.get(current_page, "Decision Platform")

    with st.sidebar:
        st.markdown("## GreenWise")
        choice = st.radio(
            "Navigation",
            list(page_map.keys()),
            index=list(page_map.keys()).index(default_label),
            key="gw_nav",
        )
        st.caption("Switch pages anytime. On **Decision Platform**, inputs appear below.")
        st.divider()

    return page_map[choice]


# =========================
# About Page (Story Page)  ← NEW
# =========================
def show_about_page():
    st.markdown(
        dedent(
            """
<div style="padding: 18px 18px; border-radius: 20px;
            background: rgba(0,0,0,0.32);
            border: 1px solid rgba(255,255,255,0.14);
            backdrop-filter: blur(12px);">
  <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
    <div class="gw-chip">About GreenWise</div>
  </div>
  <div style="font-size: 40px; font-weight: 900; letter-spacing: 0.2px; margin-top: 6px;">
    Story · Feasibility · Impact
  </div>
  <div style="margin-top: 10px; color: rgba(255,255,255,0.82); font-size: 15px; line-height: 1.6;">
    GreenWise is designed for early-stage decision-making: instead of complex engineering tools or spreadsheets for experts,
    it helps managers answer one question clearly — <b>“Is rooftop solar worth investing in for my company?”</b>
  </div>
</div>
"""
        ),
        unsafe_allow_html=True,
    )
    st.write("")

    # 3 cards
    c1, c2, c3 = st.columns(3)
    card_style = """
<div style="padding:16px 16px;border-radius:18px;background:rgba(0,0,0,0.30);
            border:1px solid rgba(255,255,255,0.14);backdrop-filter:blur(10px);min-height:170px;">
  <div class="gw-chip">{tag}</div>
  <div style="font-size:18px;font-weight:900;margin-top:8px;">{title}</div>
  <div style="margin-top:8px;color:rgba(255,255,255,0.82);font-size:14px;line-height:1.6;">{body}</div>
</div>
"""
    with c1:
        st.markdown(
            card_style.format(
                tag="The Problem",
                title="Uncertainty blocks action",
                body=(
                    "Enterprises face rising pressure to reduce emissions, and rooftop solar is attractive. "
                    "Yet adoption is slow because decision-makers lack clarity: payback is uncertain, risks are hard to quantify, "
                    "and existing tools are too technical."
                ),
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            card_style.format(
                tag="Our Solution",
                title="Fast decision support",
                body=(
                    "Users input basic information (demand, roof, price, cost assumptions). "
                    "GreenWise evaluates feasibility and outputs investment cost, savings, payback/ROI, and CO₂ reduction — "
                    "focusing on speed, transparency, and manager-friendly communication."
                ),
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            card_style.format(
                tag="Why It Matters",
                title="From goals to real investments",
                body=(
                    "By reducing financial uncertainty, GreenWise helps enterprises make faster, more confident green investment decisions, "
                    "accelerating rooftop solar adoption and contributing directly to carbon emission reduction goals."
                ),
            ),
            unsafe_allow_html=True,
        )

    st.write("")

    st.markdown(
        dedent(
            """
<div style="padding:16px 16px;border-radius:18px;background:rgba(0,0,0,0.30);
            border:1px solid rgba(255,255,255,0.14);backdrop-filter:blur(10px);">
  <div class="gw-chip">Key Innovations</div>
  <ul style="margin-top:10px;color:rgba(255,255,255,0.86);font-size:14px;line-height:1.7;">
    <li><b>Automatic capacity optimization</b>: evaluates a range of feasible PV sizes and recommends the best capacity, instead of a few fixed scenarios.</li>
    <li><b>Explainable ML (XGBoost + SHAP)</b>: reveals which inputs most influence payback, so users understand why a recommendation is made.</li>
    <li><b>Decision-first design</b>: outputs manager-facing metrics (payback/ROI/cashflow/CO₂), with clear exportable results.</li>
  </ul>
</div>
"""
        ),
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        dedent(
            """
<div style="padding:16px 16px;border-radius:18px;background:rgba(0,0,0,0.30);
            border:1px solid rgba(255,255,255,0.14);backdrop-filter:blur(10px);">
  <div class="gw-chip">Website Requirements Alignment</div>
  <div style="margin-top:10px;color:rgba(255,255,255,0.82);font-size:14px;line-height:1.7;">
    This website is structured to meet innovation-competition expectations:
    it <b>tells our story</b> (problem → solution → impact), <b>shows the product prototype</b> (Decision Platform),
    <b>builds a consistent brand</b> (name/logo/visual style), and makes the project easy to understand for both the public and judges.
  </div>
</div>
"""
        ),
        unsafe_allow_html=True,
    )

    st.write("")

    # Bottom nav buttons
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Go to Decision Platform", key="about_to_platform", use_container_width=True):
            st.session_state.gw_page = "platform"
            do_rerun()
    with b2:
        if st.button("Back to Welcome", key="about_to_welcome", use_container_width=True):
            st.session_state.gw_page = "welcome"
            do_rerun()


# =========================
# Decision Platform Page (your original core)
# =========================
def show_platform_page():
    st.markdown(
        dedent(
            """
<div style="padding: 14px 18px; border-radius: 18px;
            background: rgba(0,0,0,0.28);
            border: 1px solid rgba(255,255,255,0.14);
            backdrop-filter: blur(10px);">
  <div style="font-size: 30px; font-weight: 900; letter-spacing: 0.2px;">
    GreenWise · Solar PV Decision Platform
  </div>
  <div style="margin-top: 6px; color: rgba(255,255,255,0.78); font-size: 14px;">
    Input enterprise demand & rooftop conditions → Auto-recommend optimal PV capacity → Output payback/ROI/CO₂ reduction, with explainable AI (XGBoost + SHAP).
  </div>
</div>
"""
        ),
        unsafe_allow_html=True,
    )
    st.write("")

    # Sidebar Inputs
    with st.sidebar:
        st.header("Inputs")

        annual_consumption_kwh = st.slider("Annual consumption (kWh/year)", 0, 5_000_000, 300_000, step=10_000)
        price_y_per_kwh = st.slider("Electricity price (¥/kWh)", 0.00, 2.50, 0.95, step=0.01)

        roof_area_m2 = st.slider("Available roof area (m²)", 0, 50_000, 800, step=50)
        shading_level = st.selectbox("Shading level", list(SHADING_MAP.keys()), index=2)

        st.divider()
        st.subheader("PV & Finance")
        capex_y_per_kw = st.slider("CAPEX (¥/kW)", 0, 20_000, 3800, step=50)
        opex_pct = st.slider("OPEX (% of CAPEX per year)", 0.0, 5.0, 1.0, step=0.1)
        self_use_ratio = st.slider("Self-consumption ratio", 0.10, 1.00, 0.80, step=0.05)

        st.divider()
        st.subheader("Carbon & Assumptions")
        co2_factor_kg_per_kwh = st.slider("Grid CO₂ factor (kg/kWh)", 0.00, 1.20, 0.55, step=0.01)
        degradation_pct = st.slider("PV degradation (%/year)", 0.0, 3.0, 0.5, step=0.1)
        discount_rate_pct = st.slider("Discount rate (%/year)", 0.0, 15.0, 6.0, step=0.5)

        st.divider()
        st.subheader("Engineering Defaults")
        kw_per_m2 = st.slider("kW per m²", 0.05, 0.35, 0.20, step=0.01)
        kwh_per_kw_year = st.slider("Yield (kWh/kW·year)", 200, 2000, 1100, step=50)
        export_price_y_per_kwh = st.slider("Export price (¥/kWh) [optional]", 0.00, 1.50, 0.00, step=0.01)

        st.divider()
        st.subheader("Auto-Optimization Objective")
        objective_label = st.selectbox(
            "Choose objective",
            ["Minimize payback (simple)", "Maximize ROI (Year 1)", "Minimize discounted payback"],
            index=0,
        )
        objective_key = {
            "Minimize payback (simple)": "min_payback",
            "Maximize ROI (Year 1)": "max_roi",
            "Minimize discounted payback": "min_discounted_payback",
        }[objective_label]

    # Auto optimize + optional manual capacity
    kw_max_by_roof = roof_area_m2 * kw_per_m2

    best, curve_df = find_optimal_capacity(
        kw_max=kw_max_by_roof,
        step_kw=2.0,
        objective=objective_key,
        annual_consumption_kwh=float(annual_consumption_kwh),
        price_y_per_kwh=float(price_y_per_kwh),
        roof_area_m2=float(roof_area_m2),
        shading_level=shading_level,
        capex_y_per_kw=float(capex_y_per_kw),
        opex_pct=float(opex_pct),
        self_use_ratio=float(self_use_ratio),
        co2_factor_kg_per_kwh=float(co2_factor_kg_per_kwh),
        degradation_pct=float(degradation_pct),
        discount_rate_pct=float(discount_rate_pct),
        kw_per_m2=float(kw_per_m2),
        kwh_per_kw_year=float(kwh_per_kw_year),
        export_price_y_per_kwh=float(export_price_y_per_kwh),
    )

    if best is None or curve_df.empty:
        st.error("No feasible scenario under current inputs. Please adjust assumptions and try again.")
        st.stop()

    recommended_kw = float(best["kW"])

    use_manual = st.toggle("Use manual capacity (override recommendation)", value=False)
    if use_manual:
        manual_kw = st.slider("Manual PV capacity (kW)", 0.0, float(kw_max_by_roof), float(recommended_kw), step=1.0)
        chosen_kw = float(manual_kw)
    else:
        chosen_kw = float(recommended_kw)

    res = pv_estimate(
        annual_consumption_kwh=float(annual_consumption_kwh),
        price_y_per_kwh=float(price_y_per_kwh),
        roof_area_m2=float(roof_area_m2),
        shading_level=shading_level,
        capex_y_per_kw=float(capex_y_per_kw),
        opex_pct=float(opex_pct),
        self_use_ratio=float(self_use_ratio),
        co2_factor_kg_per_kwh=float(co2_factor_kg_per_kwh),
        degradation_pct=float(degradation_pct),
        discount_rate_pct=float(discount_rate_pct),
        pv_kw=float(chosen_kw),
        kw_per_m2=float(kw_per_m2),
        kwh_per_kw_year=float(kwh_per_kw_year),
        export_price_y_per_kwh=float(export_price_y_per_kwh),
    )

    tag, msg = recommendation(res["simple_payback_y"], res["roi_y1"])

    # Main Layout: Metrics
    colA, colB, colC = st.columns([1.25, 1, 1])

    with colA:
        st.subheader("Decision Summary")
        st.metric("Decision", tag)
        st.write(msg)
        st.caption(
            f"Rooftop capacity upper bound ≈ **{res['kw_max_by_roof']:.1f} kW** (area × kW/m²). "
            f"Shading factor = **{res['shading_factor']:.2f}**."
        )
        if use_manual:
            st.info(f"Manual capacity selected: **{res['pv_kw']:.0f} kW** (Recommendation was {recommended_kw:.0f} kW).")
        else:
            st.success(f"Auto-recommended capacity: **{res['pv_kw']:.0f} kW**")

    with colB:
        st.subheader("Key Metrics (Year 1)")
        st.metric("PV Capacity (kW)", f"{res['pv_kw']:.0f}")
        st.metric("Generation (kWh)", f"{res['gen_kwh_y1']:.0f}")
        st.metric("Savings (¥)", money(res["annual_savings_y1"]))
        st.metric("CO₂ reduction (t)", f"{res['co2_ton_y1']:.1f}")

    with colC:
        st.subheader("Finance")
        st.metric("CAPEX (¥)", money(res["capex"]))
        st.metric("Net cashflow Y1 (¥)", money(res["net_cash_y1"]))
        st.metric("Simple payback (years)", fmt_year(res["simple_payback_y"]))
        st.metric(
            "Discounted payback (years)",
            str(res["discounted_payback_y"]) if res["discounted_payback_y"] != math.inf else "∞",
        )

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Curves & Scenarios", "💸 20-Year Cashflows", "🧠 Explainability (SHAP)", "⬇️ Export"])

    # Tab 1: curves
    with tab1:
        left, right = st.columns([1.15, 1])

        with left:
            st.subheader("Capacity Sweep")
            metric_choice = st.selectbox("Metric", ["Payback(y)", "ROI(Y1)", "CO2(t/y)", "NetCashY1(¥)"], index=0)

            plot_src = curve_df.copy()
            plot_src = plot_src.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric_choice])

            if metric_choice == "ROI(Y1)":
                plot_src["ROI_pct"] = plot_src["ROI(Y1)"] * 100.0
                y_field = "ROI_pct:Q"
                y_title = "ROI (Year 1, %)"
            else:
                y_field = f"{metric_choice}:Q"
                y_title = metric_choice

            marker_df = pd.DataFrame({"kW": [float(res["pv_kw"])]})
            chart = (
                alt.Chart(plot_src)
                .mark_line(point=True)
                .encode(
                    x=alt.X("kW:Q", title="PV capacity (kW)"),
                    y=alt.Y(y_field, title=y_title),
                    tooltip=[
                        alt.Tooltip("kW:Q", format=".0f"),
                        alt.Tooltip("Payback(y):Q", format=".2f"),
                        alt.Tooltip("ROI(Y1):Q", format=".3f"),
                        alt.Tooltip("CO2(t/y):Q", format=".2f"),
                        alt.Tooltip("NetCashY1(¥):Q", format=".0f"),
                    ],
                )
                .properties(height=320)
                .interactive()
            )
            rule = alt.Chart(marker_df).mark_rule(strokeDash=[6, 6]).encode(x="kW:Q")
            st.altair_chart(chart + rule, width="stretch")

        with right:
            st.subheader("Scenario Snapshot (near chosen capacity)")
            df_disp = curve_df.copy()
            df_disp["dist"] = (df_disp["kW"] - float(res["pv_kw"])).abs()
            df_disp = df_disp.sort_values("dist").head(14).drop(columns=["dist"])

            df_disp2 = df_disp.copy()
            df_disp2["ROI(Y1)"] = (df_disp2["ROI(Y1)"] * 100).round(1).astype(str) + "%"
            df_disp2["Payback(y)"] = df_disp2["Payback(y)"].round(2)
            df_disp2["CO2(t/y)"] = df_disp2["CO2(t/y)"].round(1)
            df_disp2["NetCashY1(¥)"] = df_disp2["NetCashY1(¥)"].round(0).astype(int)

            st.dataframe(df_disp2, width="stretch", height=330)

    # Tab 2: cashflows
    with tab2:
        st.subheader("20-Year Cashflow & Discounted Cumulative")

        cash_df = res["cashflow_df"].copy()
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("**Annual Net Cashflow (¥)**")
            st.line_chart(cash_df.set_index("Year")[["NetCash_Y"]], width="stretch")

        with c2:
            st.markdown("**Cumulative Discounted Cashflow (¥)**")
            st.line_chart(cash_df.set_index("Year")[["CumDiscounted_Y"]], width="stretch")

        st.dataframe(cash_df.round(2), width="stretch", height=360)

    # Tab 3: SHAP
    with tab3:
        st.subheader("Explainability (XGBoost + SHAP)")

        model = ensure_demo_model() if HAS_ML else None

        if model is None:
            st.info("To enable SHAP visuals, install: xgboost + shap + joblib + matplotlib.")
        else:
            X_input = (
                pd.DataFrame(
                    [
                        {
                            "annual_consumption_kwh": float(annual_consumption_kwh),
                            "price_y_per_kwh": float(price_y_per_kwh),
                            "roof_area_m2": float(roof_area_m2),
                            "shading_factor": float(res["shading_factor"]),
                            "capex_y_per_kw": float(capex_y_per_kw),
                            "self_use_ratio": float(self_use_ratio),
                            "pv_kw": float(res["pv_kw"]),
                        }
                    ]
                )[FEATURE_COLS]
                .astype(float)
            )

            try:
                pred = float(np.array(model.predict(X_input)).reshape(-1)[0])
                st.metric("Model-predicted payback (years)", f"{pred:.2f}")
            except Exception:
                st.metric("Model-predicted payback (years)", "—")

            st.markdown("### Single-case SHAP explanation")
            plot_type = st.radio("Plot type", ["Bar", "Waterfall"], horizontal=True)

            if st.button("Compute SHAP for current case", key="btn_shap"):
                try:
                    fig, sv_row = shap_single_plots(model, X_input, plot_type)
                    st.pyplot(fig, clear_figure=True)
                    st.info(shap_text_explanation(X_input, sv_row))
                except Exception as e:
                    st.error("SHAP failed in your environment. Below is the real error:")
                    st.exception(e)

    # Tab 4: export
    with tab4:
        st.subheader("Export results")

        summary = pd.DataFrame(
            [
                {
                    "PV_kW": res["pv_kw"],
                    "Payback_y": res["simple_payback_y"],
                    "DiscPayback_y": res["discounted_payback_y"],
                    "ROI_Y1": res["roi_y1"],
                    "CAPEX_Y": res["capex"],
                    "NetCash_Y1": res["net_cash_y1"],
                    "Gen_kWh_Y1": res["gen_kwh_y1"],
                    "CO2_ton_Y1": res["co2_ton_y1"],
                    "SelfUsed_kWh_Y1": res["self_used_kwh_y1"],
                    "Exported_kWh_Y1": res["exported_kwh_y1"],
                }
            ]
        )

        st.markdown("<div class='gw-chip'>Summary (CSV)</div>", unsafe_allow_html=True)
        st.dataframe(summary.round(4), width="stretch")
        st.download_button(
            "Download Summary CSV",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="greenwise_summary.csv",
            mime="text/csv",
            key="dl_summary",
        )

        st.markdown("<div class='gw-chip'>Capacity Sweep (CSV)</div>", unsafe_allow_html=True)
        st.dataframe(curve_df.head(30), width="stretch")
        st.download_button(
            "Download Capacity Sweep CSV",
            data=curve_df.to_csv(index=False).encode("utf-8"),
            file_name="greenwise_capacity_sweep.csv",
            mime="text/csv",
            key="dl_capacity",
        )

        st.markdown("<div class='gw-chip'>20-Year Cashflow (CSV)</div>", unsafe_allow_html=True)
        st.dataframe(res["cashflow_df"].head(30), width="stretch")
        st.download_button(
            "Download Cashflow CSV",
            data=res["cashflow_df"].to_csv(index=False).encode("utf-8"),
            file_name="greenwise_cashflow_20y.csv",
            mime="text/csv",
            key="dl_cashflow",
        )

    st.divider()

    st.subheader("Website Copy")
    st.markdown(
        """
GreenWise is a web-based decision support platform that helps enterprises evaluate the financial and environmental feasibility of rooftop solar PV investments.

**What it provides**  
- Auto-recommended PV capacity (optimization over feasible rooftop capacity)  
- Payback period, ROI, annual savings, and a 20-year discounted cashflow view  
- Annual CO₂ emissions reduction estimates  
- Explainable AI (XGBoost + SHAP) to reveal transparent decision drivers
"""
    )


# =========================
# Page Router (Welcome / About / Platform)
# =========================
params = safe_get_query_params()

if "gw_page" not in st.session_state:
    st.session_state.gw_page = "welcome"

# Query param routing: ?page=welcome/about/platform
page_val = params.get("page", None)
if isinstance(page_val, list):
    page_val = page_val[0] if page_val else None
if page_val in {"welcome", "about", "platform"}:
    st.session_state.gw_page = page_val
    clear_query_params()

# Landing page
if st.session_state.gw_page == "welcome":
    show_welcome_screen()
    st.stop()

# App pages
inject_global_css()

# Sidebar nav always visible
next_page = sidebar_navigation(st.session_state.gw_page)
if next_page != st.session_state.gw_page:
    st.session_state.gw_page = next_page
    do_rerun()

# Render page
if st.session_state.gw_page == "about":
    show_about_page()
else:
    show_platform_page()
