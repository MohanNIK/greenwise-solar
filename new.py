# app.py
# GreenWise: Solar PV Feasibility & ROI Decision Platform (Web Prototype)
# Run:
#   pip install streamlit pandas numpy xgboost shap joblib matplotlib
#   streamlit run app.py
#
# Notes:
# - If you DON'T have a trained model yet, the app will still run (formula engine + auto-optimization).
# - If you DO have a trained model, put "xgb_payback.pkl" in the same folder to enable XGBoost+SHAP.

import math
import os
import numpy as np
import pandas as pd
import streamlit as st

# Optional ML
HAS_ML = True
try:
    import joblib
    import xgboost as xgb  # noqa: F401
    import shap
    import matplotlib.pyplot as plt
except Exception:
    HAS_ML = False

st.set_page_config(page_title="GreenWise · Solar PV Decision Platform", layout="wide")


# =========================
# Creative Background (CSS)
# =========================
def inject_creative_background():
    st.markdown(
        """
        <style>
        /* --- Page base --- */
        .stApp {
            background:
                radial-gradient(1200px 800px at 15% 20%, rgba(120, 255, 214, 0.18), transparent 60%),
                radial-gradient(900px 700px at 85% 30%, rgba(120, 170, 255, 0.20), transparent 55%),
                radial-gradient(900px 700px at 30% 85%, rgba(255, 230, 120, 0.14), transparent 60%),
                linear-gradient(180deg, rgba(10, 14, 28, 1) 0%, rgba(8, 12, 22, 1) 55%, rgba(6, 10, 18, 1) 100%);
        }

        /* --- Subtle animated grid --- */
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            background-image:
                linear-gradient(rgba(255,255,255,0.06) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.06) 1px, transparent 1px);
            background-size: 48px 48px;
            opacity: 0.10;
            mask-image: radial-gradient(circle at 30% 20%, black 0%, transparent 55%);
            animation: gridFloat 10s ease-in-out infinite alternate;
            pointer-events: none;
            z-index: 0;
        }
        @keyframes gridFloat {
            from { transform: translate3d(0,0,0); }
            to   { transform: translate3d(-20px, 12px, 0); }
        }

        /* --- Floating "solar orbs" --- */
        .stApp::after{
            content:"";
            position: fixed;
            inset: -20%;
            background:
              radial-gradient(circle at 20% 30%, rgba(255, 198, 92, 0.25) 0%, transparent 35%),
              radial-gradient(circle at 70% 20%, rgba(92, 160, 255, 0.22) 0%, transparent 38%),
              radial-gradient(circle at 65% 75%, rgba(92, 255, 210, 0.18) 0%, transparent 40%);
            filter: blur(8px);
            animation: orbs 14s ease-in-out infinite alternate;
            pointer-events:none;
            z-index: 0;
        }
        @keyframes orbs{
          from{ transform: translate3d(0,0,0) scale(1); opacity: 0.85;}
          to  { transform: translate3d(18px,-10px,0) scale(1.06); opacity: 1;}
        }

        /* --- Make content appear above background layers --- */
        section[data-testid="stSidebar"], main, header, footer {
            position: relative;
            z-index: 1;
        }

        /* --- Cards / containers polish --- */
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 16px;
            padding: 12px 14px;
            backdrop-filter: blur(6px);
        }
        .block-container {
            padding-top: 1.2rem;
        }
        h1, h2, h3, p, label, span, div {
            color: rgba(255,255,255,0.92);
        }
        .stCaption, .stMarkdown small {
            color: rgba(255,255,255,0.70) !important;
        }

        /* --- Sidebar --- */
        section[data-testid="stSidebar"] {
            background: rgba(255,255,255,0.04);
            border-right: 1px solid rgba(255,255,255,0.08);
            backdrop-filter: blur(10px);
        }
        </style>
        """,
        unsafe_allow_html=True
    )


inject_creative_background()


# =========================
# Helper Functions
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
    kw_per_m2: float = 0.20,          # ~200W/m2 (rough)
    kwh_per_kw_year: float = 1100.0,  # region-dependent
):
    shading_factor = {"低": 0.95, "中": 0.85, "高": 0.70}[shading_level]

    # Capacity upper bound by roof area
    kw_max_by_roof = max(0.0, roof_area_m2 * kw_per_m2)
    pv_kw = float(min(max(0.0, pv_kw), kw_max_by_roof))

    # Annual generation (year 1)
    gen_kwh_y1 = pv_kw * kwh_per_kw_year * shading_factor

    # Self-consumed energy capped by demand
    self_used_kwh_y1 = min(gen_kwh_y1 * self_use_ratio, annual_consumption_kwh)
    exported_kwh_y1 = max(0.0, gen_kwh_y1 - self_used_kwh_y1)

    # MVP: export revenue ignored
    export_price = 0.0
    annual_savings_y1 = self_used_kwh_y1 * price_y_per_kwh + exported_kwh_y1 * export_price

    capex = pv_kw * capex_y_per_kw
    opex_y1 = capex * (opex_pct / 100.0)

    net_cash_y1 = annual_savings_y1 - opex_y1
    simple_payback = (capex / net_cash_y1) if net_cash_y1 > 0 else math.inf
    roi_y1 = (net_cash_y1 / capex) if capex > 0 else 0.0

    # Discounted payback (20-year horizon), with degradation
    disc = discount_rate_pct / 100.0
    degr = degradation_pct / 100.0
    discounted_payback = math.inf
    cum = -capex
    for year in range(1, 21):
        gen = gen_kwh_y1 * ((1 - degr) ** (year - 1))
        self_used = min(gen * self_use_ratio, annual_consumption_kwh)
        exported = max(0.0, gen - self_used)
        savings = self_used * price_y_per_kwh + exported * export_price
        opex = opex_y1
        net = savings - opex
        pv_net = net / ((1 + disc) ** year) if disc > 0 else net
        cum += pv_net
        if cum >= 0 and discounted_payback == math.inf:
            discounted_payback = year

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
        "self_used_kwh_y1": self_used_kwh_y1
    }


def fmt_year(x):
    if x == math.inf or (isinstance(x, float) and not np.isfinite(x)):
        return "∞"
    return f"{x:.1f}"


def recommendation(payback_y, roi_y1):
    if payback_y == math.inf:
        return ("Not recommended", "当前参数下现金流为负或过低：建议优先降本、提升自用比例、或在更高电价/更好日照条件下评估。")
    if payback_y <= 4 and roi_y1 >= 0.18:
        return ("Strongly recommended", "回本快、收益率高：具备较强可行性，适合推进可研与实施计划。")
    if payback_y <= 7:
        return ("Recommended", "具备可行性：可通过优化装机规模、提升自用比例、降低CAPEX进一步提升回报。")
    return ("Borderline", "回本偏慢：建议重新评估装机规模或关键参数后再决策。")


def find_optimal_capacity(
    kw_max: float,
    step_kw: float,
    objective: str,
    **kwargs
):
    """
    objective:
      - "min_payback": minimize simple payback
      - "max_roi": maximize ROI in year1
      - "min_discounted_payback": minimize discounted payback (integer or inf)
    """
    records = []
    if kw_max <= 0:
        return None, pd.DataFrame()

    # Start at step_kw to avoid 0
    for kw in np.arange(step_kw, kw_max + 1e-9, step_kw):
        r = pv_estimate(pv_kw=float(kw), **kwargs)
        records.append({
            "kW": float(kw),
            "Payback(y)": r["simple_payback_y"],
            "ROI(Y1)": r["roi_y1"],
            "DiscPayback(y)": r["discounted_payback_y"],
            "CO2(t/y)": r["co2_ton_y1"],
            "NetCashY1(¥)": r["net_cash_y1"],
        })

    df = pd.DataFrame(records)
    # Remove infeasible rows (inf payback or negative cash)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Payback(y)", "ROI(Y1)"])
    if df.empty:
        return None, df

    if objective == "max_roi":
        best = df.loc[df["ROI(Y1)"].idxmax()]
    elif objective == "min_discounted_payback":
        # DiscPayback can be inf; drop inf
        df2 = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["DiscPayback(y)"])
        if df2.empty:
            best = df.loc[df["Payback(y)"].idxmin()]
        else:
            best = df2.loc[df2["DiscPayback(y)"].idxmin()]
    else:
        best = df.loc[df["Payback(y)"].idxmin()]

    return best, df


def try_load_model(path="xgb_payback.pkl"):
    if not HAS_ML:
        return None
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def shap_bar_plot(explainer, shap_values, X_input: pd.DataFrame):
    # Create an Explanation object for modern SHAP plotting
    exp = shap.Explanation(
        values=shap_values[0] if hasattr(shap_values, "__len__") else shap_values,
        base_values=getattr(explainer, "expected_value", None),
        data=X_input.values[0],
        feature_names=list(X_input.columns),
    )
    fig = plt.figure()
    shap.plots.bar(exp, show=False)
    return fig


# =========================
# UI: Header
# =========================
st.markdown(
    """
    <div style="padding: 14px 18px; border-radius: 18px;
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.10);
                backdrop-filter: blur(8px);">
      <div style="font-size: 32px; font-weight: 800; letter-spacing: 0.2px;">
        GreenWise · Solar PV Decision Platform
      </div>
      <div style="margin-top: 6px; color: rgba(255,255,255,0.75); font-size: 14px;">
        输入企业用电与屋顶条件 → 自动推荐最优装机规模 → 输出回收期/ROI/减排，并可选 XGBoost+SHAP 解释关键驱动因素。
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")


# =========================
# Sidebar Inputs
# =========================
with st.sidebar:
    st.header("Inputs")

    annual_consumption_kwh = st.number_input("Annual consumption (kWh/year)", min_value=0.0, value=300000.0, step=10000.0)
    price_y_per_kwh = st.number_input("Electricity price (¥/kWh)", min_value=0.0, value=0.95, step=0.01)

    roof_area_m2 = st.number_input("Available roof area (m²)", min_value=0.0, value=800.0, step=50.0)
    shading_level = st.selectbox("Shading level", ["低", "中", "高"], index=1)

    st.divider()
    st.subheader("PV & Finance")
    capex_y_per_kw = st.number_input("CAPEX (¥/kW)", min_value=0.0, value=3800.0, step=100.0)
    opex_pct = st.number_input("OPEX (% of CAPEX per year)", min_value=0.0, value=1.0, step=0.1)
    self_use_ratio = st.slider("Self-consumption ratio", min_value=0.1, max_value=1.0, value=0.8, step=0.05)

    st.divider()
    st.subheader("Carbon & Assumptions")
    co2_factor_kg_per_kwh = st.number_input("Grid CO₂ factor (kg/kWh)", min_value=0.0, value=0.55, step=0.01)
    degradation_pct = st.number_input("PV degradation (%/year)", min_value=0.0, value=0.5, step=0.1)
    discount_rate_pct = st.number_input("Discount rate (%/year)", min_value=0.0, value=6.0, step=0.5)

    st.divider()
    st.subheader("Engineering Defaults")
    kw_per_m2 = st.number_input("kW per m²", min_value=0.05, value=0.20, step=0.01)
    kwh_per_kw_year = st.number_input("Yield (kWh/kW·year)", min_value=200.0, value=1100.0, step=50.0)

    st.divider()
    st.subheader("Auto-Optimization Objective")
    objective = st.selectbox(
        "Choose objective",
        [
            "Minimize payback (simple)",
            "Maximize ROI (Year 1)",
            "Minimize discounted payback"
        ],
        index=0
    )

    st.caption("Tip: 比赛演示建议用 Minimize payback（最好讲）。")


# =========================
# Auto-Optimize PV Capacity
# =========================
kw_max_by_roof = roof_area_m2 * kw_per_m2

objective_key = {
    "Minimize payback (simple)": "min_payback",
    "Maximize ROI (Year 1)": "max_roi",
    "Minimize discounted payback": "min_discounted_payback"
}[objective]

best, curve_df = find_optimal_capacity(
    kw_max=kw_max_by_roof,
    step_kw=2.0,
    objective=objective_key,
    annual_consumption_kwh=annual_consumption_kwh,
    price_y_per_kwh=price_y_per_kwh,
    roof_area_m2=roof_area_m2,
    shading_level=shading_level,
    capex_y_per_kw=capex_y_per_kw,
    opex_pct=opex_pct,
    self_use_ratio=self_use_ratio,
    co2_factor_kg_per_kwh=co2_factor_kg_per_kwh,
    degradation_pct=degradation_pct,
    discount_rate_pct=discount_rate_pct,
    kw_per_m2=kw_per_m2,
    kwh_per_kw_year=kwh_per_kw_year,
)

if best is None:
    st.error("当前屋顶可用面积或参数导致无法形成可行方案（净现金流为负/无法计算）。请调整输入后重试。")
    st.stop()

recommended_kw = float(best["kW"])

# Compute detailed metrics at recommended kW
res = pv_estimate(
    annual_consumption_kwh=annual_consumption_kwh,
    price_y_per_kwh=price_y_per_kwh,
    roof_area_m2=roof_area_m2,
    shading_level=shading_level,
    capex_y_per_kw=capex_y_per_kw,
    opex_pct=opex_pct,
    self_use_ratio=self_use_ratio,
    co2_factor_kg_per_kwh=co2_factor_kg_per_kwh,
    degradation_pct=degradation_pct,
    discount_rate_pct=discount_rate_pct,
    pv_kw=recommended_kw,
    kw_per_m2=kw_per_m2,
    kwh_per_kw_year=kwh_per_kw_year,
)

tag, msg = recommendation(res["simple_payback_y"], res["roi_y1"])


# =========================
# Main Layout
# =========================
colA, colB, colC = st.columns([1.25, 1, 1])

with colA:
    st.subheader("Auto Recommendation")
    st.metric("Decision", tag)
    st.write(msg)
    st.caption(
        f"Roof capacity upper bound ≈ **{res['kw_max_by_roof']:.1f} kW** (area × kW/m²). "
        f"Shading factor = **{res['shading_factor']:.2f}**."
    )

with colB:
    st.subheader("Key Metrics (Year 1)")
    st.metric("Recommended PV (kW)", f"{res['pv_kw']:.0f}")
    st.metric("Generation (kWh)", f"{res['gen_kwh_y1']:.0f}")
    st.metric("Savings (¥)", f"{res['annual_savings_y1']:.0f}")
    st.metric("CO₂ reduction (t)", f"{res['co2_ton_y1']:.1f}")

with colC:
    st.subheader("Finance")
    st.metric("CAPEX (¥)", f"{res['capex']:.0f}")
    st.metric("Net cashflow Y1 (¥)", f"{res['net_cash_y1']:.0f}")
    st.metric("Simple payback (years)", fmt_year(res["simple_payback_y"]))
    st.metric("Discounted payback (years)", str(res["discounted_payback_y"]) if res["discounted_payback_y"] != math.inf else "∞")

st.divider()


# =========================
# Curves + Table
# =========================
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Capacity Sweep Curve")
    curve_plot = curve_df.copy()
    curve_plot = curve_plot.replace([np.inf, -np.inf], np.nan).dropna(subset=["Payback(y)"])
    curve_plot = curve_plot.set_index("kW")[["Payback(y)"]]
    st.line_chart(curve_plot, width="stretch")

    st.caption("这条曲线展示：装机规模变化 → 回收期变化（用于解释为何推荐该装机）。")

with right:
    st.subheader("Scenario Snapshot")
    df_disp = curve_df.copy()

    # Reduce for display: show a few rows near the recommended kW
    df_disp["dist"] = (df_disp["kW"] - recommended_kw).abs()
    df_disp = df_disp.sort_values("dist").head(12).drop(columns=["dist"])

    df_disp2 = df_disp.copy()
    df_disp2["ROI(Y1)"] = (df_disp2["ROI(Y1)"] * 100).round(1).astype(str) + "%"
    df_disp2["Payback(y)"] = df_disp2["Payback(y)"].apply(lambda x: float(x) if np.isfinite(x) else np.nan)
    df_disp2["CO2(t/y)"] = df_disp2["CO2(t/y)"].round(1)
    st.dataframe(df_disp2, width="stretch")

st.divider()


# =========================
# Explainability: XGBoost + SHAP (Real Hook)
# =========================
st.subheader("Explainability (XGBoost + SHAP)")

model = try_load_model("xgb_payback.pkl")

if not HAS_ML:
    st.warning("当前环境未安装 xgboost/shap/matplotlib/joblib。先运行：pip install xgboost shap joblib matplotlib")
elif model is None:
    st.info("未检测到 xgb_payback.pkl（模型文件）。你可以先用本页面完成 Demo；后续把模型放到同目录即可自动启用 SHAP 解释。")
else:
    st.success("已加载模型：xgb_payback.pkl（SHAP 解释已启用）")

    # Features (data structure)
    X_input = pd.DataFrame([{
        "annual_consumption_kwh": annual_consumption_kwh,
        "price_y_per_kwh": price_y_per_kwh,
        "roof_area_m2": roof_area_m2,
        "shading_factor": res["shading_factor"],
        "capex_y_per_kw": capex_y_per_kw,
        "self_use_ratio": self_use_ratio,
        "pv_kw": res["pv_kw"],
    }])

    try:
        pred_payback = float(model.predict(X_input)[0])
        st.metric("Model-predicted payback (years)", f"{pred_payback:.2f}")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)

        fig = shap_bar_plot(explainer, shap_values, X_input)
        st.pyplot(fig, clear_figure=True)

        st.caption("SHAP 展示：哪些因素把回收期推短/推长（可解释 AI 决策支持的核心亮点）。")

    except Exception as e:
        st.error(f"SHAP 解释执行失败：{e}")
        st.info("建议先确认模型训练时的特征列名与当前 X_input 完全一致。")


with st.expander("Model interface: how to train & save xgb_payback.pkl (template)"):
    st.code(
        """
# train_model.py (template)
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

# Suppose you prepared a dataset df with the following columns:
# features = ["annual_consumption_kwh","price_y_per_kwh","roof_area_m2","shading_factor","capex_y_per_kw","self_use_ratio","pv_kw"]
# target   = "payback_period"

features = ["annual_consumption_kwh","price_y_per_kwh","roof_area_m2","shading_factor","capex_y_per_kw","self_use_ratio","pv_kw"]

df = pd.read_csv("train_data.csv")  # your data
X = df[features]
y = df["payback_period"]

model = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.9,
    random_state=42
)
model.fit(X, y)

joblib.dump(model, "xgb_payback.pkl")
print("Saved: xgb_payback.pkl")
        """,
        language="python"
    )

st.divider()


# =========================
# Website-ready Summary (English)
# =========================
st.subheader("Website Copy (English)")
st.markdown(
    """
**One-liner**  
GreenWise is a web-based decision support platform that helps enterprises evaluate the financial and environmental feasibility of rooftop solar PV investments.

**What it provides**  
- Recommended PV capacity (auto-optimized)  
- Payback period, ROI, and annual savings  
- Annual CO₂ emissions reduction  
- Transparent explanations (XGBoost + SHAP)

**Why it matters**  
Many companies hesitate to adopt solar PV due to uncertainty in cost recovery and operational risks. GreenWise bridges this gap by combining financial modeling and explainable machine learning to enable faster, more confident green investment decisions.

**Current stage**  
This is a functional prototype for decision support and demonstration. Future versions will incorporate real project data and expand to more renewable energy scenarios.
    """
)

st.caption("✅ 这个版本：可直接演示、可录视频、可放官网；后续加真实数据/模型只是在同一框架内升级。")
