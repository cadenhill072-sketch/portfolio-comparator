"""
Portfolio Tracker & Comparator
A general-purpose Roth IRA / investment portfolio analysis tool.
Run: python -m streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Portfolio Comparator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# PRESET PORTFOLIOS
# ─────────────────────────────────────────────

PRESET_PORTFOLIOS = {
    "S&P 500 Only": {"SPY": 1.00},
    "3-Fund Classic (60/30/10)": {"SPY": 0.60, "VXUS": 0.30, "BND": 0.10},
    "Total Market + International": {"VTI": 0.70, "VXUS": 0.30},
    "Aggressive Growth (QQQ heavy)": {"QQQ": 0.50, "SPY": 0.30, "AVUV": 0.20},
    "Dividend + Growth": {"SCHD": 0.40, "SPY": 0.30, "SPYD": 0.15, "VIG": 0.15},
    "Small Cap Value Tilt": {"VTI": 0.50, "AVUV": 0.30, "VXUS": 0.20},
    "All-World Growth": {"VT": 0.60, "QQQ": 0.25, "AVUV": 0.15},
    "60/40 Balanced": {"SPY": 0.60, "BND": 0.40},
    "Income / Dividend Focus": {"SCHD": 0.50, "SPYD": 0.30, "VIG": 0.20},
    "Fidelity Core (FXAIX/FTIHX/FNCMX)": {"SPY": 0.40, "VXUS": 0.30, "QQQ": 0.20, "VBR": 0.10},
}

POPULAR_TICKERS = [
    "SPY", "VOO", "VTI", "QQQ", "IWM", "VT",
    "VXUS", "BND", "AGG", "TLT", "GLD", "SCHD",
    "SPYD", "VIG", "AVUV", "VBR", "IJH", "VNQ",
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
    "TSLA", "BRK-B", "JPM", "XOM",
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers, years):
    start = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    raw = yf.download(list(tickers), start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=list(tickers)[0])
    return raw.dropna(how="all")

def port_returns(prices, weights):
    available = {k: v for k, v in weights.items() if k in prices.columns}
    if not available:
        return pd.Series(dtype=float)
    total_w = sum(available.values())
    norm = {k: v / total_w for k, v in available.items()}
    cols = list(norm.keys())
    w = np.array([norm[c] for c in cols])
    daily = prices[cols].pct_change().dropna()
    return (daily * w).sum(axis=1)

def cagr(ret):
    total = (1 + ret).prod()
    yrs = len(ret) / 252
    return total ** (1 / yrs) - 1 if yrs > 0 else 0

def sharpe(ret, rf=0.05):
    excess = ret - rf / 252
    return (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0

def max_dd(ret):
    cum = (1 + ret).cumprod()
    peak = cum.cummax()
    return ((cum - peak) / peak).min()

def ann_vol(ret):
    return ret.std() * np.sqrt(252)

def dca_project(ann_return, weekly_contrib, start_value, age_now, retire_age):
    weekly_rate = (1 + ann_return) ** (1 / 52) - 1
    weeks = (retire_age - age_now) * 52
    value = start_value
    rows = []
    for w in range(weeks + 1):
        rows.append({"Age": round(age_now + w / 52, 2), "Value": round(value, 2)})
        value = value * (1 + weekly_rate) + weekly_contrib
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# SESSION STATE — portfolios the user builds
# ─────────────────────────────────────────────

if "custom_portfolios" not in st.session_state:
    st.session_state.custom_portfolios = {}

if "active_presets" not in st.session_state:
    st.session_state.active_presets = ["S&P 500 Only", "3-Fund Classic (60/30/10)", "Aggressive Growth (QQQ heavy)"]

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    st.subheader("Your Portfolio")
    total_value  = st.number_input("Total Portfolio Value ($)", value=10000.0, min_value=0.0, step=500.0)
    cash_pct     = st.slider("Cash % (not yet invested)", 0.0, 100.0, 0.0, 0.5)
    age          = st.number_input("Current Age", value=25, min_value=18, max_value=70, step=1)
    retire_age   = st.number_input("Target Retirement Age", value=65, min_value=40, max_value=80, step=1)
    dca_weekly   = st.number_input("Weekly Contribution ($)", value=100.0, min_value=0.0, step=10.0)
    backtest_yrs = st.selectbox("Backtest Period", [1, 3, 5, 7, 10], index=4)

    st.markdown("---")
    st.subheader("Compare Preset Portfolios")
    for name in PRESET_PORTFOLIOS:
        checked = name in st.session_state.active_presets
        if st.checkbox(name, value=checked, key=f"preset_{name}"):
            if name not in st.session_state.active_presets:
                st.session_state.active_presets.append(name)
        else:
            if name in st.session_state.active_presets:
                st.session_state.active_presets.remove(name)

    st.markdown("---")
    st.caption("Data via yfinance · Not financial advice")

# ─────────────────────────────────────────────
# CUSTOM PORTFOLIO BUILDER
# ─────────────────────────────────────────────

st.title("📈 Portfolio Comparator")
st.markdown("Compare investment strategies, backtest performance, and project your growth to retirement.")
st.markdown("---")

with st.expander("➕ Build a Custom Portfolio", expanded=False):
    st.markdown("Enter ticker symbols and weights (must sum to 100%).")
    col_name, _ = st.columns([2, 3])
    with col_name:
        port_name = st.text_input("Portfolio Name", placeholder="My Portfolio")

    num_assets = st.number_input("Number of assets", min_value=1, max_value=10, value=3, step=1)

    cols = st.columns([2, 1])
    tickers_input = []
    weights_input = []

    for i in range(num_assets):
        c1, c2 = st.columns([2, 1])
        with c1:
            t = st.text_input(f"Ticker {i+1}", key=f"tick_{i}", placeholder="e.g. SPY").upper().strip()
        with c2:
            w = st.number_input(f"Weight % {i+1}", key=f"wt_{i}", min_value=0.0, max_value=100.0, value=round(100/num_assets, 1))
        tickers_input.append(t)
        weights_input.append(w)

    total_w = sum(weights_input)
    if abs(total_w - 100) > 0.5:
        st.warning(f"Weights sum to {total_w:.1f}% — must equal 100%.")

    if st.button("Add Portfolio", type="primary"):
        if not port_name:
            st.error("Enter a portfolio name.")
        elif abs(total_w - 100) > 0.5:
            st.error("Fix weights before adding.")
        elif not all(tickers_input):
            st.error("Fill in all ticker fields.")
        else:
            weights_dict = {t: w / 100 for t, w in zip(tickers_input, weights_input) if t}
            st.session_state.custom_portfolios[port_name] = weights_dict
            st.success(f"Added: {port_name}")

    if st.session_state.custom_portfolios:
        st.markdown("**Your custom portfolios:**")
        to_remove = []
        for name, weights in st.session_state.custom_portfolios.items():
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"**{name}** — " + ", ".join([f"{t}: {w*100:.0f}%" for t, w in weights.items()]))
            with c2:
                if st.button("Remove", key=f"rm_{name}"):
                    to_remove.append(name)
        for name in to_remove:
            del st.session_state.custom_portfolios[name]

st.markdown("---")

# ─────────────────────────────────────────────
# BUILD ACTIVE PORTFOLIO SET
# ─────────────────────────────────────────────

active_portfolios = {}
for name in st.session_state.active_presets:
    if name in PRESET_PORTFOLIOS:
        active_portfolios[name] = PRESET_PORTFOLIOS[name]
active_portfolios.update(st.session_state.custom_portfolios)

if not active_portfolios:
    st.info("Select at least one preset portfolio in the sidebar, or build a custom one above.")
    st.stop()

all_tickers = set()
for p in active_portfolios.values():
    all_tickers.update(p.keys())

with st.spinner("Fetching market data..."):
    prices = fetch_prices(tuple(sorted(all_tickers)), backtest_yrs)

missing = [t for t in all_tickers if t not in prices.columns]
if missing:
    st.warning(f"Could not find data for: {', '.join(missing)}. They will be excluded.")

invested_value = total_value * (1 - cash_pct / 100)

# ─────────────────────────────────────────────
# METRICS TABLE
# ─────────────────────────────────────────────

st.subheader(f"📊 {backtest_yrs}-Year Backtest Metrics")

metrics_rows = []
for label, weights in active_portfolios.items():
    ret = port_returns(prices, weights)
    if ret.empty:
        continue
    metrics_rows.append({
        "Portfolio": label,
        "CAGR": f"{cagr(ret):.2%}",
        "Ann. Volatility": f"{ann_vol(ret):.2%}",
        "Sharpe Ratio": f"{sharpe(ret):.2f}",
        "Max Drawdown": f"{max_dd(ret):.2%}",
        f"Total Return ({backtest_yrs}yr)": f"{((1+ret).prod()-1):.2%}",
    })

if metrics_rows:
    st.dataframe(pd.DataFrame(metrics_rows), hide_index=True, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# GROWTH CHART
# ─────────────────────────────────────────────

st.subheader(f"📈 Growth of $10,000 — {backtest_yrs}-Year Backtest")

fig_growth = go.Figure()
colors = px.colors.qualitative.Bold
for i, (label, weights) in enumerate(active_portfolios.items()):
    ret = port_returns(prices, weights)
    if ret.empty:
        continue
    cum = (1 + ret).cumprod() * 10000
    fig_growth.add_trace(go.Scatter(
        x=cum.index, y=cum.values,
        name=label,
        mode="lines",
        line=dict(width=2, color=colors[i % len(colors)]),
    ))

fig_growth.update_layout(
    yaxis_title="Value ($)",
    xaxis_title="Date",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=440,
    margin=dict(t=50, b=40),
)
fig_growth.update_yaxes(tickprefix="$", tickformat=",.0f")
st.plotly_chart(fig_growth, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# DCA PROJECTION
# ─────────────────────────────────────────────

years_to_retire = retire_age - age
if years_to_retire > 0:
    st.subheader(f"🚀 DCA Projection — Age {age} → {retire_age} (${dca_weekly:.0f}/wk, starting ${invested_value:,.0f})")

    fig_proj = go.Figure()
    proj_summary = []

    for i, (label, weights) in enumerate(active_portfolios.items()):
        ret = port_returns(prices, weights)
        if ret.empty:
            continue
        ann = cagr(ret)
        df_proj = dca_project(ann, dca_weekly, invested_value, age, retire_age)

        fig_proj.add_trace(go.Scatter(
            x=df_proj["Age"], y=df_proj["Value"],
            name=label,
            mode="lines",
            line=dict(width=2, color=colors[i % len(colors)]),
        ))

        milestones = {"Portfolio": label}
        for m_age in [30, 40, 50, 60, retire_age]:
            if m_age > age:
                row = df_proj[df_proj["Age"] >= m_age]
                if not row.empty:
                    milestones[f"Age {m_age}"] = f"${row.iloc[0]['Value']:,.0f}"
        proj_summary.append(milestones)

    fig_proj.update_layout(
        yaxis_title="Projected Value ($)",
        xaxis_title="Age",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=440,
        margin=dict(t=50, b=40),
    )
    fig_proj.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig_proj, use_container_width=True)

    if proj_summary:
        proj_df = pd.DataFrame(proj_summary)
        cols_order = ["Portfolio"] + [c for c in proj_df.columns if c != "Portfolio"]
        st.dataframe(proj_df[cols_order], hide_index=True, use_container_width=True)

    st.markdown("---")

# ─────────────────────────────────────────────
# RISK vs RETURN SCATTER
# ─────────────────────────────────────────────

st.subheader("⚖️ Risk vs. Return")

scatter_data = []
for label, weights in active_portfolios.items():
    ret = port_returns(prices, weights)
    if ret.empty:
        continue
    scatter_data.append({
        "Portfolio": label,
        "Annual Return (CAGR %)": round(cagr(ret) * 100, 2),
        "Annual Volatility (Risk %)": round(ann_vol(ret) * 100, 2),
        "Sharpe": round(sharpe(ret), 2),
    })

if scatter_data:
    sc_df = pd.DataFrame(scatter_data)
    fig_scatter = px.scatter(
        sc_df,
        x="Annual Volatility (Risk %)",
        y="Annual Return (CAGR %)",
        text="Portfolio",
        size=[max(s, 0.1) for s in sc_df["Sharpe"]],
        color="Sharpe",
        color_continuous_scale="RdYlGn",
        size_max=40,
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(height=460, coloraxis_colorbar_title="Sharpe")
    fig_scatter.update_xaxes(ticksuffix="%")
    fig_scatter.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# TICKER LOOKUP
# ─────────────────────────────────────────────

st.subheader("🔍 Quick Ticker Lookup")
lookup = st.text_input("Enter a ticker to see its info", placeholder="e.g. AAPL, VTI, QQQ").upper().strip()
if lookup:
    try:
        info = yf.Ticker(lookup).info
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Name", info.get("shortName", lookup))
        c2.metric("Current Price", f"${info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))}")
        c3.metric("52-Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
        c4.metric("52-Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")
        if info.get("longBusinessSummary"):
            with st.expander("Description"):
                st.write(info["longBusinessSummary"])
    except Exception as e:
        st.error(f"Could not load data for {lookup}: {e}")

st.caption("⚠️ Past performance does not guarantee future results. This tool is for educational purposes only and is not financial advice.")
