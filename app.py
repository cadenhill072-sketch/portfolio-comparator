"""
Portfolio Comparator + Optimizer
Run: python -m streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
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
# BROKERAGE UNIVERSES
# ─────────────────────────────────────────────

BROKERAGES = {
    "Any / Custom": [
        # Broad Market ETFs
        "SPY", "VOO", "IVV", "VTI", "SCHB", "SCHX", "QQQ", "ONEQ", "IWM", "DIA",
        "VT", "VXUS", "VEA", "VWO", "EFA", "EEM",
        # Factor / Style
        "VUG", "VTV", "SCHG", "SCHV", "AVUV", "AVLV", "VBR", "VBK", "VO", "VB",
        "IJH", "IJS", "IJR",
        # Dividends / Income
        "SCHD", "VYM", "VIG", "SPYD", "JEPI", "JEPQ", "FDVV", "DVY", "HDV",
        # Bonds
        "BND", "AGG", "TLT", "IEF", "SHY", "SCHZ", "SCHP", "VTIP", "HYG", "LQD",
        "VCIT", "VGIT", "VGLT", "VMBS",
        # Sector ETFs
        "VGT", "XLK", "SOXX", "SMH", "VHT", "XLV", "IBB", "VFH", "XLF",
        "VNQ", "VDE", "XLE", "XLU", "XLP", "XLY", "XLI", "VAW", "VIS", "VCR",
        # Alternatives
        "GLD", "SLV", "IAU", "PDBC", "DJP",
        # Fidelity Funds
        "FXAIX", "FSKAX", "FZROX", "FZILX", "FNCMX", "FSSNX", "FSMDX",
        "FTIHX", "FXNAX", "FPADX", "FSPSX", "FBIOX", "FSPTX", "FSCSX",
        "FSELX", "FSPHX", "FSDIX",
        # Vanguard Funds
        "VFIAX", "VTSAX", "VBTLX", "VTIAX",
        # Individual Stocks
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA",
        "BRK-B", "JPM", "V", "MA", "JNJ", "PG", "XOM", "CVX", "HD", "UNH",
        "AVGO", "LLY", "WMT", "MRK", "COST", "NFLX", "AMD", "INTC", "CRM",
        "ORCL", "ADBE", "NOW", "INTU", "QCOM", "TXN", "ARM", "PLTR", "SNOW",
        "COIN", "HOOD", "SOFI", "RIVN", "NIO", "F", "GM", "BAC", "WFC", "GS",
    ],
    "Fidelity": [
        "FXAIX", "FSKAX", "FZROX", "FZILX", "FNCMX", "FSSNX", "FSMDX",
        "FTIHX", "FXNAX", "FPADX", "FSPSX", "FBIOX", "FSPTX", "FSCSX",
        "FSELX", "FSPHX", "FSDIX", "FDVV", "ONEQ",
    ],
    "Vanguard": [
        "VOO", "VTI", "VXUS", "BND", "VT", "VUG", "VYM", "VIG", "VNQ",
        "VBR", "VBK", "VO", "VB", "VEA", "VWO", "VTIP", "VCIT", "VGIT",
        "VGLT", "VMBS", "VGT", "VHT", "VFH", "VDE", "VAW", "VIS", "VCR",
    ],
    "Schwab": [
        "SCHB", "SCHX", "SCHD", "SCHF", "SCHE", "SCHG", "SCHV", "SCHA",
        "SCHM", "SCHI", "SCHZ", "SCHP", "SCHR", "SCHQ", "SCHH", "SCHC",
        "SCHY",
    ],
    "Robinhood / General ETFs": [
        "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "TLT", "HYG", "LQD",
        "SPYD", "JEPI", "JEPQ", "SCHD", "AVUV", "AVLV", "VT", "VTI",
        "VXUS", "BND", "IJH", "IVV",
    ],
    "Individual Stocks (Mag 7 + Blue Chip)": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
        "BRK-B", "JPM", "V", "JNJ", "PG", "XOM", "HD", "UNH",
        "MA", "AVGO", "LLY", "WMT", "MRK",
    ],
}

# Fidelity tickers need ETF proxies for yfinance (mutual funds not on yfinance)
FIDELITY_PROXY = {
    "FXAIX": "SPY", "FSKAX": "VTI", "FZROX": "VTI", "FZILX": "VXUS",
    "FNCMX": "QQQ", "FSSNX": "VBR", "FSMDX": "IJH", "FTIHX": "VXUS",
    "FXNAX": "BND", "FPADX": "VWO", "FSPSX": "VEA", "FBIOX": "IBB",
    "FSPTX": "XLK", "FSCSX": "XLK", "FSELX": "SOXX", "FSPHX": "XLV",
    "FSDIX": "XLF", "FDVV": "SCHD", "ONEQ": "QQQ",
}

# ─────────────────────────────────────────────
# PRESET COMPARISON PORTFOLIOS
# ─────────────────────────────────────────────

PRESET_PORTFOLIOS = {
    "S&P 500 Only":                     {"SPY": 1.00},
    "3-Fund Classic (60/30/10)":        {"SPY": 0.60, "VXUS": 0.30, "BND": 0.10},
    "Total Market + International":     {"VTI": 0.70, "VXUS": 0.30},
    "Aggressive Growth (QQQ heavy)":    {"QQQ": 0.50, "SPY": 0.30, "AVUV": 0.20},
    "Dividend + Growth":                {"SCHD": 0.40, "SPY": 0.30, "SPYD": 0.15, "VIG": 0.15},
    "Small Cap Value Tilt":             {"VTI": 0.50, "AVUV": 0.30, "VXUS": 0.20},
    "All-World Growth":                 {"VT": 0.60, "QQQ": 0.25, "AVUV": 0.15},
    "60/40 Balanced":                   {"SPY": 0.60, "BND": 0.40},
    "Income / Dividend Focus":          {"SCHD": 0.50, "SPYD": 0.30, "VIG": 0.20},
}

# ─────────────────────────────────────────────
# MATH HELPERS
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers, years):
    start = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    raw = yf.download(list(tickers), start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=list(tickers)[0])
    return raw.dropna(how="all")

def resolve_tickers(tickers, brokerage):
    """Map Fidelity mutual fund tickers to ETF proxies for yfinance."""
    if brokerage == "Fidelity":
        return {t: FIDELITY_PROXY.get(t, t) for t in tickers}
    return {t: t for t in tickers}

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

def dca_project(ann_return, weekly_contrib, start_value, age_now, retire_age, inflation=0.03):
    weekly_rate = (1 + ann_return) ** (1 / 52) - 1
    weeks = (retire_age - age_now) * 52
    value = start_value
    rows = []
    for w in range(weeks + 1):
        real_value = value / ((1 + inflation) ** (w / 52))
        rows.append({
            "Age": round(age_now + w / 52, 2),
            "Nominal Value": round(value, 2),
            "Real Value (Today's $)": round(real_value, 2),
        })
        value = value * (1 + weekly_rate) + weekly_contrib
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# PORTFOLIO OPTIMIZER (Max Sharpe)
# ─────────────────────────────────────────────

def optimize_sharpe(prices, tickers, rf=0.05, min_weight=0.0, max_weight=1.0):
    """Find weights that maximize the Sharpe ratio using mean-variance optimization."""
    available = [t for t in tickers if t in prices.columns]
    if len(available) < 2:
        return None, None

    daily = prices[available].pct_change().dropna()
    mean_returns = daily.mean() * 252
    cov_matrix = daily.cov() * 252
    n = len(available)

    def neg_sharpe(w):
        port_ret = np.dot(w, mean_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return -(port_ret - rf) / port_vol if port_vol > 0 else 0

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(min_weight, max_weight)] * n
    x0 = np.ones(n) / n

    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-9})

    if result.success:
        weights = {available[i]: round(result.x[i], 4) for i in range(n)}
        weights = {k: v for k, v in weights.items() if v > 0.0001}
        return weights, -result.fun
    return None, None

def efficient_frontier(prices, tickers, n_points=60, rf=0.05):
    available = [t for t in tickers if t in prices.columns]
    if len(available) < 2:
        return pd.DataFrame()

    daily = prices[available].pct_change().dropna()
    mean_returns = daily.mean() * 252
    cov_matrix = daily.cov() * 252
    n = len(available)

    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), n_points)
    frontier = []

    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mean_returns) - t},
        ]
        bounds = [(0.0, 1.0)] * n
        x0 = np.ones(n) / n
        result = minimize(
            lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
            x0, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"maxiter": 500},
        )
        if result.success:
            vol = result.fun
            sr = (target - rf) / vol if vol > 0 else 0
            frontier.append({"Return": target * 100, "Volatility": vol * 100, "Sharpe": sr})

    return pd.DataFrame(frontier)

# ─────────────────────────────────────────────
# SESSION STATE
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

    st.subheader("Your Info")
    total_value  = st.number_input("Total Portfolio Value ($)", value=10000.0, min_value=0.0, step=500.0)
    cash_pct     = st.slider("Cash % (not yet invested)", 0.0, 100.0, 0.0, 0.5)
    age          = st.number_input("Current Age", value=25, min_value=18, max_value=70, step=1)
    retire_age   = st.number_input("Target Retirement Age", value=65, min_value=40, max_value=80, step=1)
    dca_weekly   = st.number_input("Weekly Contribution ($)", value=100.0, min_value=0.0, step=10.0)
    backtest_yrs = st.selectbox("Backtest Period", [1, 3, 5, 7, 10], index=4)
    rf_rate      = st.number_input("Risk-Free Rate (%)", value=5.0, min_value=0.0, max_value=15.0, step=0.1) / 100

    st.markdown("---")
    st.subheader("Compare Presets")
    for name in PRESET_PORTFOLIOS:
        checked = name in st.session_state.active_presets
        if st.checkbox(name, value=checked, key=f"preset_{name}"):
            if name not in st.session_state.active_presets:
                st.session_state.active_presets.append(name)
        else:
            if name in st.session_state.active_presets:
                st.session_state.active_presets.remove(name)

    st.markdown("---")
    st.subheader("Projection Settings")
    return_mode = st.radio(
        "Expected Return Source",
        ["Preset", "Historical CAGR", "Custom"],
        index=0,
        help="Historical CAGR uses your selected backtest period — may be inflated by recent bull market."
    )
    if return_mode == "Preset":
        preset_return = st.selectbox(
            "Scenario",
            ["Conservative (7%)", "Moderate (9%)", "Aggressive (11%)"],
            index=1,
        )
        proj_return = {"Conservative (7%)": 0.07, "Moderate (9%)": 0.09, "Aggressive (11%)": 0.11}[preset_return]
    elif return_mode == "Custom":
        proj_return = st.number_input("Annual Return (%)", value=8.0, min_value=1.0, max_value=20.0, step=0.5) / 100
    else:
        proj_return = None  # use historical CAGR per portfolio

    inflation_rate = st.number_input("Inflation Rate (%)", value=3.0, min_value=0.0, max_value=10.0, step=0.1) / 100
    show_real = st.checkbox("Show inflation-adjusted (real) values", value=True)

    st.markdown("---")
    st.caption("Data via yfinance · Not financial advice")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("📈 Portfolio Comparator")
st.markdown("Compare strategies, find the **optimal Sharpe ratio portfolio**, and project growth to retirement.")
st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🏗️ Build & Compare", "🎯 Optimizer", "🔍 Ticker Lookup"])

# ══════════════════════════════════════════════
# TAB 1 — BUILD & COMPARE
# ══════════════════════════════════════════════

with tab1:

    with st.expander("➕ Build a Custom Portfolio", expanded=False):
        st.markdown("Search and select tickers below — then set each weight. Weights must sum to **100%**.")

        brokerage_custom = st.selectbox("Brokerage (filters available tickers)", list(BROKERAGES.keys()), key="brok_custom")
        port_name = st.text_input("Portfolio Name", placeholder="My Portfolio")

        ticker_pool = BROKERAGES[brokerage_custom]

        selected_tickers = st.multiselect(
            "Search & select tickers (type to filter)",
            options=ticker_pool,
            placeholder="Type a ticker e.g. SPY, QQQ, SCHD...",
            key="multi_tickers",
        )

        weights_input = []
        if selected_tickers:
            st.markdown("**Set weights for each selected ticker:**")
            default_w = round(100.0 / len(selected_tickers), 1)
            for t in selected_tickers:
                w = st.number_input(f"{t} weight (%)", min_value=0.0, max_value=100.0,
                                    value=default_w, step=0.1, format="%.1f", key=f"wt_{t}")
                weights_input.append(w)

            total_w = sum(weights_input)
            if abs(total_w - 100) > 0.1:
                st.warning(f"Weights sum to **{total_w:.1f}%** — must equal 100%.")
            else:
                st.success(f"Weights sum to {total_w:.1f}% ✓")
        else:
            total_w = 0

        if st.button("Add Portfolio", type="primary"):
            if not port_name:
                st.error("Enter a portfolio name.")
            elif not selected_tickers:
                st.error("Select at least one ticker.")
            elif abs(total_w - 100) > 0.1:
                st.error("Fix weights before adding.")
            else:
                proxy_map = resolve_tickers(selected_tickers, brokerage_custom)
                weights_dict = {proxy_map[t]: w / 100 for t, w in zip(selected_tickers, weights_input)}
                label = f"{port_name} ({brokerage_custom})" if brokerage_custom != "Any / Custom" else port_name
                st.session_state.custom_portfolios[label] = weights_dict
                st.success(f"Added: {label}")

        if st.session_state.custom_portfolios:
            st.markdown("**Your custom portfolios:**")
            to_remove = []
            for name, weights in st.session_state.custom_portfolios.items():
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f"**{name}** — " + ", ".join([f"{t}: {w*100:.1f}%" for t, w in weights.items()]))
                with c2:
                    if st.button("Remove", key=f"rm_{name}"):
                        to_remove.append(name)
            for name in to_remove:
                del st.session_state.custom_portfolios[name]
            st.rerun()

    st.markdown("---")

    # Build active set
    active_portfolios = {}
    for name in st.session_state.active_presets:
        if name in PRESET_PORTFOLIOS:
            active_portfolios[name] = PRESET_PORTFOLIOS[name]
    active_portfolios.update(st.session_state.custom_portfolios)

    if not active_portfolios:
        st.info("Select at least one preset in the sidebar, or build a custom portfolio above.")
        st.stop()

    all_tickers = set()
    for p in active_portfolios.values():
        all_tickers.update(p.keys())

    with st.spinner("Fetching market data..."):
        prices = fetch_prices(tuple(sorted(all_tickers)), backtest_yrs)

    missing = [t for t in all_tickers if t not in prices.columns]
    if missing:
        st.warning(f"No data found for: {', '.join(missing)} — excluded from analysis.")

    invested_value = total_value * (1 - cash_pct / 100)
    colors = px.colors.qualitative.Bold

    # ── Metrics table ──
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
            "Sharpe Ratio": f"{sharpe(ret, rf_rate):.2f}",
            "Max Drawdown": f"{max_dd(ret):.2%}",
            f"Total Return ({backtest_yrs}yr)": f"{((1+ret).prod()-1):.2%}",
        })
    if metrics_rows:
        st.dataframe(pd.DataFrame(metrics_rows), hide_index=True, use_container_width=True)

    st.markdown("---")

    # ── Growth chart ──
    st.subheader(f"📈 Growth of $10,000 — {backtest_yrs}-Year Backtest")
    fig_growth = go.Figure()
    for i, (label, weights) in enumerate(active_portfolios.items()):
        ret = port_returns(prices, weights)
        if ret.empty:
            continue
        cum = (1 + ret).cumprod() * 10000
        fig_growth.add_trace(go.Scatter(x=cum.index, y=cum.values, name=label,
                                        mode="lines", line=dict(width=2, color=colors[i % len(colors)])))
    fig_growth.update_layout(yaxis_title="Value ($)", xaxis_title="Date", hovermode="x unified",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                              height=420, margin=dict(t=50, b=40))
    fig_growth.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig_growth, use_container_width=True)

    st.markdown("---")

    # ── DCA Projection ──
    if retire_age > age:
        value_col = "Real Value (Today's $)" if show_real else "Nominal Value"
        value_label = "Projected Value — Today's Dollars (Inflation-Adjusted)" if show_real else "Projected Nominal Value ($)"

        if return_mode == "Preset":
            st.subheader(f"🚀 DCA Projection — Age {age} → {retire_age} · {preset_return} · ${dca_weekly:.0f}/wk")
            st.caption(f"Using {proj_return*100:.0f}% annual return · {inflation_rate*100:.1f}% inflation · All portfolios use same rate")
        elif return_mode == "Custom":
            st.subheader(f"🚀 DCA Projection — Age {age} → {retire_age} · {proj_return*100:.1f}% return · ${dca_weekly:.0f}/wk")
            st.caption(f"Custom {proj_return*100:.1f}% annual return · {inflation_rate*100:.1f}% inflation · All portfolios use same rate")
        else:
            st.subheader(f"🚀 DCA Projection — Age {age} → {retire_age} · Historical CAGR · ${dca_weekly:.0f}/wk")
            st.caption(f"⚠️ Uses each portfolio's {backtest_yrs}-yr historical CAGR — recent market was above average. {inflation_rate*100:.1f}% inflation applied.")

        fig_proj = go.Figure()
        proj_summary = []
        for i, (label, weights) in enumerate(active_portfolios.items()):
            ret = port_returns(prices, weights)
            if ret.empty:
                continue
            used_return = proj_return if proj_return is not None else cagr(ret)
            df_proj = dca_project(used_return, dca_weekly, invested_value, age, retire_age, inflation_rate)
            fig_proj.add_trace(go.Scatter(
                x=df_proj["Age"], y=df_proj[value_col], name=label,
                mode="lines", line=dict(width=2, color=colors[i % len(colors)]),
                hovertemplate=f"<b>{label}</b><br>Age: %{{x:.1f}}<br>Value: $%{{y:,.0f}}<extra></extra>",
            ))
            row = {"Portfolio": label, "Return Used": f"{used_return*100:.1f}%"}
            for m in [30, 40, 50, 60, retire_age]:
                if m > age:
                    r = df_proj[df_proj["Age"] >= m]
                    if not r.empty:
                        row[f"Age {m}"] = f"${r.iloc[0][value_col]:,.0f}"
            proj_summary.append(row)

        fig_proj.update_layout(
            yaxis_title=value_label, xaxis_title="Age", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=440, margin=dict(t=50, b=40),
        )
        fig_proj.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig_proj, use_container_width=True)

        if proj_summary:
            proj_df = pd.DataFrame(proj_summary)
            st.dataframe(proj_df[["Portfolio", "Return Used"] + [c for c in proj_df.columns if c not in ("Portfolio", "Return Used")]],
                         hide_index=True, use_container_width=True)

    st.markdown("---")

    # ── Risk vs Return scatter ──
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
            "Sharpe": round(sharpe(ret, rf_rate), 2),
        })
    if scatter_data:
        sc_df = pd.DataFrame(scatter_data)
        fig_sc = px.scatter(sc_df, x="Annual Volatility (Risk %)", y="Annual Return (CAGR %)",
                            text="Portfolio", size=[max(s, 0.1) for s in sc_df["Sharpe"]],
                            color="Sharpe", color_continuous_scale="RdYlGn", size_max=40)
        fig_sc.update_traces(textposition="top center")
        fig_sc.update_layout(height=460)
        fig_sc.update_xaxes(ticksuffix="%")
        fig_sc.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_sc, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — OPTIMIZER
# ══════════════════════════════════════════════

with tab2:
    st.subheader("🎯 Max Sharpe Ratio Optimizer")
    st.markdown("Select tickers and hit **Optimize** — the algorithm finds the exact weights that produce the highest possible Sharpe ratio using Modern Portfolio Theory.")

    # ── Controls ──
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        brokerage = st.selectbox("Brokerage", list(BROKERAGES.keys()), key="brok_opt")
    with c2:
        opt_years = st.selectbox("Data period", [1, 3, 5, 7, 10], index=3, key="opt_yrs")
    with c3:
        min_w = st.number_input("Min weight %", 0.0, 20.0, 0.0, 0.5, format="%.1f") / 100
    with c4:
        max_w = st.number_input("Max weight %", 10.0, 100.0, 100.0, 1.0, format="%.1f") / 100

    opt_tickers = st.multiselect(
        "Select tickers to optimize across (type to search)",
        options=BROKERAGES[brokerage],
        default=BROKERAGES[brokerage][:6] if len(BROKERAGES[brokerage]) >= 6 else BROKERAGES[brokerage],
        placeholder="Type a ticker...",
        key="opt_tickers",
    )

    run_opt = st.button("⚡ Optimize — Find Highest Sharpe Ratio", type="primary", use_container_width=True)

    if run_opt:
        if len(opt_tickers) < 2:
            st.error("Select at least 2 tickers.")
        else:
            proxy_map = resolve_tickers(opt_tickers, brokerage)
            etf_tickers = list(set(proxy_map.values()))
            reverse_proxy = {v: k for k, v in proxy_map.items()}

            with st.spinner("Computing max Sharpe ratio and efficient frontier..."):
                opt_prices = fetch_prices(tuple(sorted(etf_tickers)), opt_years)
                opt_weights, best_sharpe = optimize_sharpe(opt_prices, etf_tickers, rf_rate, min_w, max_w)
                frontier_df = efficient_frontier(opt_prices, etf_tickers, rf=rf_rate)

            if opt_weights is None:
                st.error("Optimization failed — try different tickers or a longer data period.")
            else:
                display_weights = {reverse_proxy.get(t, t): w for t, w in opt_weights.items()}
                opt_ret = port_returns(opt_prices, opt_weights)

                # ── Hero metric ──
                st.markdown("---")
                hero_col, _, _ = st.columns([1, 1, 1])
                with hero_col:
                    st.metric("🏆 Maximum Sharpe Ratio", f"{best_sharpe:.4f}",
                              help="Higher = better risk-adjusted return. A Sharpe above 1.0 is generally considered good.")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("CAGR", f"{cagr(opt_ret):.2%}")
                m2.metric("Annual Volatility", f"{ann_vol(opt_ret):.2%}")
                m3.metric("Max Drawdown", f"{max_dd(opt_ret):.2%}")
                m4.metric("Risk-Free Rate Used", f"{rf_rate*100:.1f}%")

                st.markdown("---")

                # ── Weights + pie side by side ──
                left, right = st.columns([1, 1])
                with left:
                    st.subheader("Optimal Weights")
                    wt_df = pd.DataFrame([
                        {
                            "Ticker": t,
                            "Weight": f"{w*100:.2f}%",
                            "$ Amount": f"${total_value * (1 - cash_pct/100) * w:,.2f}",
                        }
                        for t, w in sorted(display_weights.items(), key=lambda x: -x[1])
                    ])
                    st.dataframe(wt_df, hide_index=True, use_container_width=True)

                    if st.button("➕ Add Optimal to Compare Tab", use_container_width=True):
                        label = f"Optimal — {brokerage} ({opt_years}yr)"
                        st.session_state.custom_portfolios[label] = opt_weights
                        st.success(f"Added to Compare tab!")

                with right:
                    fig_pie = px.pie(
                        values=[w * 100 for w in display_weights.values()],
                        names=list(display_weights.keys()),
                        hole=0.45,
                        color_discrete_sequence=px.colors.qualitative.Bold,
                    )
                    fig_pie.update_traces(textinfo="percent+label", textposition="inside")
                    fig_pie.update_layout(showlegend=False, height=320, margin=dict(t=10, b=0))
                    st.plotly_chart(fig_pie, use_container_width=True)

                st.markdown("---")

                # ── Efficient Frontier ──
                st.subheader("📉 Efficient Frontier")
                if not frontier_df.empty:
                    daily = opt_prices[[t for t in etf_tickers if t in opt_prices.columns]].pct_change().dropna()
                    asset_dots = pd.DataFrame({
                        "Volatility": daily.std() * np.sqrt(252) * 100,
                        "Return": daily.mean() * 252 * 100,
                    })
                    asset_dots.index = [reverse_proxy.get(t, t) for t in asset_dots.index]

                    max_sr_idx = frontier_df["Sharpe"].idxmax()
                    fig_ef = go.Figure()

                    # Frontier line colored by Sharpe
                    fig_ef.add_trace(go.Scatter(
                        x=frontier_df["Volatility"], y=frontier_df["Return"],
                        mode="lines+markers",
                        name="Efficient Frontier",
                        marker=dict(color=frontier_df["Sharpe"], colorscale="RdYlGn",
                                    size=5, showscale=True, colorbar=dict(title="Sharpe")),
                        line=dict(color="royalblue", width=2),
                        hovertemplate="Volatility: %{x:.1f}%<br>Return: %{y:.1f}%<br>Sharpe: %{marker.color:.2f}<extra></extra>",
                    ))
                    # Individual asset dots
                    fig_ef.add_trace(go.Scatter(
                        x=asset_dots["Volatility"], y=asset_dots["Return"],
                        mode="markers+text", name="Individual Assets",
                        text=asset_dots.index, textposition="top center",
                        marker=dict(size=11, color="orange", symbol="diamond"),
                    ))
                    # Max Sharpe star
                    fig_ef.add_trace(go.Scatter(
                        x=[frontier_df.loc[max_sr_idx, "Volatility"]],
                        y=[frontier_df.loc[max_sr_idx, "Return"]],
                        mode="markers+text", name=f"Max Sharpe ({best_sharpe:.2f})",
                        text=[f"★ {best_sharpe:.2f}"], textposition="top right",
                        marker=dict(size=18, color="gold", symbol="star"),
                    ))
                    fig_ef.update_layout(
                        xaxis_title="Risk (Annual Volatility %)",
                        yaxis_title="Return (Annual CAGR %)",
                        height=500, hovermode="closest",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    fig_ef.update_xaxes(ticksuffix="%")
                    fig_ef.update_yaxes(ticksuffix="%")
                    st.plotly_chart(fig_ef, use_container_width=True)
                    st.caption("Each point on the frontier is the minimum-risk portfolio for a given return level. The gold star is the max Sharpe (tangency) portfolio.")


# ══════════════════════════════════════════════
# TAB 3 — TICKER LOOKUP
# ══════════════════════════════════════════════

with tab3:
    st.subheader("🔍 Ticker Lookup")
    lookup = st.text_input("Enter a ticker symbol", placeholder="e.g. AAPL, VTI, QQQ, SCHD").upper().strip()
    if lookup:
        try:
            ticker = yf.Ticker(lookup)
            info = ticker.info
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Name", info.get("shortName", lookup))
            price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("navPrice", "N/A")
            c2.metric("Price", f"${price}" if price != "N/A" else "N/A")
            c3.metric("52-Wk High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
            c4.metric("52-Wk Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")

            col_a, col_b = st.columns(2)
            col_a.metric("Expense Ratio", f"{info.get('annualReportExpenseRatio', info.get('netExpenseRatio', 'N/A'))}")
            col_a.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A")
            col_b.metric("Category", info.get("category", info.get("sector", "N/A")))
            col_b.metric("Fund Family", info.get("fundFamily", info.get("industry", "N/A")))

            if info.get("longBusinessSummary"):
                with st.expander("Description"):
                    st.write(info["longBusinessSummary"])

            # Price chart
            hist = ticker.history(period=f"{backtest_yrs}y")
            if not hist.empty:
                fig_tick = go.Figure()
                fig_tick.add_trace(go.Scatter(x=hist.index, y=hist["Close"],
                                              mode="lines", name=lookup,
                                              line=dict(color="royalblue", width=2)))
                fig_tick.update_layout(title=f"{lookup} Price History ({backtest_yrs}yr)",
                                       yaxis_title="Price ($)", height=350,
                                       margin=dict(t=40, b=30))
                fig_tick.update_yaxes(tickprefix="$")
                st.plotly_chart(fig_tick, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load data for {lookup}: {e}")

st.markdown("---")
st.caption("⚠️ Past performance does not guarantee future results. This tool is for educational purposes only and is not financial advice.")
