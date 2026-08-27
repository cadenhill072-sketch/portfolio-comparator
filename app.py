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
    "Any / Custom": [],
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
        st.markdown("Enter tickers and weights. Weights must sum to **100%**.")

        brokerage_custom = st.selectbox("Brokerage (filters suggested tickers)", list(BROKERAGES.keys()), key="brok_custom")
        if BROKERAGES[brokerage_custom]:
            st.caption("Available tickers: " + "  ·  ".join(BROKERAGES[brokerage_custom]))

        port_name = st.text_input("Portfolio Name", placeholder="My Portfolio")
        num_assets = st.number_input("Number of assets", min_value=1, max_value=15, value=3, step=1)

        tickers_input, weights_input = [], []
        ticker_options = BROKERAGES[brokerage_custom] if BROKERAGES[brokerage_custom] else []

        for i in range(num_assets):
            c1, c2 = st.columns([3, 1])
            with c1:
                if ticker_options:
                    t = st.selectbox(f"Ticker {i+1}", [""] + ticker_options, key=f"tick_{i}")
                else:
                    t = st.text_input(f"Ticker {i+1}", key=f"tick_{i}", placeholder="e.g. SPY").upper().strip()
            with c2:
                w = st.number_input(f"Weight % {i+1}", key=f"wt_{i}",
                                    min_value=0.0, max_value=100.0,
                                    value=round(100.0 / num_assets, 1), step=0.1, format="%.1f")
            tickers_input.append(str(t).upper().strip())
            weights_input.append(w)

        total_w = sum(weights_input)
        if abs(total_w - 100) > 0.1:
            st.warning(f"Weights sum to **{total_w:.1f}%** — must equal 100%.")
        else:
            st.success(f"Weights sum to {total_w:.1f}% ✓")

        if st.button("Add Portfolio", type="primary"):
            if not port_name:
                st.error("Enter a portfolio name.")
            elif abs(total_w - 100) > 0.1:
                st.error("Fix weights before adding.")
            elif not all(tickers_input):
                st.error("Fill in all ticker fields.")
            else:
                proxy_map = resolve_tickers(tickers_input, brokerage_custom)
                weights_dict = {proxy_map[t]: w / 100 for t, w in zip(tickers_input, weights_input) if t}
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
        st.subheader(f"🚀 DCA Projection — Age {age} → {retire_age} (${dca_weekly:.0f}/wk)")
        fig_proj = go.Figure()
        proj_summary = []
        for i, (label, weights) in enumerate(active_portfolios.items()):
            ret = port_returns(prices, weights)
            if ret.empty:
                continue
            df_proj = dca_project(cagr(ret), dca_weekly, invested_value, age, retire_age)
            fig_proj.add_trace(go.Scatter(x=df_proj["Age"], y=df_proj["Value"], name=label,
                                          mode="lines", line=dict(width=2, color=colors[i % len(colors)])))
            row = {"Portfolio": label}
            for m in [30, 40, 50, 60, retire_age]:
                if m > age:
                    r = df_proj[df_proj["Age"] >= m]
                    if not r.empty:
                        row[f"Age {m}"] = f"${r.iloc[0]['Value']:,.0f}"
            proj_summary.append(row)

        fig_proj.update_layout(yaxis_title="Projected Value ($)", xaxis_title="Age", hovermode="x unified",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                height=420, margin=dict(t=50, b=40))
        fig_proj.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig_proj, use_container_width=True)

        if proj_summary:
            proj_df = pd.DataFrame(proj_summary)
            st.dataframe(proj_df[["Portfolio"] + [c for c in proj_df.columns if c != "Portfolio"]],
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
    st.markdown("""
    Select a brokerage and tickers. The optimizer finds the **exact weights** that maximize the
    Sharpe ratio (best risk-adjusted return) using **Modern Portfolio Theory**.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        brokerage = st.selectbox("Brokerage", list(BROKERAGES.keys()), key="brok_opt")
        opt_years = st.selectbox("Data period for optimization", [1, 3, 5, 7, 10], index=3, key="opt_yrs")
        min_w = st.number_input("Min weight per asset (%)", 0.0, 20.0, 0.0, 0.5, format="%.1f") / 100
        max_w = st.number_input("Max weight per asset (%)", 10.0, 100.0, 100.0, 1.0, format="%.1f") / 100

        if BROKERAGES[brokerage]:
            opt_tickers = st.multiselect(
                "Select tickers to optimize across",
                BROKERAGES[brokerage],
                default=BROKERAGES[brokerage][:6],
                key="opt_tickers"
            )
        else:
            raw = st.text_area("Enter tickers (one per line or comma-separated)",
                               "SPY\nQQQ\nVXUS\nBND\nAVUV\nSCHD", key="opt_raw")
            opt_tickers = [t.strip().upper() for t in raw.replace(",", "\n").split("\n") if t.strip()]

    with col2:
        if st.button("🚀 Find Optimal Portfolio", type="primary", use_container_width=True):
            if len(opt_tickers) < 2:
                st.error("Select at least 2 tickers.")
            else:
                proxy_map = resolve_tickers(opt_tickers, brokerage)
                etf_tickers = list(set(proxy_map.values()))

                with st.spinner("Running optimization..."):
                    opt_prices = fetch_prices(tuple(sorted(etf_tickers)), opt_years)
                    opt_weights, best_sharpe = optimize_sharpe(opt_prices, etf_tickers, rf_rate, min_w, max_w)

                if opt_weights is None:
                    st.error("Optimization failed. Try different tickers or a longer period.")
                else:
                    # Reverse proxy map for display
                    reverse_proxy = {v: k for k, v in proxy_map.items()}
                    display_weights = {reverse_proxy.get(t, t): w for t, w in opt_weights.items()}

                    st.success(f"Optimal Sharpe Ratio: **{best_sharpe:.3f}**")

                    # Weights table
                    wt_df = pd.DataFrame([
                        {"Ticker": t, "Weight": f"{w*100:.2f}%", "$ Amount": f"${invested_value * w:,.2f}"}
                        for t, w in sorted(display_weights.items(), key=lambda x: -x[1])
                    ])
                    st.dataframe(wt_df, hide_index=True, use_container_width=True)

                    # Pie chart
                    fig_pie = px.pie(
                        values=[w * 100 for w in display_weights.values()],
                        names=list(display_weights.keys()),
                        title="Optimal Allocation",
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Bold,
                    )
                    fig_pie.update_traces(textinfo="percent+label", textposition="inside")
                    fig_pie.update_layout(showlegend=False, height=350, margin=dict(t=40, b=0))
                    st.plotly_chart(fig_pie, use_container_width=True)

                    # Metrics for the optimal portfolio
                    opt_ret = port_returns(opt_prices, opt_weights)
                    if not opt_ret.empty:
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("CAGR", f"{cagr(opt_ret):.2%}")
                        m2.metric("Volatility", f"{ann_vol(opt_ret):.2%}")
                        m3.metric("Sharpe Ratio", f"{best_sharpe:.3f}")
                        m4.metric("Max Drawdown", f"{max_dd(opt_ret):.2%}")

                    # Add to comparison
                    label = f"Optimal ({brokerage})"
                    if st.button(f"➕ Add to Comparison Tab"):
                        st.session_state.custom_portfolios[label] = opt_weights
                        st.success(f"Added '{label}' to the Compare tab!")

        st.markdown("---")

        # Efficient Frontier
        st.subheader("📉 Efficient Frontier")
        if st.button("Plot Efficient Frontier", use_container_width=True):
            if len(opt_tickers) < 2:
                st.error("Select at least 2 tickers first.")
            else:
                proxy_map = resolve_tickers(opt_tickers, brokerage)
                etf_tickers = list(set(proxy_map.values()))
                with st.spinner("Computing efficient frontier..."):
                    opt_prices = fetch_prices(tuple(sorted(etf_tickers)), opt_years)
                    frontier_df = efficient_frontier(opt_prices, etf_tickers, rf=rf_rate)

                if not frontier_df.empty:
                    # Individual asset dots
                    daily = opt_prices[[t for t in etf_tickers if t in opt_prices.columns]].pct_change().dropna()
                    asset_dots = pd.DataFrame({
                        "Volatility": daily.std() * np.sqrt(252) * 100,
                        "Return": daily.mean() * 252 * 100,
                    })
                    reverse_proxy = {v: k for k, v in proxy_map.items()}
                    asset_dots.index = [reverse_proxy.get(t, t) for t in asset_dots.index]

                    fig_ef = go.Figure()
                    fig_ef.add_trace(go.Scatter(
                        x=frontier_df["Volatility"], y=frontier_df["Return"],
                        mode="lines", name="Efficient Frontier",
                        line=dict(color="royalblue", width=3),
                        marker=dict(color=frontier_df["Sharpe"], colorscale="RdYlGn",
                                    showscale=True, colorbar=dict(title="Sharpe")),
                    ))
                    fig_ef.add_trace(go.Scatter(
                        x=asset_dots["Volatility"], y=asset_dots["Return"],
                        mode="markers+text", name="Individual Assets",
                        text=asset_dots.index, textposition="top center",
                        marker=dict(size=10, color="orange", symbol="diamond"),
                    ))
                    # Max Sharpe point
                    max_sr_idx = frontier_df["Sharpe"].idxmax()
                    fig_ef.add_trace(go.Scatter(
                        x=[frontier_df.loc[max_sr_idx, "Volatility"]],
                        y=[frontier_df.loc[max_sr_idx, "Return"]],
                        mode="markers+text", name="Max Sharpe",
                        text=["★ Max Sharpe"], textposition="top right",
                        marker=dict(size=16, color="gold", symbol="star"),
                    ))
                    fig_ef.update_layout(
                        xaxis_title="Risk (Volatility %)", yaxis_title="Return (CAGR %)",
                        height=480, hovermode="closest",
                    )
                    fig_ef.update_xaxes(ticksuffix="%")
                    fig_ef.update_yaxes(ticksuffix="%")
                    st.plotly_chart(fig_ef, use_container_width=True)

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
