import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="NSE Portfolio Analytics",
    page_icon="📊",
    layout="wide"
)

# =========================
# HEADER
# =========================
col_title, col_btn = st.columns([5, 1])

with col_title:
    st.title("📊 Quant Portfolio Dashboard")

# =========================
# STOCK UNIVERSE
# =========================
stocks = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "LT.NS",
    "BHARTIARTL.NS", "ASIANPAINT.NS"
]

selected = st.multiselect(
    "Select Stocks",
    stocks,
    default=["RELIANCE.NS", "TCS.NS", "INFY.NS"]
)

start = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
end = st.date_input("End Date", pd.to_datetime("today"))

if len(selected) == 0:
    st.warning("Select at least one stock")
    st.stop()

# =========================
# DATA DOWNLOAD
# =========================
data = yf.download(
    selected,
    start=start,
    end=end,
    auto_adjust=True
)["Close"]

# FIX: ensure DataFrame
if isinstance(data, pd.Series):
    data = data.to_frame()

data = data.dropna(how="all")

returns = data.pct_change().dropna()

# =========================
# NIFTY DATA
# =========================
nifty = yf.download(
    "^NSEI",
    start=start,
    end=end,
    auto_adjust=True
)["Close"].squeeze().dropna()

nifty_returns = nifty.pct_change().reindex(returns.index).fillna(0)
nifty_cum = (1 + nifty_returns).cumprod()

# =========================
# WEIGHTS
# =========================
st.subheader("Portfolio Allocation")

weights = []
cols = st.columns(len(selected))

for i, s in enumerate(selected):
    weights.append(
        cols[i].number_input(s, 0.0, 100.0, 100/len(selected))
    )

weights = np.array(weights)
weights = weights / weights.sum()

# =========================
# PORTFOLIO CALCULATIONS
# =========================
portfolio_returns = returns.dot(weights)
portfolio_cum = (1 + portfolio_returns).cumprod()

drawdown = portfolio_cum / portfolio_cum.cummax() - 1

annual_return = np.sum(returns.mean() * weights) * 252
annual_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
sharpe = annual_return / annual_vol

portfolio_total = float(portfolio_cum.iloc[-1] - 1)
nifty_total = float(nifty_cum.iloc[-1] - 1)
alpha = portfolio_total - nifty_total

# =========================
# DOWNLOAD REPORT (TOP RIGHT)
# =========================
report = pd.DataFrame({
    "Stock": selected,
    "Weight": weights
})

report.loc[len(report)] = ["Portfolio Return", portfolio_total]
report.loc[len(report)] = ["NIFTY Return", nifty_total]
report.loc[len(report)] = ["Alpha", alpha]

csv = report.to_csv(index=False).encode("utf-8")

with col_btn:
    st.download_button(
        "⬇ Export",
        data=csv,
        file_name="portfolio_report.csv",
        use_container_width=True
    )

# =========================
# KPI CARDS
# =========================
st.markdown("### Performance Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Annual Return", f"{annual_return:.2%}")
c2.metric("Volatility", f"{annual_vol:.2%}")
c3.metric("Sharpe Ratio", f"{sharpe:.2f}")

# =========================
# BENCHMARK
# =========================
st.markdown("### Benchmark Comparison")

b1, b2, b3 = st.columns(3)

b1.metric("Portfolio Return", f"{portfolio_total:.2%}")
b2.metric("NIFTY Return", f"{nifty_total:.2%}")
b3.metric("Alpha", f"{alpha:.2%}")

# =========================
# 1. HISTORICAL STOCK PRICES (SEPARATE)
# =========================
st.markdown("### 📈 Historical Stock Prices")

fig_prices = go.Figure()

for col in data.columns:
    fig_prices.add_trace(go.Scatter(
        x=data.index,
        y=data[col],
        mode="lines",
        name=col,
        hovertemplate="%{y:.2f}<extra>%{fullData.name}</extra>"
    ))

fig_prices.update_layout(
    hovermode="x unified",
    template="plotly_dark",
    height=500,
    title="Stock Price Movement"
)

st.plotly_chart(fig_prices, use_container_width=True)

# =========================
# 2. PORTFOLIO vs NIFTY (SEPARATE)
# =========================
st.markdown("### 📊 Portfolio Growth vs NIFTY 50")

fig_compare = go.Figure()

fig_compare.add_trace(go.Scatter(
    x=portfolio_cum.index,
    y=portfolio_cum,
    name="Portfolio",
    line=dict(width=3)
))

fig_compare.add_trace(go.Scatter(
    x=nifty_cum.index,
    y=nifty_cum,
    name="NIFTY 50",
    line=dict(dash="dash", width=2)
))

fig_compare.update_layout(
    hovermode="x unified",
    template="plotly_dark",
    height=500,
    title="Performance Comparison"
)

st.plotly_chart(fig_compare, use_container_width=True)

# =========================
# DRAWDOWN
# =========================
st.markdown("### Drawdown Analysis")

fig_dd = go.Figure()

fig_dd.add_trace(go.Scatter(
    x=drawdown.index,
    y=drawdown,
    fill="tozeroy",
    line=dict(color="red")
))

fig_dd.update_layout(template="plotly_dark", height=350)

st.plotly_chart(fig_dd, use_container_width=True)

# =========================
# ROLLING RETURNS
# =========================
st.markdown("### Rolling Returns (30D)")

st.line_chart(portfolio_returns.rolling(30).mean())

# =========================
# CONTRIBUTION ANALYSIS
# =========================
st.markdown("### Stock Contribution")

contrib = (returns.mean() * weights) * 252

contrib_df = pd.DataFrame({
    "Stock": selected,
    "Contribution": contrib
}).sort_values("Contribution", ascending=False)

st.dataframe(contrib_df, use_container_width=True)

fig_bar = px.bar(contrib_df, x="Stock", y="Contribution")
st.plotly_chart(fig_bar, use_container_width=True)

# =========================
# EFFICIENT FRONTIER
# =========================
st.markdown("### Efficient Frontier")

cov = returns.cov() * 252

points = []

for _ in range(3000):
    w = np.random.random(len(selected))
    w /= w.sum()

    r = np.sum(returns.mean() * w) * 252
    v = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    s = r / v

    points.append([r, v, s])

frontier = pd.DataFrame(points, columns=["Return", "Volatility", "Sharpe"])

best = frontier.loc[frontier["Sharpe"].idxmax()]

fig_f = px.scatter(frontier, x="Volatility", y="Return", color="Sharpe")

fig_f.add_trace(go.Scatter(
    x=[best["Volatility"]],
    y=[best["Return"]],
    mode="markers",
    marker=dict(size=14, symbol="star"),
    name="Max Sharpe"
))

st.plotly_chart(fig_f, use_container_width=True)
