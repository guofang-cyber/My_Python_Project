# app/app.py
# -----------------------------------------
# Market Pulse CN - Streamlit Dashboard
# -----------------------------------------
import sys
from pathlib import Path
from typing import Optional, Union, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---- Make project package importable (mpulse/) ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- Try to import load_csv (mpulse.data_io) ----
try:
    from mpulse.data_io import load_csv as _load_csv
except ImportError:
    _load_csv = None

# ---- Try to import your own lib functions; fall back to inline implementations ----
try:
    from mpulse.compute import (
        log_returns as _log_returns,
        annualized_vol as _annualized_vol,
        max_drawdown as _max_drawdown,
        sharpe_ratio as _sharpe_ratio,
        period_performance as _period_performance,
    )
except ImportError:
    _log_returns = _annualized_vol = _max_drawdown = _sharpe_ratio = _period_performance = None


# ---------- Inline metric implementations (fallback) ----------
def _fallback_log_returns(prices: pd.Series) -> pd.Series:
    """Computes log returns for a price series."""
    return np.log(prices).diff().dropna()

def _fallback_annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Computes annualized volatility fallback."""
    if len(returns) == 0:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))

def _fallback_max_drawdown(prices: pd.Series) -> float:
    """Computes maximum drawdown fallback."""
    if prices.empty:
        return np.nan
    cummax = prices.cummax()
    dd = prices / cummax - 1.0
    return float(dd.min())

def _fallback_sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """Computes Sharpe Ratio fallback."""
    if returns.empty:
        return np.nan
    
    # Convert annual risk-free rate to period rate approximation
    rf_per_period = (1 + rf) ** (1 / periods_per_year) - 1
    excess = returns - rf_per_period
    vol = excess.std(ddof=1)
    
    if vol == 0 or np.isnan(vol):
        return np.nan
        
    return float((excess.mean() / vol) * np.sqrt(periods_per_year))

def _fallback_period_performance(prices: pd.Series) -> float:
    """Computes simple return (end / start - 1)."""
    if len(prices) < 2:
        return np.nan
    return float(prices.iloc[-1] / prices.iloc[0] - 1.0)


# ---------- Choose impls / wrappers ----------
# Prefer library functions, fallback to local implementation if import failed
log_returns = _log_returns or _fallback_log_returns
annualized_vol = _annualized_vol or _fallback_annualized_vol

def max_drawdown_safe(prices: Union[pd.Series, pd.DataFrame]) -> float:
    """
    Wrapper for Maximum Drawdown calculation.
    
    Uses fallback for Series inputs, library function for DataFrames if available.
    """
    if isinstance(prices, pd.Series):
        return _fallback_max_drawdown(prices)
    fn = _max_drawdown or _fallback_max_drawdown
    return float(fn(prices))

def sharpe_ratio_func(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Unified wrapper for Sharpe Ratio, handling different function signatures.
    
    Args:
        returns (pd.Series): Return series.
        rf (float): Risk-free rate (annual).
        periods_per_year (int): Annualization factor.
    """
    if _sharpe_ratio is not None:
        # Try calling with keyword 'rf'
        try:
            return float(_sharpe_ratio(returns, rf_annual=rf, trading_days=periods_per_year))
        except TypeError:
            pass
        # Try positional arguments as last resort
        try:
            return float(_sharpe_ratio(returns, rf, periods_per_year))
        except TypeError:
            # Fallback to local implementation if signature mismatch
            pass

    return float(_fallback_sharpe_ratio(returns, rf=rf, periods_per_year=periods_per_year))


def period_performance_func(prices: Union[pd.Series, pd.DataFrame], periods: Optional[int] = None) -> float:
    """
    Unified wrapper for period performance.
    
    - For Series inputs: returns simple start-to-end return.
    - For DataFrame inputs: delegates to library function if available.
    """
    if isinstance(prices, pd.Series):
        if len(prices) < 2:
            return float("nan")
        return float(prices.iloc[-1] / prices.iloc[0] - 1.0)

    fn = _period_performance or _fallback_period_performance
    try:
        return float(fn(prices, periods))
    except TypeError:
        return float(fn(prices))


# ---------- Streamlit page config ----------
st.set_page_config(
    page_title="Market Pulse CN",
    page_icon="📈",
    layout="wide",
)

# ---------- Caching data loader ----------
@st.cache_data(show_spinner=False)
def load_prices_df() -> pd.DataFrame:
    """
    Load cleaned price data from CSV.
    Parses dates and sets the index.
    """
    csv_path = PROJECT_ROOT / "data" / "clean_prices.csv"
    
    # Use library loader if available
    if _load_csv is not None:
        try:
            df = _load_csv(csv_path.stem) # library expects name without extension
            return df
        except Exception:
            pass # Fallback if library load fails
            
    # Direct Pandas fallback
    if not csv_path.exists():
        st.error(f"Data file not found at {csv_path}. Please run 'python demo.py' first to generate data.")
        st.stop()
        
    df = pd.read_csv(csv_path)
    # Parse first column as date and set index
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    return df


# ---------- Session State ----------
if "favorite" not in st.session_state:
    st.session_state.favorite = None

# ---------- Data ----------
with st.spinner("Loading data..."):
    prices_wide = load_prices_df()

assets = prices_wide.columns.tolist()

# ---------- Sidebar Controls ----------
st.sidebar.header("Controls")

# Default asset selection logic
default_asset = st.session_state.favorite if st.session_state.favorite in assets else (assets[0] if assets else None)
asset = st.sidebar.selectbox("Asset / Ticker", assets, index=assets.index(default_asset) if default_asset else 0)

date_min = prices_wide.index.min()
date_max = prices_wide.index.max()

# Robust date input handling
d_range = st.sidebar.date_input(
    "Date range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max
)

# Handle cases where user selects only one date
if isinstance(d_range, tuple):
    if len(d_range) == 2:
        d1, d2 = d_range
    else:
        d1 = d_range[0]
        d2 = d_range[0]
else:
    d1 = d_range
    d2 = d_range

rf = st.sidebar.number_input(
    "Risk-free (annual, %)", 
    value=2.0, 
    step=0.25, 
    help="Annual risk-free rate in percentage points."
)
btn_fav = st.sidebar.button("⭐ Set as favorite")

if btn_fav and asset:
    st.session_state.favorite = asset
    st.sidebar.success(f"Saved favorite: {asset}")

show_raw = st.sidebar.checkbox("Show raw data", value=False)
rolling_win = st.sidebar.number_input("Rolling window (days)", value=20, min_value=5, max_value=252, step=5)

# ---------- Filtered Series ----------
mask = (prices_wide.index >= pd.to_datetime(d1)) & (prices_wide.index <= pd.to_datetime(d2))
series = prices_wide.loc[mask, asset].dropna()

st.title("📈 Market Pulse (CN)")
st.caption("Interactive dashboard for quick asset diagnostics.")

if st.session_state.favorite:
    st.info(f"Favorite asset: **{st.session_state.favorite}**")

if series.empty:
    st.warning("No data available for the selected period.")
    st.stop()

# ---------- Compute metrics ----------
rets = log_returns(series)
ann_vol = annualized_vol(rets)
mdd = max_drawdown_safe(series)
sr = sharpe_ratio_func(rets, rf=rf / 100.0)
perf = period_performance_func(series, periods=252)

# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Period Return", f"{perf*100:,.2f}%")
c2.metric("Ann. Volatility", f"{ann_vol*100:,.2f}%")
c3.metric("Max Drawdown", f"{mdd*100:,.2f}%")
c4.metric("Sharpe Ratio", f"{sr:,.2f}")

st.divider()

# ---------- Charts ----------
# Price chart
st.subheader(f"Price • {asset}")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(series.index, series.values)
ax1.set_xlabel("Date")
ax1.set_ylabel("Price")
ax1.set_title(f"{asset} Price")
ax1.grid(True, linestyle="--", alpha=0.5)
fig1.tight_layout()
st.pyplot(fig1)

# Drawdown chart
st.subheader("Drawdown")
roll_max = series.cummax()
dd = series / roll_max - 1.0
fig2, ax2 = plt.subplots(figsize=(10, 3.5))
ax2.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
ax2.plot(dd.index, dd.values, color="red", lw=1)
ax2.set_xlabel("Date")
ax2.set_ylabel("Drawdown")
ax2.set_title("Drawdown (from peak)")
ax2.grid(True, linestyle="--", alpha=0.5)
fig2.tight_layout()
st.pyplot(fig2)

# Rolling volatility
st.subheader(f"Rolling Volatility ({rolling_win}d)")
rolling_rets = rets.rolling(rolling_win).std(ddof=1) * np.sqrt(252)
fig3, ax3 = plt.subplots(figsize=(10, 3.5))
ax3.plot(rolling_rets.index, rolling_rets.values, color="orange")
ax3.set_xlabel("Date")
ax3.set_ylabel("Ann. Vol")
ax3.set_title("Rolling Annualized Volatility")
ax3.grid(True, linestyle="--", alpha=0.5)
fig3.tight_layout()
st.pyplot(fig3)

# ---------- Data table / download ----------
if show_raw:
    st.subheader("Raw Prices (filtered)")
    st.dataframe(series.to_frame(name=asset))

# ---- Download button ----
st.download_button(
    label="Download filtered prices (CSV)",
    data=series.to_csv().encode("utf-8"),
    file_name=f"{asset}_prices_{pd.to_datetime(d1):%Y%m%d}_{pd.to_datetime(d2):%Y%m%d}.csv",
    mime="text/csv",
)

st.caption("Tip: Set your favorite asset in the sidebar to remember it for this session.")