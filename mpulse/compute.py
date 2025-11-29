import numpy as np
import pandas as pd
from typing import List, Tuple

def align_and_clean(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns trading dates and fills missing values.
    
    Uses forward-fill to propagate last valid observations (handling holidays),
    then drops any remaining NaNs at the beginning of the series.

    Args:
        prices (pd.DataFrame): Raw price data.

    Returns:
        pd.DataFrame: Cleaned data with no missing values.
    """
    # Sort by date
    df = prices.sort_index()
    # Drop rows where ALL columns are NaN
    df = df.dropna(how="all")
    
    df = df.ffill()
    # Drop initial rows that are still NaN (before the first valid price)
    df = df.dropna(how="any")
    return df


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes logarithmic returns: R_t = ln(P_t / P_{t-1}).

    Args:
        prices (pd.DataFrame): Adjusted close prices.

    Returns:
        pd.DataFrame: Log returns.
    """
    # Vectorized operation using NumPy's log
    return np.log(prices / prices.shift(1)).dropna()


def annualized_vol(returns: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """
    Computes annualized volatility (standard deviation * sqrt(T)).

    Args:
        returns (pd.DataFrame): Log returns.
        trading_days (int): Number of trading days per year (default 252).

    Returns:
        pd.Series: Annualized volatility for each asset.
    """
    return returns.std() * np.sqrt(trading_days)


def max_drawdown(prices: pd.DataFrame) -> pd.Series:
    """
    Computes the Maximum Drawdown (MDD) for the entire history.

    MDD = Min((Price / Rolling_Max) - 1)

    Args:
        prices (pd.DataFrame): Asset prices.

    Returns:
        pd.Series: The maximum percentage loss from a peak for each asset.
    """
    # 1. Calculate cumulative maximum (the "Peak" so far)
    peak = prices.cummax()
    # 2. Calculate drawdown series
    dd = (prices / peak) - 1.0
    # 3. Find the minimum value (deepest trough)
    # Note: We use .min() directly on the DataFrame for vectorized computation
    return dd.min()


def sharpe_ratio(returns: pd.DataFrame, rf_annual: float = 0.0, trading_days: int = 252) -> pd.Series:
    """
    Computes the annualized Sharpe Ratio.

    Sharpe = (Mean_Return - Risk_Free) / Volatility

    Args:
        returns (pd.DataFrame): Log returns.
        rf_annual (float): Annual risk-free rate (e.g., 0.02 for 2%).
        trading_days (int): Annualization factor.

    Returns:
        pd.Series: Sharpe ratio for each asset.
    """
    # Convert annual risk-free rate to daily equivalent approximation is negligible for ranking,
    # but here we usually adjust the numerator annually: (Mean * 252 - Rf)
    mean_ret_annual = returns.mean() * trading_days
    vol_annual = returns.std() * np.sqrt(trading_days)
    
    # Avoid division by zero
    if (vol_annual == 0).any():
        return pd.Series(np.nan, index=returns.columns)

    return (mean_ret_annual - rf_annual) / vol_annual


def drawdown_series_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the drawdown series over time.

    Args:
        prices (pd.DataFrame): Asset prices.

    Returns:
        pd.DataFrame: Drawdown values (negative floats) over time.
    """
    peak = prices.cummax()
    dd = (prices / peak) - 1.0
    return dd


def cumulative_return_from_logrets(log_rets: pd.DataFrame) -> pd.Series:
    """
    Computes total cumulative return from log returns.
    
    CumRet = exp(Sum(r)) - 1

    Args:
        log_rets (pd.DataFrame): Log returns.

    Returns:
        pd.Series: Total return (e.g., 0.5 means 50% growth).
    """
    return np.exp(log_rets.sum()) - 1.0


def period_performance(
    prices: pd.DataFrame,
    periods: List[Tuple[str, str, str]],
    trading_days: int = 252,
    rf_annual: float = 0.0,
) -> pd.DataFrame:
    """
    Computes performance metrics (CumRet, Vol, Sharpe) for specific time windows.

    Args:
        prices (pd.DataFrame): Price data.
        periods (List[Tuple]): List of (Label, StartDate, EndDate).
        trading_days (int): Annualization factor.
        rf_annual (float): Risk-free rate.

    Returns:
        pd.DataFrame: A formatted table with MultiIndex columns.
    """
    assets = list(prices.columns)
    metrics = ["CumRet", "AnnVol", "Sharpe"]
    
    # MultiIndex for columns: Metric -> Asset
    cols = pd.MultiIndex.from_product([metrics, assets], names=["Metric", "Asset"])
    out = pd.DataFrame(index=[p[0] for p in periods], columns=cols, dtype=float)

    for label, start, end in periods:
        # Slicing with .loc handles dates intelligently
        subp = prices.loc[start:end]
        
        if subp.empty:
            continue
            
        rets = log_returns(subp)
        
        # Compute metrics
        cumret = cumulative_return_from_logrets(rets)
        vol = annualized_vol(rets, trading_days)
        sharpe = sharpe_ratio(rets, rf_annual, trading_days)
        
        # Assign to the output table
        # Using slice(None) effectively selects all assets for that metric
        out.loc[label, ("CumRet", slice(None))] = cumret.values
        out.loc[label, ("AnnVol", slice(None))] = vol.values
        out.loc[label, ("Sharpe", slice(None))] = sharpe.values

    return out