from pathlib import Path
from typing import List
import pandas as pd
import yfinance as yf

# === anchor to project root ===
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

def download_prices(tickers: List[str], start: str = "2015-01-01") -> pd.DataFrame:
    """
    Downloads historical adjusted close prices from Yahoo Finance.
    
    Setting threads=False to avoid 'database is locked' errors on macOS.
    """
    print(f"Downloading data for: {tickers}...")
    
    try:
        # threads=False is CRITICAL here to avoid SQLite locking errors
        df = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=False)
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Yahoo Finance API: {e}")

    if df.empty:
        raise RuntimeError("API returned empty data. Check your internet or tickers.")

    # Handle MultiIndex columns if necessary
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df["Close"]
        except KeyError:
             pass

    if isinstance(df, pd.Series):
        df = df.to_frame()

    # Data Manipulation: Forward fill and drop NaNs
    df = df.ffill().dropna()

    if df.empty:
        raise RuntimeError("Data is empty after cleaning (check if tickers share trading dates).")

    return df

def save_csv(df: pd.DataFrame, name: str) -> Path:
    """Saves DataFrame to CSV."""
    p = DATA_DIR / f"{name}.csv"
    try:
        df.to_csv(p, index=True)
        print(f"Saved cache to: {p}")
        return p
    except OSError as e:
        print(f"Warning: Could not save CSV to {p}. Error: {e}")
        return p

def load_csv(name: str) -> pd.DataFrame:
    """Loads DataFrame from CSV."""
    p = DATA_DIR / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Data file not found at {p}. Please run demo.py first.")
    
    return pd.read_csv(p, parse_dates=[0], index_col=0)