import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# === Anchor to project root ===
# Ensures figures are saved in the correct folder relative to this file
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

def _save_plot(fig, name: str):
    """Helper function to save and close plots consistently."""
    p = FIG_DIR / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {p}")

def plot_prices(prices: pd.DataFrame, title: str = "Asset Prices"):
    """
    Plots the normalized price history (Base 100).
    """
    # Normalize prices to start at 100 for better comparison
    normalized = (prices / prices.iloc[0]) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    normalized.plot(ax=ax, lw=1.5)
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price (Base=100)")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    _save_plot(fig, "prices")

def plot_rolling_vol(returns: pd.DataFrame, window: int = 63, title: str = "Rolling Volatility"):
    """
    Plots the annualized rolling volatility.
    """
    # Annualized Volatility = StdDev * sqrt(252)
    vol = returns.rolling(window).std() * (252**0.5)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    vol.plot(ax=ax, lw=1.5)
    
    ax.set_title(f"{title} (Window={window} days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    _save_plot(fig, "rolling_vol")

def plot_corr_heatmap(returns: pd.DataFrame, title: str = "Correlation Matrix"):
    """
    Plots a correlation heatmap using Seaborn.
    """
    corr = returns.corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use seaborn for a more professional look with annotations
    sns.heatmap(
        corr, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        vmin=-1, 
        vmax=1, 
        ax=ax,
        square=True
    )
    
    ax.set_title(title)
    
    # Adjust layout to prevent clipping of labels
    _save_plot(fig, "corr_heatmap")

def plot_drawdown_curves(prices: pd.DataFrame, title: str = "Drawdown Profile"):
    """
    Plots the historical drawdown curves.
    Drawdown = (Price / Peak) - 1
    """
    peak = prices.cummax()
    drawdown = (prices / peak) - 1.0