import sys
from pathlib import Path
from mpulse.data_io import download_prices, save_csv
from mpulse.compute import (
    align_and_clean,
    log_returns,
    annualized_vol,
    max_drawdown,
    sharpe_ratio,
    period_performance,
)
from mpulse.viz import plot_prices, plot_rolling_vol, plot_corr_heatmap, plot_drawdown_curves

def main():
    # ---- Setup: Ensure output directories exist ----
    Path("figures").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    # ---- 1. Configuration: Select Assets ----
    tickers = ["MCHI", "FXI", "KWEB", "ASHR", "SPY"]
    print(f"Starting analysis for: {tickers}")

    # ---- 2. Data Ingestion & Cleaning ----

    prices = download_prices(tickers, start="2018-01-01")
    prices = align_and_clean(prices)
    save_csv(prices, "clean_prices")
    print("Data downloaded and cleaned successfully.")

    # ---- 3. Basic Metrics Calculation ----
    rets = log_returns(prices)
    
    print("\n--- Annualized Volatility ---")
    print(annualized_vol(rets).round(3))
    
    print("\n--- Sharpe Ratio (Rf=0) ---")
    print(sharpe_ratio(rets).round(3))
    
    print("\n--- Max Drawdown ---")
    print(max_drawdown(prices).round(3))

    # ---- 4. Visualization Generation ----
    
    print("\nGenerating plots...")
    plot_prices(prices, "China Focus: Prices")
    plot_rolling_vol(rets, 63, "China Focus: Rolling Vol (63d)")
    plot_corr_heatmap(rets, "China Focus: Correlation")
    plot_drawdown_curves(prices, "China Focus: Drawdown Curves")

    # ---- 5. Period Analysis ----
    # Define specific market phases for deeper analysis
    last = prices.index.max().strftime("%Y-%m-%d")
    periods = [
        ("Pre-COVID", "2018-01-01", "2019-12-31"),
        ("COVID Shock", "2020-01-01", "2020-12-31"),
        ("Post-Peak Tightening", "2021-01-01", "2022-12-31"),
        ("Recent", "2023-01-01", last),
    ]
    
    perf = period_performance(prices, periods).round(3)
    print("\n--- Period Performance (CumRet / AnnVol / Sharpe) ---")
    print(perf)

    print("\n[Done] Figures saved to ./figures ; Cleaned prices to ./data/clean_prices.csv")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during execution: {e}")
        sys.exit(1)