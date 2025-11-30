# Market Pulse — China Focus

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Market Pulse** is a quantitative finance project analyzing the performance of China-related assets against the US market. It leverages the **Yahoo Finance API** to fetch historical data and performs statistical analysis to compute risk-adjusted returns, correlation structures, and volatility dynamics.

The project includes a reproducible pipeline (`demo.py`), a presentation notebook (`presentation/final.ipynb`), and an interactive web dashboard (`app/app.py`).

---

## 📂 Project Structure

The codebase follows a modular design pattern to ensure scalability and maintainability.

```text
market-pulse-cn/
├── .gitignore
├── README.md
├── requirements.txt
├── demo.py                 # CLI Entry point: Runs the full analysis pipeline
├── app/
│   └── app.py              # Interactive Dashboard (Streamlit)
├── mpulse/                 # Core Package (Business Logic)
│   ├── __init__.py
│   ├── data_io.py          # Data ingestion & error handling
│   ├── compute.py          # Scientific computing (NumPy/Pandas)
│   ├── viz.py              # Visualization (Matplotlib)

├── presentation/
│   ├── figures/            # Notebook-generated assets
│   └── final.ipynb         # Presentation layer (Jupyter Notebook)
├── data/                   # Local cache for CSV data
└── figures/                # Pipeline-generated charts