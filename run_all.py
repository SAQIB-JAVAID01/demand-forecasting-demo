"""Run the whole project: download -> clean -> backtest -> model lab -> scenarios -> inventory -> score -> report."""
import urllib.request

from src import backtest, build_dashboard, config as C, evaluate, inventory, model_lab, prepare, report, scenarios

if __name__ == "__main__":
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not C.RAW_ZIP.exists() and not (C.PROCESSED_DIR / "transactions_raw.parquet").exists():
        print("Downloading", C.DATA_URL)
        urllib.request.urlretrieve(C.DATA_URL, C.RAW_ZIP)
    prepare.main()
    backtest.run()
    evaluate.main()
    model_lab.run()          # statistical models, XGBoost, HistGradientBoosting, engineered features, ensemble
    scenarios.run()          # 26-week projection and what-if economic scenarios
    inventory.run()          # replenishment simulation, same policy for forecast and benchmark
    report.main()
    build_dashboard.main()
