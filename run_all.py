"""Run the whole project: download -> clean -> backtest -> score -> report."""
import urllib.request

from src import backtest, build_dashboard, config as C, evaluate, prepare, report

if __name__ == "__main__":
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not C.RAW_ZIP.exists() and not (C.PROCESSED_DIR / "transactions_raw.parquet").exists():
        print("Downloading", C.DATA_URL)
        urllib.request.urlretrieve(C.DATA_URL, C.RAW_ZIP)
    prepare.main()
    backtest.run()
    evaluate.main()
    report.main()
    build_dashboard.main()
