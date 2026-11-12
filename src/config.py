"""Project settings. Every number that shapes the result lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_ZIP = RAW_DIR / "online_retail_II.zip"
DATA_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

# Series selection (uses ONLY the first INITIAL_TRAIN_WEEKS weeks, so no
# information from the backtest period decides which products are forecast).
N_SERIES = 200
MIN_ACTIVE_WEEKS = 26

# Rolling-origin backtest
INITIAL_TRAIN_WEEKS = 52   # first forecast is made after one full year
HORIZON = 4                # forecast 1 to 4 weeks ahead
STEP = 4                   # move the forecast date forward 4 weeks each fold
N_SELECT_FOLDS = 5         # earliest folds: used to choose model settings
                           # remaining folds: final test, never used for choices

# Prediction interval
Q_LO, Q_HI = 0.10, 0.90    # 80% interval

SEED = 42
N_BOOTSTRAP = 2000
