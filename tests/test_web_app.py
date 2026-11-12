"""The web app runs every page without an error and shows the same headline number as the shared KPI code."""
from pathlib import Path

import pandas as pd
import pytest

from src import config as C
from src import kpis as K

APP = str(Path(__file__).resolve().parents[1] / "web" / "streamlit_app.py")
PAGES = ["Executive summary", "Forecast accuracy", "Inventory simulation", "Sales projection", "Data and method"]


@pytest.fixture(scope="module")
def at():
    testing = pytest.importorskip("streamlit.testing.v1")
    if not (C.PROCESSED_DIR / "predictions.parquet").exists():
        pytest.skip("backtest output not built")
    return testing.AppTest.from_file(APP, default_timeout=120).run()


def test_summary_leads_with_weekly_total_accuracy(at):
    assert not at.exception
    preds = pd.read_parquet(C.PROCESSED_DIR / "predictions.parquet")
    h = K.headline(K.final_rows(preds), K.DEFAULT_SETTING)
    first = at.metric[0]
    assert first.label == "Weekly-total accuracy"
    assert first.value == f"{100 * h['total_acc']:.1f}%"


@pytest.mark.parametrize("page", PAGES[1:])
def test_other_pages_run(at, page):
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception
    assert at.title[0].value
