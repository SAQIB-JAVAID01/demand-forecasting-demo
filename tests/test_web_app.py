"""The web app runs every page without an error and shows the same headline number as the shared KPI code."""
import secrets
from pathlib import Path

import pandas as pd
import pytest

from src import config as C
from src import kpis as K

APP = str(Path(__file__).resolve().parents[1] / "web" / "streamlit_app.py")
# Throwaway values made up for each test run; the real demo credentials exist only in the hosting secrets.
USER, PASSWORD = "tester", secrets.token_urlsafe(16)
PAGES = ["Executive summary", "Forecast accuracy", "Inventory simulation", "Sales projection", "Data and method"]


@pytest.fixture(scope="module")
def at():
    testing = pytest.importorskip("streamlit.testing.v1")
    if not (C.PROCESSED_DIR / "predictions.parquet").exists():
        pytest.skip("backtest output not built")
    app = testing.AppTest.from_file(APP, default_timeout=120)
    app.secrets["demo_user"], app.secrets["demo_password"] = USER, PASSWORD
    app.run()
    sign_in(app, USER, PASSWORD)
    return app


def sign_in(app, user, password):
    app.text_input[0].input(user)
    app.text_input[1].input(password)
    app.button[0].click().run()


def fresh_app(with_secrets: bool):
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(APP, default_timeout=120)
    if with_secrets:
        app.secrets["demo_user"], app.secrets["demo_password"] = USER, PASSWORD
    return app.run()


def test_gate_refuses_every_login_without_secrets():
    app = fresh_app(with_secrets=False)
    assert not app.exception
    assert len(app.text_input) == 2 and not app.metric
    sign_in(app, "", "")
    assert app.error and not app.metric
    sign_in(app, USER, PASSWORD)
    assert app.error and not app.metric


def test_gate_refuses_wrong_password():
    app = fresh_app(with_secrets=True)
    sign_in(app, USER, PASSWORD + "x")
    assert app.error and not app.metric


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
