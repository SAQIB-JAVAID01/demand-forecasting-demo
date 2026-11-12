"""Checks of the headline KPI code shared by the desktop app and the web app."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src import kpis as K


def toy() -> pd.DataFrame:
    """Two products, two weeks. Product errors cancel inside each week for 'm'."""
    return pd.DataFrame({
        "sku": ["a", "b", "a", "b"], "week": pd.to_datetime(["2011-05-02"] * 2 + ["2011-05-09"] * 2),
        "y": [10.0, 10.0, 20.0, 20.0], "m": [15.0, 5.0, 25.0, 15.0], "ma13": [8.0, 8.0, 16.0, 16.0]})


def test_total_accuracy_lets_product_errors_cancel():
    df = toy()
    assert K.wape(df, "m") == pytest.approx(20 / 60)            # per product-week
    assert K.total_wape(df, "m") == pytest.approx(0.0)          # weekly totals are exact
    assert K.bias(df, "m") == pytest.approx(0.0)
    assert K.total_wape(df, "ma13") == pytest.approx(12 / 60)
    assert K.bias(df, "ma13") == pytest.approx(-12 / 60)


def test_headline_is_consistent_with_its_parts():
    h = K.headline(toy(), "m", "ma13")
    assert h["total_fva"] == pytest.approx(h["total_acc"] - h["total_acc_base"])
    assert h["product_fva"] == pytest.approx(h["product_acc"] - h["product_acc_base"])
    assert h["weeks"] == 2 and h["products"] == 2 and h["rows"] == 4


def test_verdict_needs_the_range_to_exclude_zero():
    assert K.verdict(0.01, 0.05) == "better"
    assert K.verdict(-0.05, -0.01) == "worse"
    assert K.verdict(-0.01, 0.05) == "level"


def test_bias_tag_uses_the_five_percent_target():
    assert K.bias_tag(-0.021).startswith("Unbiased")
    assert K.bias_tag(-0.23) == "Under-forecast"
    assert K.bias_tag(0.08) == "Over-forecast"


@pytest.fixture(scope="module")
def preds():
    path = C.PROCESSED_DIR / "predictions.parquet"
    if not path.exists():
        pytest.skip("backtest output not built")
    return pd.read_parquet(path)


def test_default_setting_is_the_unbiased_one_on_the_real_backtest(preds):
    settings = K.settings_for(preds)
    assert next(iter(settings)) == K.DEFAULT_SETTING
    fin = K.final_rows(preds)
    h = K.headline(fin, K.DEFAULT_SETTING)
    assert abs(h["bias"]) <= K.BIAS_TARGET                      # the label "Unbiased totals" has to be true
    lo, hi = h["total_ci"]
    assert lo <= h["total_fva"] <= hi
    assert h["total_verdict"] == K.verdict(lo, hi)


def test_filters_only_ever_remove_rows(preds):
    full = K.final_rows(preds)
    part = K.final_rows(preds, abc=["A"], horizons=[1])
    assert 0 < len(part) < len(full)
    assert set(part["abc"]) == {"A"} and set(part["h"]) == {1}
    assert not full["closed"].any() and (full["fold_type"] == "final").all()


def test_inventory_view_scales_money_with_the_assumptions():
    run = dict(lead=1, service=0.95,
               model_run=dict(weeks=26, lost_revenue=100.0), base_run=dict(lost_revenue=300.0, avg_inventory_value=1000.0),
               model_same_service=dict(avg_inventory_value=900.0))
    v = K.inventory_view(run, cost=0.5, hold=0.2)
    assert v["inv_change"] == pytest.approx(-0.10)
    assert v["saving"] == pytest.approx(100 * 0.5 * 0.2)
    assert v["lost_rev"] == pytest.approx(200 / 0.5)            # 26 weeks scaled to a year
    assert v["lost_margin"] == pytest.approx(200 * 0.5 / 0.5)
    assert np.isclose(K.inventory_view(run, 0.5, 0.4)["saving"], 2 * v["saving"])
