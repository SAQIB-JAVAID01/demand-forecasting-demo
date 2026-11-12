"""Leakage tests for the engineered features, and checks of the inventory simulation and scenario maths.

Run:  python -m pytest tests -q
"""
import numpy as np
import pandas as pd
import pytest

from src import inventory as INV
from src import macro
from src import scenarios as SC
from src.features_ext import EXT_HISTORY, build_features_ext
from tests.test_no_leakage import toy_panel


def toy_ext(n_weeks: int = 80):
    p = toy_panel(n_weeks=n_weeks)
    p["abc"] = p["sku"].map(lambda s: "ABC"[int(s[1:]) % 3])
    p["demand_type"] = np.where(p["sku"].isin(["S0", "S3"]), "intermittent", "regular")
    weeks = pd.DatetimeIndex(sorted(p["week"].unique()))
    rng = np.random.default_rng(1)
    m = pd.DataFrame({k: rng.normal(size=len(weeks)).cumsum() for k in SC.MACRO}, index=weeks)
    return p, m


@pytest.mark.parametrize("h", [1, 2, 3, 4])
def test_engineered_features_ignore_the_future(h):
    panel, m = toy_ext()
    cutoff = 60
    base = build_features_ext(panel, h, m)
    changed = panel.copy()
    future = changed["t"] > cutoff
    changed.loc[future, "units"] = changed.loc[future, "units"] * 5 + 50
    changed.loc[future, "price"] = 99.0
    m2 = m.copy()
    m2.loc[m2.index > panel.loc[panel["t"] == cutoff, "week"].iloc[0]] += 40.0      # the economy changes later
    alt = build_features_ext(changed, h, m2)
    for rows in (lambda f: f["t"] == cutoff + h, lambda f: f["t"] <= cutoff):
        a = base[rows(base)][EXT_HISTORY].to_numpy(float)
        b = alt[rows(alt)][EXT_HISTORY].to_numpy(float)
        assert np.allclose(a, b, equal_nan=True)


def test_economic_figures_wait_for_publication():
    mon = macro.monthly()
    w = macro.weekly(pd.DatetimeIndex(["2011-03-14", "2011-03-21"]))
    # January figures are public only 45 days after the end of January (17 March 2011)
    dec, jan = mon.loc["2010-12-01"], mon.loc["2011-01-01"]
    assert w.loc["2011-03-14", "unemployment"] == pytest.approx(dec["unemployment"])
    assert w.loc["2011-03-21", "unemployment"] == pytest.approx(jan["unemployment"])


def toy_grid(noise: float, seed: int = 0) -> INV.Grid:
    rng = np.random.default_rng(seed)
    n, T = 30, 60
    y = rng.poisson(20, (n, T)).astype(float)
    f = {"perfect": y.copy(), "noisy": np.maximum(y + rng.normal(0, noise, (n, T)), 0)}
    final = np.arange(T) >= 20
    return INV.Grid(np.arange(n), pd.date_range("2010-01-04", periods=T, freq="W-MON"), y, final, np.full(n, 2.0), f)


def test_perfect_forecast_of_steady_demand_serves_all_demand():
    g = toy_grid(8.0)
    g.y[:] = 20.0
    g.fc["perfect"] = g.y.copy()
    r = INV.simulate(g, "perfect", lead=2, z=1.645)
    assert r["fill_rate"] == pytest.approx(1.0)
    assert r["units_short"] == pytest.approx(0.0)


def test_more_safety_stock_means_higher_fill_and_more_inventory():
    g = toy_grid(8.0)
    lo, hi = INV.simulate(g, "noisy", 1, 0.5), INV.simulate(g, "noisy", 1, 2.0)
    assert hi["fill_rate"] > lo["fill_rate"]
    assert hi["avg_inventory_value"] > lo["avg_inventory_value"]


def test_money_scales_with_unit_cost():
    g = toy_grid(8.0)
    a, b = INV.simulate(g, "noisy", 2, 1.645, 1.0), INV.simulate(g, "noisy", 2, 1.645, 0.45)
    assert b["avg_inventory_value"] == pytest.approx(0.45 * a["avg_inventory_value"])
    assert b["fill_rate"] == pytest.approx(a["fill_rate"])


def test_scenario_multiplier_phases_in_and_has_the_right_sign():
    flat = SC.multiplier(26, SC.SCENARIOS["Base"], SC.ASSUMED_SENSITIVITY)
    assert np.allclose(flat, 1.0)
    rec = SC.multiplier(26, SC.SCENARIOS["Recession"], SC.ASSUMED_SENSITIVITY)
    boom = SC.multiplier(26, SC.SCENARIOS["Boom"], SC.ASSUMED_SENSITIVITY)
    assert (rec <= 1).all() and (boom >= 1).all()
    assert rec[0] > rec[SC.PHASE_IN_WEEKS - 1]                     # the change builds up ...
    assert np.allclose(rec[SC.PHASE_IN_WEEKS - 1:], rec[-1])       # ... then holds
    cut = SC.multiplier(26, SC.SCENARIOS["Base"], SC.ASSUMED_SENSITIVITY, price_change=-0.10, elasticity=-3.0)
    assert (cut >= 1).all() and cut[-1] > 1                        # a price cut lifts units
