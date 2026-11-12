"""The forecast for week t made h weeks ahead must not depend on anything
that happened after week t-h. These tests change the future and check that the
features (and the fold layout) do not move.

Run:  python -m pytest tests -q
"""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.backtest import make_folds
from src.features import HISTORY_FEATURES, build_features


def toy_panel(n_sku: int = 6, n_weeks: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2010-01-04", periods=n_weeks, freq="W-MON")
    rows = []
    for s in range(n_sku):
        units = rng.poisson(20 + 5 * s, n_weeks).astype(float)
        units[rng.random(n_weeks) < 0.15] = 0
        for t, (w, u) in enumerate(zip(weeks, units)):
            rows.append((f"S{s}", w, t, u, max(1.0, u // 3), 2.0 + 0.1 * s + 0.01 * t, 5, False))
    return pd.DataFrame(rows, columns=["sku", "week", "t", "units", "orders", "price",
                                       "trading_days", "closed"])


@pytest.mark.parametrize("h", [1, 2, 3, 4])
def test_history_features_ignore_the_future(h):
    panel = toy_panel()
    cutoff = 60
    base = build_features(panel, h)

    changed = panel.copy()
    future = changed["t"] > cutoff
    changed.loc[future, "units"] = changed.loc[future, "units"] * 7 + 100
    changed.loc[future, "orders"] = 99
    changed.loc[future, "price"] = 123.0
    alt = build_features(changed, h)

    # rows the model would be asked to forecast from this cutoff: t = cutoff + h
    a = base[base["t"] == cutoff + h][HISTORY_FEATURES].to_numpy(float)
    b = alt[alt["t"] == cutoff + h][HISTORY_FEATURES].to_numpy(float)
    assert np.allclose(a, b, equal_nan=True)

    # and every training row (target week <= cutoff) is untouched as well
    a = base[base["t"] <= cutoff][HISTORY_FEATURES + ["y"]].to_numpy(float)
    b = alt[alt["t"] <= cutoff][HISTORY_FEATURES + ["y"]].to_numpy(float)
    assert np.allclose(a, b, equal_nan=True)


def test_newest_value_used_is_h_weeks_old():
    panel = toy_panel()
    for h in (1, 4):
        f = build_features(panel, h)
        row = f[(f["sku"] == "S0") & (f["t"] == 50)].iloc[0]
        truth = panel[(panel["sku"] == "S0") & (panel["t"] == 50 - h)]["units"].iloc[0]
        assert row["last"] == truth


def test_folds_do_not_overlap_and_selection_comes_first():
    folds = make_folds(104)
    cut = [f["cutoff_t"] for f in folds]
    assert cut[0] == C.INITIAL_TRAIN_WEEKS - 1
    assert all(b - a == C.STEP for a, b in zip(cut, cut[1:]))
    assert cut[-1] + C.HORIZON <= 103
    kinds = [f["fold_type"] for f in folds]
    assert kinds[:C.N_SELECT_FOLDS] == ["selection"] * C.N_SELECT_FOLDS
    assert set(kinds[C.N_SELECT_FOLDS:]) == {"final"}
    # every target week is forecast exactly once
    targets = [c + h for c in cut for h in range(1, C.HORIZON + 1)]
    assert len(targets) == len(set(targets))
