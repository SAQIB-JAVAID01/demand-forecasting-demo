"""Step 1: raw transactions -> clean weekly demand panel.

Run:  python -m src.prepare
Writes data/processed/{transactions.parquet, panel.parquet, calendar.parquet,
cleaning_log.json}.
"""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd

from . import config as C


def load_raw() -> pd.DataFrame:
    """Read both sheets of the UCI workbook (cached as parquet after first run)."""
    cache = C.PROCESSED_DIR / "transactions_raw.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    with zipfile.ZipFile(C.RAW_ZIP) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".xlsx"))
        xlsx = C.RAW_DIR / name
        if not xlsx.exists():
            z.extract(name, C.RAW_DIR)

    sheets = pd.ExcelFile(xlsx, engine="openpyxl").sheet_names
    frames = []
    for i, sheet in enumerate(sheets):
        df = pd.read_excel(xlsx, sheet_name=sheet, engine="openpyxl",
                           dtype={"Invoice": str, "StockCode": str})
        df["sheet"] = i
        frames.append(df)
        print(f"  read sheet '{sheet}': {len(df):,} rows")
    raw = pd.concat(frames, ignore_index=True)
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    raw = raw.rename(columns={"stockcode": "sku", "invoicedate": "date"})
    raw["description"] = raw["description"].astype("string")
    raw["country"] = raw["country"].astype("string")
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(cache, index=False)
    xlsx.unlink()  # keep the disk footprint small; the zip stays
    return raw


def remove_matched_cancellations(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    """A cancellation (invoice starting with 'C') reverses an earlier sale.

    Dropping only the 'C' rows would leave the original sale in the data as
    fake demand (one product has an 80,995-unit order that was cancelled the
    same day). So each cancellation is matched to the most recent earlier sale
    with the same customer, product and quantity, and that sale is removed too.
    """
    is_cancel = df["invoice"].str.startswith("C", na=False)
    sales = df[~is_cancel].copy()
    cancels = df[is_cancel & (df["quantity"] < 0) & df["customer_id"].notna()]

    key_cols = ["customer_id", "sku", "quantity"]
    ckeys = cancels.assign(quantity=-cancels["quantity"])[key_cols + ["date"]]
    wanted = set(map(tuple, ckeys[key_cols].itertuples(index=False, name=None)))

    cand = sales[sales["customer_id"].notna() & (sales["quantity"] > 0)]
    cand_keys = list(map(tuple, cand[key_cols].itertuples(index=False, name=None)))
    mask = np.fromiter((k in wanted for k in cand_keys), bool, len(cand_keys))
    cand = cand[mask].sort_values("date")

    pool: dict[tuple, list] = {}
    for idx, k, d in zip(cand.index, map(tuple, cand[key_cols].itertuples(index=False, name=None)),
                         cand["date"]):
        pool.setdefault(k, []).append((d, idx))

    drop, matched = [], 0
    for k, d in zip(map(tuple, ckeys[key_cols].itertuples(index=False, name=None)),
                    ckeys["date"]):
        lst = pool.get(k)
        if not lst:
            continue
        # most recent sale at or before the cancellation
        j = len(lst) - 1
        while j >= 0 and lst[j][0] > d:
            j -= 1
        if j >= 0:
            drop.append(lst.pop(j)[1])
            matched += 1

    log["cancellation_rows"] = int(is_cancel.sum())
    log["cancellations_matched_to_a_sale"] = matched
    log["units_removed_as_cancelled_sales"] = int(sales.loc[drop, "quantity"].sum())
    return sales.drop(index=drop)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log: dict = {"raw_rows": int(len(raw))}

    # The two sheets overlap for 1-9 Dec 2010. Keep that period from sheet 2 only.
    start2 = raw.loc[raw["sheet"] == 1, "date"].min().normalize()
    dup = (raw["sheet"] == 0) & (raw["date"] >= start2)
    log["rows_dropped_sheet_overlap"] = int(dup.sum())
    df = raw[~dup].copy()

    df = remove_matched_cancellations(df, log)

    bad = (df["quantity"] <= 0) | (df["price"] <= 0)
    log["rows_dropped_nonpositive_qty_or_price"] = int(bad.sum())
    df = df[~bad]

    # Real products have codes like 85123A. Postage, fees, samples etc. do not.
    is_product = df["sku"].str.fullmatch(r"\d{5}[A-Za-z]{0,2}", na=False)
    log["rows_dropped_non_product_codes"] = int((~is_product).sum())
    df = df[is_product].copy()
    df["sku"] = df["sku"].str.upper()

    df["revenue"] = df["quantity"] * df["price"]
    log["clean_rows"] = int(len(df))
    log["clean_units"] = int(df["quantity"].sum())
    log["date_min"] = str(df["date"].min().date())
    log["date_max"] = str(df["date"].max().date())
    return df[["invoice", "sku", "description", "quantity", "price", "revenue",
               "date", "customer_id", "country"]], log


def weekly_panel(tx: pd.DataFrame, log: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    tx = tx.copy()
    tx["week"] = tx["date"].dt.to_period("W-SUN").dt.start_time  # Monday start

    # keep only weeks that lie fully inside the data
    first_full = tx["week"].min() if tx["date"].min().normalize() <= tx["week"].min() \
        else tx["week"].min() + pd.Timedelta(weeks=1)
    last_week = tx["week"].max()
    last_full = last_week if tx["date"].max().normalize() >= last_week + pd.Timedelta(days=6) \
        else last_week - pd.Timedelta(weeks=1)
    weeks = pd.date_range(first_full, last_full, freq="W-MON")
    tx = tx[tx["week"].isin(weeks)]

    # Business calendar: trading days per week. The retailer shuts over
    # Christmas/New Year, which a planner knows in advance.
    days = tx.groupby("week")["date"].apply(lambda s: s.dt.normalize().nunique())
    cal = pd.DataFrame({"week": weeks})
    cal["trading_days"] = cal["week"].map(days).fillna(0).astype(int)
    cal["closed"] = cal["trading_days"] == 0
    cal["t"] = np.arange(len(cal))

    wk = (tx.groupby(["sku", "week"], as_index=False)
            .agg(units=("quantity", "sum"), revenue=("revenue", "sum"),
                 orders=("invoice", "nunique")))

    # ---- choose the series using the first year only ----
    sel_end = weeks[C.INITIAL_TRAIN_WEEKS - 1]
    first_year = wk[wk["week"] <= sel_end]
    stats = first_year.groupby("sku").agg(active_weeks=("week", "nunique"),
                                         revenue=("revenue", "sum"),
                                         units=("units", "sum"))
    eligible = stats[stats["active_weeks"] >= C.MIN_ACTIVE_WEEKS]
    chosen = eligible.sort_values("revenue", ascending=False).head(C.N_SERIES)
    log["skus_total"] = int(wk["sku"].nunique())
    log["skus_eligible_first_year"] = int(len(eligible))
    log["skus_selected"] = int(len(chosen))
    log["selected_share_of_first_year_revenue"] = round(
        float(chosen["revenue"].sum() / stats["revenue"].sum()), 4)

    # ABC class and intermittency, again from the first year only
    chosen = chosen.copy()
    rank = np.arange(len(chosen)) / len(chosen)
    chosen["abc"] = np.where(rank < 0.2, "A", np.where(rank < 0.5, "B", "C"))
    n_open = int((~cal.loc[cal["week"] <= sel_end, "closed"]).sum())
    chosen["zero_share_y1"] = 1 - chosen["active_weeks"] / n_open
    chosen["demand_type"] = np.where(chosen["zero_share_y1"] > 0.2, "intermittent", "regular")

    # ---- full grid from each product's first sale ----
    wk = wk[wk["sku"].isin(chosen.index)]
    first_sale = wk.groupby("sku")["week"].min()
    grid = (pd.MultiIndex.from_product([chosen.index, weeks], names=["sku", "week"])
              .to_frame(index=False))
    grid = grid[grid["week"] >= grid["sku"].map(first_sale)]
    panel = grid.merge(wk, on=["sku", "week"], how="left")
    panel[["units", "revenue", "orders"]] = panel[["units", "revenue", "orders"]].fillna(0)
    panel["price"] = (panel["revenue"] / panel["units"]).where(panel["units"] > 0)
    panel["price"] = panel.groupby("sku")["price"].ffill()
    panel = panel.merge(cal, on="week").merge(
        chosen[["abc", "demand_type"]].reset_index(), on="sku")
    desc = (tx[tx["sku"].isin(chosen.index)].groupby("sku")["description"]
              .agg(lambda s: s.dropna().mode().iat[0] if s.notna().any() else ""))
    panel["description"] = panel["sku"].map(desc).astype("string")
    panel = panel.sort_values(["sku", "week"]).reset_index(drop=True)

    log["weeks"] = int(len(weeks))
    log["week_first"] = str(weeks[0].date())
    log["week_last"] = str(weeks[-1].date())
    log["closed_weeks"] = [str(w.date()) for w in cal.loc[cal["closed"], "week"]]
    log["panel_rows"] = int(len(panel))
    log["panel_zero_share"] = round(float((panel["units"] == 0).mean()), 4)
    return panel, cal


def main() -> None:
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("Reading raw workbook (first run takes a few minutes)...")
    raw = load_raw()
    tx, log = clean(raw)
    tx.to_parquet(C.PROCESSED_DIR / "transactions.parquet", index=False)
    panel, cal = weekly_panel(tx, log)
    panel.to_parquet(C.PROCESSED_DIR / "panel.parquet", index=False)
    cal.to_parquet(C.PROCESSED_DIR / "calendar.parquet", index=False)
    (C.PROCESSED_DIR / "cleaning_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
