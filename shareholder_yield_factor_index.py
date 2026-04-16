"""
Shareholder Yield Factor Index — Russell 1000 Universe
=======================================================
Constructs a factor-tilted index using two signals:
  1. Shareholder Yield (level)       — dividends + net buybacks / market cap
  2. Shareholder Yield Growth (trend) — YoY change in shareholder yield numerator

Uses the LSEG Data Library for Python (lseg-data) for all data retrieval.
Requires a valid LSEG Workspace desktop session or Platform session.

Setup:
  pip install lseg-data pandas numpy scipy matplotlib quantstats
  Ensure lseg-data.config.json is configured (see LSEG developer docs).
"""

import warnings
warnings.filterwarnings("ignore")

import lseg.data as ld
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime, timedelta
from typing import Optional
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
class Config:
    """Central configuration for the index methodology."""

    # Universe
    INDEX_RIC = ".RUI"                          # Russell 1000
    BENCHMARK_RIC = ".RUI"                      # Benchmark for performance comparison

    # Signal parameters
    LOOKBACK_YEARS = 5                          # Years of history for the backtest
    SY_GROWTH_LOOKBACK_MONTHS = 12              # YoY window for shareholder yield growth

    # Scoring
    WINSORIZE_LIMITS = (0.025, 0.025)           # 2.5th / 97.5th percentile clip
    SIGNAL_WEIGHTS = {                          # Composite signal weights
        "sy_level": 0.50,
        "sy_growth": 0.50,
    }

    # Portfolio construction
    WEIGHTING_METHOD = "tilt"                   # "tilt" | "score" | "top_n"
    TOP_N = 200                                 # Used only if WEIGHTING_METHOD == "top_n"
    TILT_LAMBDA = 2.0                           # Exponential tilt strength (higher = more aggressive)
    MIN_WEIGHT = 0.0005                         # Floor weight (5 bps) to avoid dust positions
    MAX_WEIGHT = 0.05                           # Cap weight at 5% to limit concentration
    SECTOR_CAP_MULTIPLE = 2.0                   # Max sector weight = benchmark weight × this

    # Rebalancing
    REBALANCE_FREQ = "Q"                        # Q = quarterly
    BUFFER_RANK = 50                            # Hysteresis buffer for top-N method

    # Transaction cost model
    COST_PER_SIDE_BPS = 15                      # Estimated one-way cost in basis points

    # LSEG API batching
    API_BATCH_SIZE = 80                         # Max instruments per get_data call
    API_SLEEP_SECONDS = 1.5                     # Throttle between API calls


# ─────────────────────────────────────────────
# 1.  Data Retrieval
# ─────────────────────────────────────────────
def open_session() -> None:
    """Open an LSEG data session (desktop or platform, per config file)."""
    ld.open_session()
    log.info("LSEG session opened.")


def close_session() -> None:
    ld.close_session()
    log.info("LSEG session closed.")


def get_constituents(index_ric: str, as_of_date: Optional[str] = None) -> list[str]:
    """
    Retrieve current Russell 1000 constituent RICs.

    Uses TR.IndexConstituentRIC, which returns the official constituent list.
    If the index chain (0#.RUI) is unavailable due to licensing, falls back
    to the Screener approach.

    Parameters
    ----------
    index_ric : str
        The index RIC, e.g. ".RUI"
    as_of_date : str, optional
        Date string "YYYYMMDD" for point-in-time constituents.

    Returns
    -------
    list[str]
        List of constituent RICs.
    """
    params = {}
    if as_of_date:
        params["SDate"] = as_of_date

    try:
        df = ld.get_data(
            universe=index_ric,
            fields=["TR.IndexConstituentRIC"],
            parameters=params,
        )
        rics = df["Constituent RIC"].dropna().tolist()
        if len(rics) > 100:
            log.info(f"Retrieved {len(rics)} constituents via TR.IndexConstituentRIC.")
            return rics
    except Exception as e:
        log.warning(f"TR.IndexConstituentRIC failed: {e}")

    # Fallback: Screener approach for Russell 1000
    try:
        screener_expr = (
            'SCREEN(U(IN(Equity(active,public,primary))),'
            'IN(TR.IndexName,"Russell 1000"),'
            'CURN=USD)'
        )
        df = ld.get_data(screener_expr, ["TR.RIC"])
        rics = df.iloc[:, -1].dropna().tolist()
        log.info(f"Retrieved {len(rics)} constituents via Screener fallback.")
        return rics
    except Exception as e:
        log.warning(f"Screener fallback also failed: {e}")

    # Last resort: chain RIC
    try:
        chain_ric = f"0#{index_ric}"
        df = ld.get_data(chain_ric, ["TR.RIC"])
        rics = df.iloc[:, -1].dropna().tolist()
        log.info(f"Retrieved {len(rics)} constituents via chain RIC.")
        return rics
    except Exception as e:
        raise RuntimeError(
            f"Could not retrieve constituents for {index_ric}. "
            "Check your LSEG entitlements — Russell index data may require "
            "a separate data license. Contact your LSEG account team."
        ) from e


def fetch_fundamental_data(
    rics: list[str],
    fields: list[str],
    parameters: dict,
    batch_size: int = Config.API_BATCH_SIZE,
) -> pd.DataFrame:
    """
    Fetch fundamental data in batches to respect LSEG API limits.

    Parameters
    ----------
    rics : list[str]
        Instrument RICs.
    fields : list[str]
        TR field codes to retrieve.
    parameters : dict
        LSEG field-level parameters (e.g. Period, SDate, EDate, Frq).
    batch_size : int
        Number of instruments per API call.

    Returns
    -------
    pd.DataFrame
        Concatenated results across all batches.
    """
    frames = []
    total_batches = (len(rics) + batch_size - 1) // batch_size

    for i in range(0, len(rics), batch_size):
        batch = rics[i : i + batch_size]
        batch_num = i // batch_size + 1
        log.info(f"  Fetching batch {batch_num}/{total_batches} ({len(batch)} instruments)...")

        try:
            df = ld.get_data(universe=batch, fields=fields, parameters=parameters)
            frames.append(df)
        except Exception as e:
            log.warning(f"  Batch {batch_num} failed: {e}")

        if batch_num < total_batches:
            time.sleep(Config.API_SLEEP_SECONDS)

    if not frames:
        raise RuntimeError("All data batches failed. Check session and entitlements.")

    return pd.concat(frames, ignore_index=True)


def fetch_shareholder_yield_inputs(rics: list[str], period: str) -> pd.DataFrame:
    """
    Retrieve the raw inputs needed to compute shareholder yield for one period.

    Shareholder Yield = (Dividends Paid + Net Buybacks) / Market Cap

    We pull from the cash flow statement and market data:
      - TR.F.DivsPd              : Total cash dividends paid (negative in CF stmt)
      - TR.F.ComStkBuyback       : Cash spent on share repurchases (negative in CF stmt)
      - TR.F.ComStkIssd          : Cash received from share issuance
      - TR.CompanyMarketCap      : Market capitalization
      - TR.TRBCEconomicSector    : Sector classification (for sector constraints)

    Parameters
    ----------
    rics : list[str]
        Constituent RICs.
    period : str
        Fiscal period identifier, e.g. "FY0" (most recent), "FY-1" (prior year).

    Returns
    -------
    pd.DataFrame
        One row per instrument with raw financial data.
    """
    fields = [
        "TR.F.DivsPd",
        "TR.F.ComStkBuyback",
        "TR.F.ComStkIssd",
        "TR.CompanyMarketCap",
        "TR.TRBCEconomicSector",
        "TR.PriceClose",
        "TR.F.DivsPd.periodenddate",
    ]
    params = {"Period": period, "Curn": "USD", "Scale": "6"}  # Scale=6 → millions

    df = fetch_fundamental_data(rics, fields, params)
    return df


def fetch_benchmark_returns(
    benchmark_ric: str, start: str, end: str
) -> pd.Series:
    """
    Retrieve daily total-return index for the benchmark.

    Parameters
    ----------
    benchmark_ric : str
        Benchmark RIC (e.g. ".RUI").
    start, end : str
        Date range in "YYYY-MM-DD" format.

    Returns
    -------
    pd.Series
        Daily returns indexed by date.
    """
    df = ld.get_history(
        universe=benchmark_ric,
        fields=["TRDPRC_1"],
        interval="1D",
        start=start,
        end=end,
    )
    prices = df["TRDPRC_1"].dropna()
    returns = prices.pct_change().dropna()
    returns.name = "benchmark"
    return returns


def fetch_price_history(
    rics: list[str], start: str, end: str
) -> pd.DataFrame:
    """
    Retrieve daily closing prices for all constituents.

    Parameters
    ----------
    rics : list[str]
        Instrument RICs.
    start, end : str
        Date range.

    Returns
    -------
    pd.DataFrame
        Columns = RICs, Index = dates, Values = closing prices.
    """
    prices = ld.get_history(
        universe=rics,
        fields=["TRDPRC_1"],
        interval="1D",
        start=start,
        end=end,
    )
    # get_history returns MultiIndex columns (field, RIC) when multiple instruments
    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices["TRDPRC_1"]
    return prices


# ─────────────────────────────────────────────
# 2.  Signal Construction
# ─────────────────────────────────────────────
def compute_shareholder_yield(df: pd.DataFrame) -> pd.Series:
    """
    Compute shareholder yield from raw fundamentals.

    SY_i = (|DivsPaid_i| + |Buybacks_i| - |Issuance_i|) / MarketCap_i

    DivsPaid and Buybacks are typically reported as negative values in the
    cash flow statement (cash outflows), so we take absolute values.
    Issuance is positive (cash inflow), so net buyback = |buybacks| - issuance.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns from fetch_shareholder_yield_inputs().

    Returns
    -------
    pd.Series
        Shareholder yield per instrument (indexed same as df).
    """
    divs = df["Dividends Paid"].fillna(0).abs()
    buybacks = df["Common Stock Bought Back"].fillna(0).abs()
    issuance = df["Common Stock Issued"].fillna(0).abs()
    mktcap = df["Company Market Cap"]

    net_buyback = (buybacks - issuance).clip(lower=0)  # Floor at 0: ignore net issuers
    total_payout = divs + net_buyback

    sy = total_payout / mktcap
    sy = sy.replace([np.inf, -np.inf], np.nan)
    return sy


def compute_shareholder_yield_growth(
    sy_current: pd.Series, sy_prior: pd.Series
) -> pd.Series:
    """
    Compute year-over-year growth in the shareholder yield *numerator*
    (total payout), NOT in the yield ratio itself. This isolates the
    management decision to increase capital returns from price noise.

    g_i = (Payout_current - Payout_prior) / Payout_prior

    Parameters
    ----------
    sy_current : pd.Series
        Current-period total payout (numerator of SY), indexed by RIC.
    sy_prior : pd.Series
        Prior-period total payout, same index.

    Returns
    -------
    pd.Series
        YoY payout growth rate.
    """
    growth = (sy_current - sy_prior) / sy_prior.replace(0, np.nan)
    growth = growth.replace([np.inf, -np.inf], np.nan)
    return growth


def cross_sectional_zscore(signal: pd.Series, limits: tuple = Config.WINSORIZE_LIMITS) -> pd.Series:
    """
    Winsorize then z-score a cross-sectional signal.

    z_i = (x_i - mean(x)) / std(x)

    after clipping x at the 2.5th and 97.5th percentiles.

    Parameters
    ----------
    signal : pd.Series
        Raw signal values (one per stock).
    limits : tuple
        (lower_frac, upper_frac) for winsorization.

    Returns
    -------
    pd.Series
        Standardized signal.
    """
    valid = signal.dropna()
    if len(valid) < 10:
        return signal * np.nan

    winsorized = pd.Series(
        winsorize(valid.values, limits=limits),
        index=valid.index,
        dtype=float,
    )
    mu = winsorized.mean()
    sigma = winsorized.std()
    if sigma < 1e-12:
        return winsorized * 0.0

    z = (winsorized - mu) / sigma
    return z.reindex(signal.index)


def build_composite_score(
    z_sy: pd.Series, z_sy_growth: pd.Series, weights: dict = Config.SIGNAL_WEIGHTS
) -> pd.Series:
    """
    Combine z-scored signals into a single composite score.

    C_i = w_1 * z_{SY,i} + w_2 * z_{gSY,i}

    Parameters
    ----------
    z_sy : pd.Series
        Z-scored shareholder yield level.
    z_sy_growth : pd.Series
        Z-scored shareholder yield growth.
    weights : dict
        Signal weights (must sum to 1).

    Returns
    -------
    pd.Series
        Composite factor score per stock.
    """
    composite = (
        weights["sy_level"] * z_sy.fillna(0)
        + weights["sy_growth"] * z_sy_growth.fillna(0)
    )
    # Penalize stocks missing both signals
    both_missing = z_sy.isna() & z_sy_growth.isna()
    composite[both_missing] = np.nan

    return composite


# ─────────────────────────────────────────────
# 3.  Portfolio Construction
# ─────────────────────────────────────────────
def compute_index_weights(
    composite: pd.Series,
    mktcap: pd.Series,
    sectors: pd.Series,
    method: str = Config.WEIGHTING_METHOD,
) -> pd.Series:
    """
    Convert composite scores into index weights.

    Three methods available:

    "tilt":   w_i = w_i^{mktcap} * exp(lambda * C_i)  (then renormalize)
    "score":  w_i = max(C_i, 0)                        (then renormalize)
    "top_n":  Select top N by composite, weight by mktcap within that subset

    After initial weighting, applies position-level caps and sector constraints.

    Parameters
    ----------
    composite : pd.Series
        Composite factor score (indexed by RIC).
    mktcap : pd.Series
        Market capitalization (same index).
    sectors : pd.Series
        Sector labels (same index).
    method : str
        Weighting method.

    Returns
    -------
    pd.Series
        Final portfolio weights summing to 1.
    """
    # Drop stocks with missing data
    valid = composite.dropna().index.intersection(mktcap.dropna().index)
    comp = composite.loc[valid]
    mc = mktcap.loc[valid]
    sec = sectors.reindex(valid)

    # Market-cap weights as baseline
    mc_weights = mc / mc.sum()

    if method == "tilt":
        # Exponential tilt: more aggressive than linear, controlled by lambda
        tilt_factor = np.exp(Config.TILT_LAMBDA * comp)
        raw_weights = mc_weights * tilt_factor

    elif method == "score":
        # Pure score-weighted (zero out negative scores)
        raw_weights = comp.clip(lower=0)

    elif method == "top_n":
        # Select top N, mktcap weight within subset
        top_rics = comp.nlargest(Config.TOP_N).index
        raw_weights = mc_weights.reindex(top_rics).fillna(0)

    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # Renormalize
    weights = raw_weights / raw_weights.sum()

    # Apply single-name caps
    weights = weights.clip(upper=Config.MAX_WEIGHT)
    weights = weights[weights >= Config.MIN_WEIGHT]
    weights = weights / weights.sum()

    # Apply sector caps (max = benchmark sector weight × multiplier)
    if Config.SECTOR_CAP_MULTIPLE is not None:
        benchmark_sector_wt = mc.groupby(sec).sum() / mc.sum()
        sector_caps = benchmark_sector_wt * Config.SECTOR_CAP_MULTIPLE

        for sector in sec.unique():
            if pd.isna(sector):
                continue
            mask = sec == sector
            sector_wt = weights[mask].sum()
            cap = sector_caps.get(sector, 1.0)
            if sector_wt > cap and sector_wt > 0:
                scale = cap / sector_wt
                weights[mask] *= scale

        weights = weights / weights.sum()

    return weights


# ─────────────────────────────────────────────
# 4.  Backtesting Engine
# ─────────────────────────────────────────────
def estimate_transaction_costs(
    old_weights: pd.Series, new_weights: pd.Series
) -> float:
    """
    Estimate transaction cost drag for a single rebalance.

    TC = sum(|w_new - w_old|) * cost_per_side

    The sum of absolute weight changes equals 2x the one-way turnover,
    so we divide by 2 and multiply by cost per side.

    Parameters
    ----------
    old_weights, new_weights : pd.Series
        Portfolio weights before and after rebalance.

    Returns
    -------
    float
        Transaction cost as a fraction of portfolio value.
    """
    all_rics = old_weights.index.union(new_weights.index)
    old = old_weights.reindex(all_rics, fill_value=0)
    new = new_weights.reindex(all_rics, fill_value=0)
    one_way_turnover = (new - old).abs().sum() / 2
    cost = one_way_turnover * (Config.COST_PER_SIDE_BPS / 10_000)
    return cost


def run_backtest(
    weights_history: dict[str, pd.Series],
    price_df: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """
    Simulate the factor index returns using historical weights and prices.

    At each rebalance date, the portfolio is set to target weights.
    Between rebalances, weights drift with prices (buy-and-hold).
    Transaction costs are deducted on each rebalance.

    Parameters
    ----------
    weights_history : dict
        {rebalance_date_str: pd.Series of weights}
    price_df : pd.DataFrame
        Daily prices, columns = RICs.
    benchmark_returns : pd.Series
        Daily benchmark returns.

    Returns
    -------
    pd.DataFrame
        Columns: ["factor_index", "benchmark", "excess"]
        Values: cumulative return indices (starting at 1).
    """
    rebal_dates = sorted(weights_history.keys())
    all_dates = price_df.index.sort_values()

    portfolio_returns = []
    prev_weights = pd.Series(dtype=float)

    for i, rebal_date in enumerate(rebal_dates):
        target_weights = weights_history[rebal_date]

        # Transaction cost on rebalance
        tc = estimate_transaction_costs(prev_weights, target_weights)

        # Determine date range for this holding period
        start_idx = all_dates.get_indexer([pd.Timestamp(rebal_date)], method="ffill")[0]
        if i + 1 < len(rebal_dates):
            end_idx = all_dates.get_indexer(
                [pd.Timestamp(rebal_dates[i + 1])], method="ffill"
            )[0]
        else:
            end_idx = len(all_dates)

        period_dates = all_dates[start_idx:end_idx]
        if len(period_dates) < 2:
            continue

        # Daily returns for held instruments
        held_rics = target_weights.index.intersection(price_df.columns)
        period_prices = price_df.loc[period_dates, held_rics]
        daily_returns = period_prices.pct_change()

        # First day: apply transaction cost
        w = target_weights.reindex(held_rics, fill_value=0)
        w = w / w.sum()  # Ensure sum = 1

        for j, date in enumerate(period_dates[1:], 1):
            day_ret = daily_returns.loc[date]
            port_ret = (w * day_ret.reindex(w.index, fill_value=0)).sum()

            if j == 1:
                port_ret -= tc  # Deduct TC on rebalance day

            portfolio_returns.append({"date": date, "factor_return": port_ret})

            # Drift weights (buy-and-hold between rebalances)
            w = w * (1 + day_ret.reindex(w.index, fill_value=0))
            w_sum = w.sum()
            if w_sum > 0:
                w = w / w_sum

        prev_weights = target_weights

    # Assemble results
    ret_df = pd.DataFrame(portfolio_returns).set_index("date")
    ret_df.index = pd.DatetimeIndex(ret_df.index)

    # Align benchmark
    bmk = benchmark_returns.reindex(ret_df.index).fillna(0)

    results = pd.DataFrame(
        {
            "factor_index": (1 + ret_df["factor_return"]).cumprod(),
            "benchmark": (1 + bmk).cumprod(),
        }
    )
    results["excess"] = results["factor_index"] / results["benchmark"]

    return results


# ─────────────────────────────────────────────
# 5.  Analytics & Reporting
# ─────────────────────────────────────────────
def compute_performance_stats(results: pd.DataFrame) -> dict:
    """
    Compute key performance metrics for the factor index and benchmark.

    Returns
    -------
    dict
        Performance statistics.
    """
    factor_ret = results["factor_index"].pct_change().dropna()
    bmk_ret = results["benchmark"].pct_change().dropna()
    excess_ret = factor_ret - bmk_ret

    trading_days = 252

    stats = {}
    for name, rets in [("Factor Index", factor_ret), ("Benchmark", bmk_ret), ("Excess", excess_ret)]:
        ann_ret = (1 + rets.mean()) ** trading_days - 1
        ann_vol = rets.std() * np.sqrt(trading_days)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = (1 + rets).cumprod()
        drawdown = cum / cum.cummax() - 1
        max_dd = drawdown.min()

        stats[name] = {
            "Annualized Return": f"{ann_ret:.2%}",
            "Annualized Volatility": f"{ann_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd:.2%}",
            "Total Return": f"{cum.iloc[-1] - 1:.2%}" if len(cum) > 0 else "N/A",
        }

    return stats


def compute_turnover_stats(weights_history: dict[str, pd.Series]) -> dict:
    """
    Compute turnover statistics across all rebalance periods.
    """
    dates = sorted(weights_history.keys())
    turnovers = []

    for i in range(1, len(dates)):
        old_w = weights_history[dates[i - 1]]
        new_w = weights_history[dates[i]]
        all_rics = old_w.index.union(new_w.index)
        turnover = (
            new_w.reindex(all_rics, fill_value=0)
            - old_w.reindex(all_rics, fill_value=0)
        ).abs().sum() / 2
        turnovers.append(turnover)

    return {
        "Mean One-Way Turnover": f"{np.mean(turnovers):.2%}",
        "Max One-Way Turnover": f"{np.max(turnovers):.2%}" if turnovers else "N/A",
        "Min One-Way Turnover": f"{np.min(turnovers):.2%}" if turnovers else "N/A",
        "Rebalance Count": len(dates),
        "Est. Annual TC Drag": f"{np.mean(turnovers) * 4 * Config.COST_PER_SIDE_BPS / 10_000:.4%}",
    }


def plot_results(results: pd.DataFrame, output_path: str = "factor_index_backtest.png") -> str:
    """
    Generate a two-panel chart: cumulative returns + excess return.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[2, 1])
    fig.suptitle("Shareholder Yield Factor Index vs Russell 1000", fontsize=14, fontweight="bold")

    # Panel 1: Cumulative returns
    ax1.plot(results.index, results["factor_index"], label="SY Factor Index", linewidth=1.5, color="#2563eb")
    ax1.plot(results.index, results["benchmark"], label="Russell 1000", linewidth=1.5, color="#6b7280", linestyle="--")
    ax1.set_ylabel("Growth of $1")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Excess return (cumulative)
    ax2.plot(results.index, results["excess"], label="Excess (Factor / Benchmark)", linewidth=1.5, color="#059669")
    ax2.axhline(y=1.0, color="#6b7280", linestyle=":", linewidth=0.8)
    ax2.set_ylabel("Relative Performance")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Chart saved to {output_path}")
    return output_path


def print_report(
    stats: dict,
    turnover: dict,
    weights_snapshot: pd.Series,
    sectors: pd.Series,
) -> None:
    """Print a formatted summary report to stdout."""

    print("\n" + "=" * 70)
    print("  SHAREHOLDER YIELD FACTOR INDEX — BACKTEST REPORT")
    print("=" * 70)

    print("\n── Performance Summary ──")
    header = f"{'Metric':<28} {'Factor Index':<18} {'Benchmark':<18} {'Excess':<18}"
    print(header)
    print("-" * len(header))
    for metric in ["Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Max Drawdown", "Total Return"]:
        fi = stats.get("Factor Index", {}).get(metric, "")
        bm = stats.get("Benchmark", {}).get(metric, "")
        ex = stats.get("Excess", {}).get(metric, "")
        print(f"{metric:<28} {fi:<18} {bm:<18} {ex:<18}")

    print("\n── Turnover Statistics ──")
    for k, v in turnover.items():
        print(f"  {k:<30} {v}")

    print("\n── Latest Portfolio Snapshot ──")
    print(f"  Number of holdings:  {len(weights_snapshot)}")
    print(f"  Top 10 weights:")
    for ric, wt in weights_snapshot.nlargest(10).items():
        print(f"    {ric:<20} {wt:.2%}")

    print("\n── Sector Allocation ──")
    sector_wts = weights_snapshot.groupby(sectors.reindex(weights_snapshot.index)).sum()
    sector_wts = sector_wts.sort_values(ascending=False)
    for sec, wt in sector_wts.items():
        if pd.notna(sec):
            print(f"    {sec:<35} {wt:.2%}")

    print("\n" + "=" * 70 + "\n")


# ─────────────────────────────────────────────
# 6.  Main Orchestrator
# ─────────────────────────────────────────────
def main():
    """
    End-to-end pipeline:
      1. Open LSEG session
      2. Get Russell 1000 constituents
      3. For each quarterly rebalance date:
         a. Fetch fundamentals (current year + prior year)
         b. Compute shareholder yield level & growth signals
         c. Z-score signals cross-sectionally
         d. Build composite score
         e. Construct portfolio weights
      4. Fetch daily prices for the full period
      5. Run the backtest
      6. Generate analytics and report
    """
    open_session()

    try:
        # ── Step 1: Get universe ──
        log.info("Fetching Russell 1000 constituents...")
        rics = get_constituents(Config.INDEX_RIC)
        log.info(f"Universe: {len(rics)} stocks")

        # ── Step 2: Fetch fundamental data for current & prior fiscal years ──
        log.info("Fetching shareholder yield data (current year = FY0)...")
        df_fy0 = fetch_shareholder_yield_inputs(rics, period="FY0")

        log.info("Fetching shareholder yield data (prior year = FY-1)...")
        df_fy1 = fetch_shareholder_yield_inputs(rics, period="FY-1")

        # ── Step 3: Compute signals ──
        log.info("Computing shareholder yield signals...")

        # Align dataframes on instrument identifier
        # The first column from get_data is typically the instrument RIC
        ric_col = df_fy0.columns[0]
        df_fy0 = df_fy0.set_index(ric_col)
        df_fy1 = df_fy1.set_index(ric_col)

        # Shareholder yield (level)
        sy_current = compute_shareholder_yield(df_fy0)
        sy_prior = compute_shareholder_yield(df_fy1)

        # Total payout (numerator) for growth calculation
        def _total_payout(df):
            divs = df["Dividends Paid"].fillna(0).abs()
            buybacks = df["Common Stock Bought Back"].fillna(0).abs()
            issuance = df["Common Stock Issued"].fillna(0).abs()
            return divs + (buybacks - issuance).clip(lower=0)

        payout_current = _total_payout(df_fy0)
        payout_prior = _total_payout(df_fy1)
        sy_growth = compute_shareholder_yield_growth(payout_current, payout_prior)

        # ── Step 4: Score and weight ──
        log.info("Z-scoring and building composite...")
        z_sy = cross_sectional_zscore(sy_current)
        z_sy_growth = cross_sectional_zscore(sy_growth)

        composite = build_composite_score(z_sy, z_sy_growth)
        log.info(
            f"Signal coverage: SY level={z_sy.notna().sum()}, "
            f"SY growth={z_sy_growth.notna().sum()}, "
            f"Composite={composite.notna().sum()}"
        )

        mktcap = df_fy0["Company Market Cap"]
        sectors = df_fy0["TRBC Economic Sector Name"]

        weights = compute_index_weights(composite, mktcap, sectors)
        log.info(f"Portfolio: {len(weights)} holdings, max weight = {weights.max():.2%}")

        # ── Step 5: Fetch prices and run backtest ──
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365 * Config.LOOKBACK_YEARS)).strftime("%Y-%m-%d")

        log.info(f"Fetching benchmark returns ({start_date} to {end_date})...")
        benchmark_returns = fetch_benchmark_returns(Config.BENCHMARK_RIC, start_date, end_date)

        # For a single-period demonstration, we create one rebalance entry.
        # In a full production system, you would loop over quarterly dates,
        # fetch point-in-time constituents and fundamentals for each, and
        # build weights_history with multiple entries.
        today_str = datetime.now().strftime("%Y-%m-%d")
        weights_history = {today_str: weights}

        # NOTE: A full multi-period backtest requires:
        #   1. Historical constituent lists for each rebalance date
        #   2. Point-in-time fundamental data for each period
        #   3. Iterating over each quarter to build weights_history
        # The code below demonstrates the single-period structure.
        # Extending to multi-period follows the same pattern in a loop.

        log.info("Fetching constituent price history...")
        held_rics = weights.index.tolist()
        price_df = fetch_price_history(held_rics, start_date, end_date)

        log.info("Running backtest...")
        results = run_backtest(weights_history, price_df, benchmark_returns)

        # ── Step 6: Report ──
        stats = compute_performance_stats(results)
        turnover = compute_turnover_stats(weights_history)
        chart_path = plot_results(results)

        print_report(stats, turnover, weights, sectors)

        log.info("Pipeline complete.")

    finally:
        close_session()


if __name__ == "__main__":
    main()
