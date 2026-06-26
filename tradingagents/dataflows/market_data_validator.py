"""Deterministic market-data verification snapshot.

The market analyst is an LLM that can confabulate exact numbers — citing a
Bollinger band or a "historically validated bounce" that the underlying data
doesn't support (#830). This module computes a ground-truth snapshot (latest
OHLCV row on or before the analysis date, common indicators, recent closes)
the analyst is told to treat as the source of truth for any exact numeric
claim. Deterministic, no LLM involved.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from io import StringIO

import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.alpha_vantage_stock import get_stock as get_alpha_vantage_stock
from tradingagents.dataflows.symbol_utils import NoMarketDataError, normalize_symbol
from tradingagents.dataflows.stockstats_utils import load_ohlcv

# A fixed, common indicator set so the snapshot is the same shape every run.
DEFAULT_SNAPSHOT_INDICATORS: tuple[str, ...] = (
    "close_10_ema", "close_50_sma", "close_200_sma",
    "rsi", "boll", "boll_ub", "boll_lb",
    "macd", "macds", "macdh", "atr",
)


def _verified_rows_from_frame(symbol: str, curr_date: str, data: pd.DataFrame) -> pd.DataFrame:
    """从给定 OHLCV 表中取 curr_date 当日或之前的可验证行。"""
    if data is None or data.empty:
        raise NoMarketDataError(symbol, normalize_symbol(symbol), f"no OHLCV rows on or before {curr_date}")

    df = _normalize_ohlcv_frame(data)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["Date"] <= pd.to_datetime(curr_date)].sort_values("Date")
    if df.empty:
        raise NoMarketDataError(symbol, normalize_symbol(symbol), f"no OHLCV rows on or before {curr_date}")
    return df


def _verified_rows(symbol: str, curr_date: str) -> pd.DataFrame:
    """使用 yfinance OHLCV 缓存生成可验证行。"""
    return _verified_rows_from_frame(symbol, curr_date, load_ohlcv(symbol, curr_date))


def _normalize_ohlcv_frame(data: pd.DataFrame) -> pd.DataFrame:
    columns = {str(col).lower().strip(): col for col in data.columns}
    date_col = columns.get("date") or columns.get("timestamp") or data.columns[0]
    field_map = {
        "Date": date_col,
        "Open": columns.get("open"),
        "High": columns.get("high"),
        "Low": columns.get("low"),
        "Close": columns.get("close"),
        "Volume": columns.get("volume"),
    }
    missing = [target for target, source in field_map.items() if source is None]
    if missing:
        raise ValueError(f"OHLCV 数据缺少字段：{', '.join(missing)}")
    normalized = data[[source for source in field_map.values()]].copy()
    normalized.columns = list(field_map.keys())
    for col in ("Open", "High", "Low", "Close", "Volume"):
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    return normalized.dropna(subset=["Close"])


def _fmt(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_verified_market_snapshot(
    symbol: str,
    curr_date: str,
    look_back_days: int = 30,
    indicators: Iterable[str] | None = None,
) -> str:
    """Render a ground-truth snapshot: latest OHLCV row, indicators, recent closes."""
    return build_verified_market_snapshot_from_ohlcv(
        symbol,
        curr_date,
        _verified_rows(symbol, curr_date),
        look_back_days,
        indicators,
    )


def build_verified_market_snapshot_alpha_vantage(
    symbol: str,
    curr_date: str,
    look_back_days: int = 30,
    indicators: Iterable[str] | None = None,
) -> str:
    """使用 Alpha Vantage 日线数据生成验证快照，供 vendor router 备用。"""
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date = (pd.Timestamp(curr_date_dt) - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    csv_text = get_alpha_vantage_stock(symbol, start_date, curr_date)
    data = pd.read_csv(StringIO(csv_text))
    return build_verified_market_snapshot_from_ohlcv(
        symbol,
        curr_date,
        _verified_rows_from_frame(symbol, curr_date, data),
        look_back_days,
        indicators,
    )


def build_verified_market_snapshot_from_ohlcv(
    symbol: str,
    curr_date: str,
    df: pd.DataFrame,
    look_back_days: int = 30,
    indicators: Iterable[str] | None = None,
) -> str:
    """基于已经解析好的 OHLCV 表渲染确定性验证快照。"""
    # `df` keeps the original capitalized OHLCV columns (Open/High/Low/Close/
    # Volume); stockstats `wrap()` lowercases columns and adds indicator
    # columns, so read raw prices from `df` and indicators from `stock_df`.
    stock_df = wrap(df.copy())

    selected = tuple(indicators or DEFAULT_SNAPSHOT_INDICATORS)
    indicator_values: dict[str, str] = {}
    for name in selected:
        try:
            stock_df[name]  # triggers stockstats calculation
            indicator_values[name] = _fmt(stock_df.iloc[-1][name])
        except Exception as exc:  # noqa: BLE001 — one bad indicator shouldn't sink the snapshot
            indicator_values[name] = f"N/A ({type(exc).__name__})"

    latest = df.iloc[-1]
    latest_date = _fmt(latest["Date"])
    window = max(1, min(int(look_back_days), 30))
    recent = df.tail(window)

    lines = [
        f"## Verified market data snapshot for {symbol.upper()}",
        "",
        f"- Requested analysis date: {curr_date}",
        f"- Latest trading row used: {latest_date}",
        "- Rows after the requested analysis date are excluded before verification.",
        "",
        "### Latest verified OHLCV row",
        "",
        "| Field | Value |",
        "|---|---:|",
    ]
    for field in ("Open", "High", "Low", "Close", "Volume"):
        lines.append(f"| {field} | {_fmt(latest.get(field))} |")

    lines += ["", "### Verified technical indicators (latest row)", "",
              "| Indicator | Value |", "|---|---:|"]
    for name, value in indicator_values.items():
        lines.append(f"| {name} | {value} |")

    lines += ["", f"### Recent verified closes (last {len(recent)} rows)", "",
              "| Date | Close |", "|---|---:|"]
    for _, row in recent.iterrows():
        lines.append(f"| {_fmt(row['Date'])} | {_fmt(row.get('Close'))} |")

    lines += [
        "",
        "Use this snapshot as the source of truth for exact OHLCV, price-level, "
        "and indicator-value claims. If another tool output conflicts with it, "
        "flag the discrepancy rather than inventing a reconciled number. Do not "
        "claim historical validation, support/resistance bounces, or exact "
        "percentage moves unless directly supported by tool output with concrete "
        "dates and prices.",
    ]
    return "\n".join(lines)
