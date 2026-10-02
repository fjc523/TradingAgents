"""复权日线来源注册、缓存及按配置顺序降级。"""
from contextvars import ContextVar
from contextlib import contextmanager
import logging
from pathlib import Path
import pandas as pd
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import VendorUnavailableError, NoMarketDataError
from tradingagents.dataflows.files import replace_file
from tradingagents.dataflows.symbols import normalize_symbol, safe_ticker_component

logger = logging.getLogger(__name__)
_LOADERS = {}
_OVERRIDE = ContextVar("ohlcv_source", default=None)
LAST_SOURCE = ContextVar("ohlcv_last_source", default="未注明")


def register_ohlcv_source(name, loader):
    _LOADERS[name] = loader


@contextmanager
def using_source(name):
    token = _OVERRIDE.set(name)
    try:
        yield
    finally:
        _OVERRIDE.reset(token)


def load_ohlcv(symbol, as_of_date, fill_gaps=True):
    from tradingagents.dataflows.vendors.yahoo.ohlcv import (
        _load_yahoo_ohlcv, _clean_dataframe, _fill_price_gaps,
        _cache_is_fresh, _assert_ohlcv_not_stale,
    )
    from tradingagents.dataflows.vendors.alpaca.ohlcv import load_alpaca_ohlcv
    loaders = {"yfinance": None, "alpaca": load_alpaca_ohlcv, **_LOADERS}
    config = get_config()
    chain = _OVERRIDE.get() or config.get("data_vendors", {}).get("core_stock_apis", "yfinance")
    if chain == "default":
        chain = "alpaca,yfinance"
    cutoff = pd.Timestamp(as_of_date).normalize()
    now = pd.Timestamp.today()
    canonical = normalize_symbol(symbol)
    failures = []
    for name in dict.fromkeys(x.strip() for x in chain.split(",") if x.strip()):
        if name not in loaders:
            logger.warning("未注册日线来源%s，跳过", name)
            continue
        try:
            if name == "yfinance":
                data = _load_yahoo_ohlcv(symbol, as_of_date, fill_gaps)
            else:
                cache = Path(config["data_cache_dir"]) / f"{safe_ticker_component(canonical)}-{name}-data.csv"
                cache.parent.mkdir(parents=True, exist_ok=True)
                data = pd.read_csv(cache) if cache.exists() and _cache_is_fresh(cache, cutoff, now) else None
                if data is not None and not data.empty and "Date" in data:
                    latest = pd.to_datetime(data["Date"]).max().tz_localize(None).normalize()
                    if latest < cutoff and (now - pd.Timestamp.fromtimestamp(cache.stat().st_mtime)).total_seconds() > 900:
                        data = None
                if data is None or data.empty or "Close" not in data:
                    data = loaders[name](canonical, (now-pd.DateOffset(years=5)).strftime("%Y-%m-%d"), as_of_date)
                    if data is None or data.empty:
                        raise NoMarketDataError(symbol, canonical, "没有日线")
                    replace_file(cache, lambda temp: data.to_csv(temp, index=False))
                data = _clean_dataframe(data)
                data = data[data.Date <= cutoff].sort_values("Date")
                data = _fill_price_gaps(data) if fill_gaps else data.dropna(subset=["Close"]).copy()
                if data.empty:
                    raise NoMarketDataError(symbol, canonical, "截止日前没有日线")
                _assert_ohlcv_not_stale(data, as_of_date, symbol, canonical)
            data.attrs["source"] = name
            LAST_SOURCE.set(name)
            return data
        except Exception as exc:
            failures.append(exc)
            logger.warning("日线来源%s不可用：%s", name, type(exc).__name__)
    if failures and all(isinstance(exc, NoMarketDataError) for exc in failures):
        raise failures[-1]
    if failures and isinstance(failures[-1], VendorUnavailableError):
        raise failures[-1]
    raise VendorUnavailableError("DATA_UNAVAILABLE：配置的日线来源均不可用")


def stock_data_for_source(name, symbol, start_date, end_date):
    with using_source(name):
        data = load_ohlcv(symbol, end_date, fill_gaps=False)
        data = data[data.Date >= pd.Timestamp(start_date)]
        return f"# 日线来源：{name}；截至{end_date}\n" + data.to_csv(index=False)


def indicators_for_source(name, *args, **kwargs):
    from tradingagents.dataflows.vendors.yahoo.market import get_stock_stats_indicators_window
    with using_source(name):
        return get_stock_stats_indicators_window(*args, **kwargs)
