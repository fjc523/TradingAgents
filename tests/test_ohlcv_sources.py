"""日线来源顺序、隔离缓存和截止日期的契约。"""
import pandas as pd
import pytest
from tradingagents.dataflows import ohlcv_sources as sources
from tradingagents.dataflows.errors import VendorUnavailableError


def frame():
    return pd.DataFrame({"Date": ["2026-10-01", "2026-10-02"], "Open": [10., 20.], "High": [11., 21.],
                         "Low": [9., 19.], "Close": [10., 20.], "Volume": [100, 200]})


def test_chain_fallback_cutoff_and_cache_isolation(tmp_path, monkeypatch):
    config = {"data_cache_dir": str(tmp_path), "data_vendors": {"core_stock_apis": "unknown,bad,first,second"}}
    monkeypatch.setattr(sources, "get_config", lambda: config)
    monkeypatch.setattr(sources, "_LOADERS", {})
    calls = []
    def bad(*args):
        calls.append("bad")
        raise VendorUnavailableError("断线")
    def good(*args):
        calls.append(args)
        return frame()
    sources.register_ohlcv_source("bad", bad)
    sources.register_ohlcv_source("first", good)
    sources.register_ohlcv_source("second", good)
    data = sources.load_ohlcv("TSLA", "2026-10-01", False)
    assert data.attrs["source"] == "first" and len(data) == 1
    assert calls[1][2] == "2026-10-01"
    with sources.using_source("second"):
        assert sources.load_ohlcv("TSLA", "2026-10-01").attrs["source"] == "second"
    assert len(list(tmp_path.glob("*-data.csv"))) == 2
    assert not list(tmp_path.glob("*unknown*"))


def test_all_failed_sources_raise_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "get_config", lambda: {"data_cache_dir": str(tmp_path), "data_vendors": {"core_stock_apis": "absent"}})
    with pytest.raises(VendorUnavailableError, match="DATA_UNAVAILABLE"):
        sources.load_ohlcv("TSLA", "2026-10-01")


def test_fill_gaps_is_optional(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "get_config", lambda: {"data_cache_dir": str(tmp_path), "data_vendors": {"core_stock_apis": "sample"}})
    data = frame(); data.loc[0, "Open"] = float("nan")
    monkeypatch.setitem(sources._LOADERS, "sample", lambda *args: data)
    assert pd.isna(sources.load_ohlcv("TSLA", "2026-10-02", False).Open.iloc[0])
    assert sources.load_ohlcv("TSLA", "2026-10-02", True).Open.iloc[0] == 20
