from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.date_window import get_current_date, in_window, price_data_end_date


@pytest.mark.unit
def test_market_timezone_drives_current_date(monkeypatch):
    import tradingagents.dataflows.date_window as date_window

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            instant = datetime(2026, 10, 2, 0, 30, tzinfo=UTC)
            return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)

    monkeypatch.setattr(date_window, "datetime", FrozenDateTime)
    with run_config({"market_timezone": "America/New_York"}):
        assert get_current_date() == "2026-10-01"
    with run_config({"market_timezone": "Asia/Shanghai"}):
        assert get_current_date() == "2026-10-02"


@pytest.mark.unit
def test_price_end_date_and_news_cutoff_are_point_in_time():
    with run_config({"price_data_end_date": "2026-10-01"}):
        assert price_data_end_date("2026-10-02") == "2026-10-01"
        assert price_data_end_date("2026-09-30") == "2026-09-30"
        assert price_data_end_date(None) is None
    start = datetime(2026, 10, 2, 0, tzinfo=UTC)
    end = datetime(2026, 10, 2, 0, tzinfo=UTC)
    before = datetime(2026, 10, 2, 12, 30, tzinfo=UTC)
    at_cutoff = datetime(2026, 10, 2, 12, 31, tzinfo=UTC)
    with run_config({"news_cutoff_utc": "2026-10-02T12:31:00Z"}):
        assert in_window(before, start, end)
        assert not in_window(at_cutoff, start, end)
        assert not in_window(None, start, end)


@pytest.mark.unit
def test_dated_price_tools_pass_the_configured_cap(monkeypatch):
    import tradingagents.agents.tools as tools

    calls = []
    monkeypatch.setattr(tools, "route_to_vendor", lambda *args: calls.append(args) or "ok")
    with run_config({"price_data_end_date": "2026-10-01"}):
        assert tools.get_stock_data.func("AAPL", "2026-09-01", "2026-10-02", "2026-10-02") == "ok"
        assert tools.get_indicators.func("AAPL", "rsi", "2026-10-02", 30, "2026-10-02") == "ok"
    assert calls[0] == ("get_stock_data", "AAPL", "2026-09-01", "2026-10-01")
    assert calls[1] == ("get_indicators", "AAPL", "rsi", "2026-10-01", 30)


@pytest.mark.unit
def test_verified_snapshot_receives_capped_date(monkeypatch):
    import tradingagents.agents.tools as tools

    captured = []
    monkeypatch.setattr(
        tools, "build_verified_market_snapshot",
        lambda symbol, date, lookback: captured.append((symbol, date, lookback)) or "snapshot",
    )
    with run_config({"price_data_end_date": "2026-10-01"}):
        result = tools.get_verified_market_snapshot.func(
            "AAPL", "2026-10-02", 10, "2026-10-02"
        )
    assert result == "snapshot"
    assert captured == [("AAPL", "2026-10-01", 10)]
