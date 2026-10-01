from __future__ import annotations

from datetime import UTC, datetime
import json

import pytest

from tradingagents.dataflows.errors import VendorNotConfiguredError
import tradingagents.dataflows.vendors.alpaca.client as alpaca
import tradingagents.dataflows.vendors.alpaca.news as news


class FakeResponse:
    def __init__(self, body, status=200, headers=None):
        self._body = body
        self.status_code = status
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, responses, clock=None):
        self.responses = list(responses)
        self.calls = []
        self.clock = clock

    def get(self, url, *, params, headers, timeout):
        self.calls.append((url, dict(params), dict(headers), timeout))
        if self.clock is not None:
            self.clock.request_times.append(self.clock.now)
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "test-key-id")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret")


@pytest.mark.unit
def test_news_paginates_filters_created_at_marks_revision_and_truncation():
    responses = [
        FakeResponse({"news": [
            {"created_at": "2026-10-02T12:00:00Z", "updated_at": "2026-10-02T12:05:00Z", "headline": "in"},
            {"created_at": "2026-10-02T13:00:00Z", "updated_at": "2026-10-02T13:00:00Z", "headline": "outside"},
        ], "next_page_token": "page-2"}),
        FakeResponse({"news": [
            {"created_at": "2026-10-02T11:00:00Z", "updated_at": "2026-10-02T12:31:00Z", "headline": "revised"},
        ], "next_page_token": "page-3"}),
    ]
    session = FakeSession(responses)
    client = alpaca.AlpacaClient(session=session, sleeper=lambda _: None)
    start = datetime(2026, 10, 2, 10, tzinfo=UTC)
    end = datetime(2026, 10, 2, 12, 30, tzinfo=UTC)
    result = client.get_news(["AAPL"], start, end, max_pages=2)
    assert [item["headline"] for item in result["articles"]] == ["in", "revised"]
    assert result["articles"][0]["revised_after_cutoff"] is False
    assert result["articles"][1]["revised_after_cutoff"] is True
    assert result["truncated"] is True
    assert session.calls[0][1]["limit"] == 50
    assert session.calls[1][1]["page_token"] == "page-2"
    assert session.calls[0][2]["APCA-API-KEY-ID"] == "test-key-id"


@pytest.mark.unit
def test_shared_client_bars_snapshots_and_missing_credentials(monkeypatch):
    responses = [
        FakeResponse({"bars": {"AAPL": [{"t": "2026-10-01T00:00:00Z", "c": 100}]}},
                     headers={"X-Ratelimit-Remaining": "177"}),
        FakeResponse({"AAPL": {"latestTrade": {"p": 101}}}),
    ]
    session = FakeSession(responses)
    client = alpaca.AlpacaClient(session=session, sleeper=lambda _: None)
    bars = client.get_bars(["AAPL"], "2026-10-01", "2026-10-01", feed="sip", adjustment="all")
    assert client.rate_limit_remaining == 177
    snapshot = client.get_snapshots(["AAPL"], feed="overnight")
    assert bars["AAPL"][0]["c"] == 100
    assert session.calls[0][1]["end"].startswith("2026-10-02T00:00:00")
    assert session.calls[0][1]["adjustment"] == "all"
    assert snapshot["AAPL"]["latestTrade"]["p"] == 101
    assert session.calls[1][1]["feed"] == "overnight"
    monkeypatch.delenv("APCA_API_KEY_ID")
    with pytest.raises(VendorNotConfiguredError):
        _ = client.headers


@pytest.mark.unit
def test_bars_refuse_boats_and_recent_sip():
    client = alpaca.AlpacaClient(session=FakeSession([]))
    with pytest.raises(ValueError, match="feed must be 'sip' or 'iex'"):
        client.get_bars(["AAPL"], "2026-10-01", "2026-10-01", feed="boats")
    now = datetime.now(UTC).isoformat()
    with pytest.raises(ValueError, match="最近 15 分钟"):
        client.get_bars(["AAPL"], now, now, feed="sip", timeframe="1Min")


@pytest.mark.unit
def test_http_429_waits_for_reset_then_retries():
    session = FakeSession([
        FakeResponse({}, status=429, headers={"X-Ratelimit-Reset": "1005"}),
        FakeResponse({"news": []}),
    ])
    waits = []
    client = alpaca.AlpacaClient(session=session, sleeper=waits.append, clock=lambda: 1000)
    result = client.get_news(None, datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC))
    assert result == {"articles": [], "truncated": False}
    assert waits == [5]


@pytest.mark.unit
def test_shared_limiter_never_exceeds_180_in_any_rolling_minute(monkeypatch):
    class FakeClock:
        now = 0.0
        request_times = []

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    clock = FakeClock()
    session = FakeSession([FakeResponse({"news": []}) for _ in range(200)], clock=clock)
    client = alpaca.AlpacaClient(
        requests_per_minute=180, session=session, monotonic=clock.monotonic, sleeper=clock.sleep,
    )
    monkeypatch.setattr(alpaca, "_SINGLETON", client)
    for index in range(200):
        if index % 2:
            # 通过上游新闻数据源的共享客户端入口。
            news.get_news("AAPL", "2026-10-02", "2026-10-02")
        else:
            # 通过主项目直接调用共享客户端的入口。
            alpaca.get_shared_client().get_snapshots(["SPY"], feed="overnight")
    assert len(clock.request_times) == 200
    assert max(sum(start <= sent < start + 60 for sent in clock.request_times)
               for start in clock.request_times) <= 180
    assert clock.request_times[180] - clock.request_times[0] >= 60


@pytest.mark.unit
def test_news_adapter_uses_shared_client_and_cutoff(monkeypatch):
    captured = {}

    class Client:
        def get_news(self, symbols, start, end, *, limit=None, max_pages=20):
            captured.update(symbols=symbols, start=start, end=end, limit=limit)
            return {"articles": [
                {"created_at": "2026-10-02T12:30:00Z", "updated_at": "2026-10-02T12:32:00Z",
                 "headline": "Macro", "source": "Benzinga", "revised_after_cutoff": True}
            ], "truncated": True}

    monkeypatch.setattr(news, "get_shared_client", lambda: Client())
    from tradingagents.dataflows.config import run_config
    with run_config({"news_cutoff_utc": "2026-10-02T12:31:00Z", "news_article_limit": 7}):
        text = news.get_news("AAPL", "2026-10-02", "2026-10-02")
    assert captured["symbols"] == ["AAPL"] and captured["limit"] == 7
    assert captured["end"] == datetime(2026, 10, 2, 12, 31, tzinfo=UTC)
    assert "Macro" in text and "内容可能已在截止后修订" in text and "结果已截断" in text


@pytest.mark.unit
def test_router_falls_back_when_alpaca_credentials_are_missing(monkeypatch):
    import tradingagents.dataflows.router as router
    from tradingagents.dataflows.config import run_config

    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.setitem(router.VENDOR_METHODS["get_news"], "yfinance", lambda *args: "Yahoo fallback")
    with run_config({"data_vendors": {"news_data": "alpaca,yfinance"}}):
        result = router.route_to_vendor("get_news", "AAPL", "2026-10-01", "2026-10-02")
    assert result == "Yahoo fallback"
