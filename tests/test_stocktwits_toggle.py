"""StockTwits 预取开关：关闭时不请求，并以“未配置”上报。"""
import pytest
from langchain_core.messages import AIMessage

from tradingagents.agents.analysts import sentiment_analyst
from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.vendor_observer import reset_vendor_observer, set_vendor_observer


class _LLM:
    def __init__(self):
        self.messages = None

    def with_structured_output(self, *a, **k):
        raise NotImplementedError

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content="report")


def _run(monkeypatch, enabled):
    calls, events = [], []
    monkeypatch.setattr(sentiment_analyst, "jev_screen", lambda ticker: None)
    monkeypatch.setattr(sentiment_analyst.get_news, "func", lambda *a: "news")
    monkeypatch.setattr(sentiment_analyst, "fetch_stocktwits_messages",
                        lambda *a, **k: calls.append("stocktwits") or "stocktwits posts")
    monkeypatch.setattr(sentiment_analyst, "fetch_reddit_posts", lambda *a, **k: "reddit posts")
    llm = _LLM()
    token = set_vendor_observer(events.append)
    try:
        with run_config({"stocktwits_enabled": enabled, "sentiment_min_social_posts": 0}):
            sentiment_analyst.create_sentiment_analyst(llm)(
                {"company_of_interest": "TSLA", "trade_date": "2026-10-02", "messages": []})
    finally:
        reset_vendor_observer(token)
    prompt = "\n".join(str(message.content) for message in llm.messages)
    return calls, [e for e in events if e["method"] == "fetch_stocktwits"], prompt


@pytest.mark.unit
def test_disabled_stocktwits_is_not_requested_and_reported_unconfigured(monkeypatch):
    calls, events, prompt = _run(monkeypatch, False)
    assert calls == []
    assert [e["outcome"] for e in events] == ["unconfigured"]
    assert "Cloudflare" in events[0]["error"]
    assert "not an absence of discussion" in prompt and "reddit posts" in prompt


@pytest.mark.unit
def test_enabled_stocktwits_keeps_fetching(monkeypatch):
    calls, events, prompt = _run(monkeypatch, True)
    assert calls == ["stocktwits"]
    assert [e["outcome"] for e in events] == ["success"]
    assert "stocktwits posts" in prompt
