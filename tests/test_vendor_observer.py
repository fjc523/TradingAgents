"""来源观察钩子必须保持调用结果与降级顺序。"""
import pytest
from tradingagents.dataflows.vendor_observer import observed_call, set_vendor_observer, reset_vendor_observer
from tradingagents.dataflows.errors import VendorUnavailableError


def test_observer_reports_outcomes_and_resets():
    events = []
    token = set_vendor_observer(events.append)
    try:
        assert observed_call("get_news", "alpaca", lambda: "新闻") == "新闻"
        def failed():
            raise VendorUnavailableError("失败")
        with pytest.raises(VendorUnavailableError):
            observed_call("get_news", "futu", failed)
        assert [row["outcome"] for row in events] == ["success", "failed"]
        assert events[-1]['error'] == 'VendorUnavailableError：失败'
        assert all(row["duration_seconds"] >= 0 for row in events)
    finally:
        reset_vendor_observer(token)
    observed_call("get_news", "alpaca", lambda: "新闻")
    assert len(events) == 2


def test_observer_exception_does_not_change_result():
    def bad_observer(event):
        raise ValueError("观察者错误")
    token = set_vendor_observer(bad_observer)
    try:
        assert observed_call("get_news", "alpaca", lambda: "新闻") == "新闻"
    finally:
        reset_vendor_observer(token)


def test_placeholder_failure_text_is_reported_as_reason():
    events = []
    token = set_vendor_observer(events.append)
    try:
        observed_call("fetch_reddit", "Reddit", lambda: "<Reddit unavailable: fetch failed (r/a): HTTP 429 after one retry; x>")
    finally:
        reset_vendor_observer(token)
    assert events[0]["outcome"] == "failed" and "HTTP 429" in events[0]["error"]
