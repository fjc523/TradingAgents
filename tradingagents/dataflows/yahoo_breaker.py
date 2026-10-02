"""同一批次首次连接或限频失败后跳过Yahoo网络请求。"""
import threading
from tradingagents.dataflows.errors import VendorUnavailableError

_lock = threading.Lock()
_reason = None
_session = None


def reset_yahoo_breaker():
    global _reason
    with _lock:
        _reason = None


def trip_yahoo_breaker(reason):
    global _reason
    with _lock:
        if _reason is None:
            _reason = str(reason)


def check_yahoo_breaker():
    with _lock:
        if _reason is not None:
            from tradingagents.dataflows.vendor_observer import report_vendor
            report_vendor("yahoo_breaker", "yfinance", "failed", error="批次熔断，跳过请求")
            raise VendorUnavailableError(f"Yahoo已熔断：{_reason}")


def is_connection_failure(error):
    text = (type(error).__name__ + " " + str(error)).lower()
    return any(word in text for word in ("429", "ratelimit", "rate limit", "timeout", "timed out", "connect", "ssl", "tls", "curl", "resolve", "network"))


def ensure_bounded_session():
    """通过SDK共享session限制每次HTTP等待，含info等无timeout参数入口。"""
    global _session
    with _lock:
        if _session is None:
            from curl_cffi.requests import Session
            from yfinance.data import YfData
            class BoundedSession(Session):
                def request(self, *args, **kwargs):
                    check_yahoo_breaker()
                    requested = kwargs.get("timeout")
                    kwargs["timeout"] = min(requested, 10) if isinstance(requested, (int, float)) else 10
                    try:
                        from tradingagents.dataflows.vendor_observer import observed_call
                        response = observed_call("yahoo_request", "yfinance", super().request, *args, **kwargs)
                        if response.status_code == 429:
                            trip_yahoo_breaker("HTTP 429")
                        return response
                    except Exception as exc:
                        if is_connection_failure(exc):
                            trip_yahoo_breaker(type(exc).__name__)
                        raise
            _session = BoundedSession(impersonate="chrome")
            YfData(session=_session)
