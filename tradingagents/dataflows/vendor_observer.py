"""按分析上下文隔离来源观察者，不改变取数和降级语义。"""
from contextvars import ContextVar
from datetime import datetime, timezone
from time import perf_counter

_OBSERVER = ContextVar("vendor_observer", default=None)


def observer_enabled():
    return _OBSERVER.get() is not None


def set_vendor_observer(callback):
    return _OBSERVER.set(callback)


def reset_vendor_observer(token):
    _OBSERVER.reset(token)


def report_vendor(method, vendor, outcome, *, duration=0, error=None, symbol=None, statement_metadata=None):
    callback = _OBSERVER.get()
    if callback is None:
        return
    try:
        callback({"method": method, "source": vendor, "outcome": outcome,
                  "duration_seconds": duration, "error": error, "symbol": symbol,
                  "recorded_at": datetime.now(timezone.utc).isoformat(),
                  **({"statement_metadata": statement_metadata} if statement_metadata else {})})
    except Exception:
        # 观察者不能改变来源调用本身的结果。
        pass


def observed_call(method, vendor, impl, *args, observation_symbol=None, **kwargs):
    if _OBSERVER.get() is None:
        return impl(*args, **kwargs)
    from tradingagents.dataflows.errors import VendorNotConfiguredError, NoMarketDataError
    started = perf_counter()
    try:
        value = impl(*args, **kwargs)
    except Exception as exc:
        outcome = "unconfigured" if isinstance(exc, VendorNotConfiguredError) else "no_data" if isinstance(exc, NoMarketDataError) else "failed"
        report_vendor(method, vendor, outcome, duration=perf_counter()-started,
                      error=type(exc).__name__+"："+str(exc), symbol=observation_symbol)
        raise
    text = str(value) if isinstance(value, str) else ""
    if "unavailable:" in text.lower() or text.startswith(("DATA_UNAVAILABLE", "NO_DATA_AVAILABLE")):
        outcome = "failed"
    elif text.startswith(("<no ", "<none ")) or value is None:
        outcome = "no_data"
    else:
        outcome = "success"
    # 占位文本本身说明了失败原因（如 HTTP 状态码），直接透传；只有空值才写泛化原因。
    outcome = getattr(value, "source_outcome", outcome)
    reason = getattr(value, "source_reason", None) or (None if outcome == "success" else (text.strip()[:200] or "没有可用数据"))
    report_vendor(method, vendor, outcome, duration=perf_counter()-started,
                  error=reason, symbol=observation_symbol,
                  statement_metadata=getattr(value, "statement_metadata", None))
    return value
