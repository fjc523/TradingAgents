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


def report_vendor(method, vendor, outcome, *, duration=0, error=None, symbol=None):
    callback = _OBSERVER.get()
    if callback is None:
        return
    try:
        callback({"method": method, "source": vendor, "outcome": outcome,
                  "duration_seconds": duration, "error": error, "symbol": symbol,
                  "recorded_at": datetime.now(timezone.utc).isoformat()})
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
    report_vendor(method, vendor, outcome, duration=perf_counter()-started,
                  error="没有可用数据" if outcome != "success" else None, symbol=observation_symbol)
    return value
