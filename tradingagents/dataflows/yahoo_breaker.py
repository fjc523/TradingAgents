"""同一批次首次连接或限频失败后跳过Yahoo网络请求。"""
from copy import deepcopy
from datetime import datetime, timezone
import re
import threading
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit
from uuid import uuid4
from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.tls_diagnostics import tls_diagnostic

_lock = threading.Lock()
_reason = None
_session = None
_first_trigger = None


def _transport_diagnostic(args, kwargs, error):
    try:
        return _transport_diagnostic_impl(args, kwargs, error)
    except Exception:
        return {'method':'UNKNOWN','url':None,'error':'传输诊断构造失败；原异常保持不变','symbol':None}


def _transport_diagnostic_impl(args, kwargs, error):
    """仅留下可定位的操作和host/path；凭据、query和headers不落盘。"""
    from tradingagents.llm_clients.codex_exec.audit import redact
    raw_url = str(kwargs.get('url') or (args[1] if len(args) > 1 else ''))
    method = str(kwargs.get('method') or (args[0] if args else 'UNKNOWN')).upper()
    secrets = []
    try:
        parts = urlsplit(raw_url)
        secrets.extend(value for _, value in parse_qsl(parts.query))
        secrets.extend(filter(None, (parts.username, parts.password)))
        # hostname剥离userinfo；port在合法URL中保留，异常URL不推测。
        host = parts.hostname or ''
        if parts.port:
            host += ':' + str(parts.port)
        url = urlunsplit((parts.scheme, host, parts.path, '', '')) if parts.scheme in ('http','https') and host else None
    except ValueError:
        url = None
    for key in ('params','headers','cookies'):
        values = kwargs.get(key)
        if hasattr(values, 'values'):
            secrets.extend(str(value) for value in values.values())
    message = str(error)
    for secret in sorted(set(filter(None, secrets)), key=len, reverse=True):
        message = message.replace(secret, '[已脱敏]')
    message = redact(message)
    message = re.sub(r'https?://[^\s\"\'<>]+', '[请求URL已脱敏]', message)
    message = re.sub(r'(?i)((?:crumb|cookie|set-cookie|authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|secret)[\"\']?\s*[:=]\s*[\"\']?)(?:bearer\s+)?[^\s,;\"\']+', r'\1[已脱敏]', message)
    symbol = unquote(urlsplit(url).path.rsplit('/',1)[-1]) if url and '/v8/finance/chart/' in urlsplit(url).path else None
    return {'method':method,'url':url,'error':message[:2000],'symbol':symbol}


def reset_yahoo_breaker():
    global _reason, _first_trigger
    with _lock:
        _reason = None
        _first_trigger = None


def trip_yahoo_breaker(reason, *, diagnostic=None):
    global _reason, _first_trigger
    created = False
    with _lock:
        if _reason is None:
            _reason = str(reason)
            _first_trigger = {'event_id':uuid4().hex,'kind':'first_breaker_trigger',
                'recorded_at':datetime.now(timezone.utc).isoformat(),'reason':_reason,
                **{key:(diagnostic or {}).get(key, _reason if key=='error' else 'UNKNOWN' if key=='method' else None) for key in ('method','url','error','symbol')}}
            if (diagnostic or {}).get('tls'):
                _first_trigger['tls'] = diagnostic['tls']
            created = True
        first = deepcopy(_first_trigger)
    # 文件IO和观察者可能重入；不能持熔断锁通知。顺序指首次成功trip，不指物理网络先后。
    if created:
        from tradingagents.dataflows.vendor_observer import report_vendor
        report_vendor('yahoo_provider_health','yfinance','failed',provider_health=first)
    return first


def check_yahoo_breaker():
    with _lock:
        reason, first = _reason, deepcopy(_first_trigger)
    if reason is not None:
        from tradingagents.dataflows.vendor_observer import report_vendor
        report_vendor('yahoo_breaker', 'yfinance', 'failed', error='批次熔断，跳过请求', provider_health=first)
        error = VendorUnavailableError(f'Yahoo已熔断：{reason}')
        error.provider_health_event_id = first['event_id'] if first else None
        error.yahoo_tls_details = first.get('tls') if first else None
        raise error


def yahoo_tls_details(error):
    return _chain_value(error, 'yahoo_tls_details')


def provider_health_event_id(error):
    return _chain_value(error, 'provider_health_event_id')


def _chain_value(error, field):
    """沿原异常链取首事件引用，不改变异常传播或补造请求。"""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        value = getattr(error, field, None)
        if value:
            return value
        error = error.__cause__ or error.__context__
    return None


def is_connection_failure(error):
    text = (type(error).__name__ + ' ' + str(error)).lower()
    return any(word in text for word in ('429', 'ratelimit', 'rate limit', 'timeout', 'timed out', 'connect', 'ssl', 'tls', 'curl', 'resolve', 'network'))


def ensure_bounded_session():
    """通过SDK共享session限制每次HTTP等待，含info等无timeout参数入口。"""
    global _session
    with _lock:
        if _session is None:
            from curl_cffi.requests import Session
            from curl_cffi.requests.exceptions import CertificateVerifyError, SSLError
            from yfinance.data import YfData
            class BoundedSession(Session):
                def _safe_request(self, *args, **kwargs):
                    try:
                        return super().request(*args, **kwargs)
                    except Exception as exc:
                        url = kwargs.get('url') or (args[1] if len(args) > 1 else None)
                        detail = tls_diagnostic(exc, url=url, backend='curl', session=self,
                                                verify=kwargs.get('verify'))
                        if detail is None:
                            raise
                        error_type = CertificateVerifyError if isinstance(exc, CertificateVerifyError) else SSLError
                        code = getattr(exc, 'code', 0)
                        error = error_type(detail, code=code if type(code) is int else 0)
                        error.yahoo_tls_details = detail
                    # 脱敏后再进入观察者与 yfinance，避免记录 TLS 异常原文或 traceback。
                    raise error

                def request(self, *args, **kwargs):
                    check_yahoo_breaker()
                    requested = kwargs.get('timeout')
                    kwargs['timeout'] = min(requested, 10) if isinstance(requested, (int, float)) else 10
                    try:
                        from tradingagents.dataflows.vendor_observer import observed_call
                        response = observed_call('yahoo_request', 'yfinance', self._safe_request, *args, **kwargs)
                        if response.status_code == 429:
                            trip_yahoo_breaker('HTTP 429', diagnostic=_transport_diagnostic(args, kwargs, 'HTTP 429'))
                        return response
                    except Exception as exc:
                        if is_connection_failure(exc):
                            detail = yahoo_tls_details(exc)
                            diagnostic = _transport_diagnostic(args, kwargs, detail or type(exc).__name__+'：'+str(exc))
                            if detail:
                                diagnostic['tls'] = detail
                            first = trip_yahoo_breaker(type(exc).__name__, diagnostic=diagnostic)
                            exc.provider_health_event_id = first['event_id']
                        raise
            _session = BoundedSession(impersonate='chrome')
            YfData(session=_session)
