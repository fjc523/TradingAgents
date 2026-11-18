"""只从 TLS 异常提取已知类型和数字，不输出异常原文或网络配置值。"""

import os
import ssl
from urllib.request import getproxies_environment

import requests

_TLS_TYPES = {
    'SSLError', 'SSLCertVerificationError', 'CertificateVerifyError',
    'SSLEOFError', 'SSLZeroReturnError', 'SSLWantReadError', 'SSLWantWriteError',
}
_SSL_REASONS = {
    'CERTIFICATE_VERIFY_FAILED', 'TLSV1_ALERT_UNKNOWN_CA', 'SSLV3_ALERT_HANDSHAKE_FAILURE',
    'UNEXPECTED_EOF_WHILE_READING', 'WRONG_VERSION_NUMBER', 'TLSV1_ALERT_PROTOCOL_VERSION',
    'SSLV3_ALERT_BAD_CERTIFICATE', 'UNSUPPORTED_PROTOCOL', 'NO_SHARED_CIPHER',
}
_CA_ENV = ('REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE', 'SSL_CERT_FILE')


def _tls_facts(error):
    pending, visited = [error], set()
    facts = []
    while pending:
        item = pending.pop()
        if id(item) in visited:
            continue
        visited.add(id(item))
        if isinstance(item, (list, tuple)):
            pending.extend(reversed(item))
            continue
        if not isinstance(item, BaseException):
            continue
        name = type(item).__name__
        if name in _TLS_TYPES or isinstance(item, ssl.SSLError):
            details = [name if name in _TLS_TYPES else 'SSLError']
            for key in ('verify_code', 'errno', 'code'):
                number = getattr(item, key, None)
                if isinstance(number, int) and not isinstance(number, bool) and 0 < number <= 65535:
                    details.append(f'{key}={int(number)}')
            reason = getattr(item, 'reason', None)
            if isinstance(item, ssl.SSLError) and isinstance(reason, str) and reason in _SSL_REASONS:
                details.append(f'reason={reason}')
            diagnostic = ' '.join(details)
            if diagnostic not in facts:
                facts.append(diagnostic)
        pending.extend(reversed((item.__cause__, item.__context__, getattr(item, 'reason', None),
                                 getattr(item, 'original_error', None), getattr(item, 'exceptions', ()), item.args)))
    return facts


def tls_diagnostic(error, *, url=None, backend='requests', session=None, verify=None):
    """生成可见诊断。代理字段只表示环境代理是否匹配，不能证明实际网络路径。"""
    facts = _tls_facts(error)
    if not facts:
        return None

    trust_env = getattr(session, 'trust_env', True)
    proxies = getproxies_environment() if trust_env else {}
    try:
        proxy = (bool(requests.utils.select_proxy(url, proxies))
                 and not requests.utils.should_bypass_proxies(url, no_proxy=proxies.get('no'))) if url else (
                     any(proxies.get(name) for name in ('https', 'http', 'all')))
    except ValueError:
        proxy = False

    session_verify = verify if isinstance(verify, str) else getattr(session, 'verify', True)
    if backend == 'curl':
        from curl_cffi.curl import DEFAULT_CACERT

        bundle = session_verify if isinstance(session_verify, str) else DEFAULT_CACERT
        ca = next((name for name in _CA_ENV if os.environ.get(name) == bundle), None)
        if ca is None:
            ca = ('会话配置' if isinstance(session_verify, str) else
                  'certifi' if bundle == requests.certs.where() else '系统默认')
    else:
        ca = next((name for name in _CA_ENV[:2] if trust_env and os.environ.get(name)), None)
        if ca is None:
            ca = '会话配置' if isinstance(session_verify, str) else 'certifi'
    return f"TLS {' / '.join(facts)}；代理环境={'true' if proxy else 'false'}；CA来源={ca}"
