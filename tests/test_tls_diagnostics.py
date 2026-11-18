"""用真实异常类的离线夹具验证供应商报错链和脱敏。"""

import ssl
import traceback
from unittest.mock import Mock

import pytest
import requests
from urllib3 import exceptions as urllib_errors

from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.vendors import sec_edgar
from tradingagents.dataflows.vendors.alpaca.client import AlpacaClient


@pytest.mark.parametrize('vendor', ['alpaca', 'sec_json', 'sec_xbrl'])
def test_tls_wrapper_does_not_keep_sensitive_exception_chain(monkeypatch, tmp_path, caplog, vendor):
    proxy = 'https://synthetic-name:synthetic-password@proxy-canary.invalid'
    bundle = '/private/ca-path-canary'
    monkeypatch.setenv('https_proxy', proxy)
    monkeypatch.setenv('no_proxy', '')
    monkeypatch.setenv('REQUESTS_CA_BUNDLE', bundle)
    monkeypatch.setenv('APCA_API_KEY_ID', 'offline-fixture')
    monkeypatch.setenv('APCA_API_SECRET_KEY', 'offline-fixture')
    monkeypatch.setenv('SEC_EDGAR_USER_AGENT', 'SEC fixture contact@fixture-research.edu')
    original = ssl.SSLCertVerificationError(1, 'raw-canary https://host.invalid/?token=token-canary Cookie=cookie-canary')
    original.verify_code = 20
    original.reason = 'CERTIFICATE_VERIFY_FAILED'
    nested = urllib_errors.MaxRetryError(None, '/?token=sensitive', reason=urllib_errors.SSLError(original))

    def fail(*args, **kwargs):
        assert kwargs.get('verify', True) is True
        raise requests.exceptions.ProxyError(urllib_errors.ProxyError('proxy-message', nested))

    if vendor == 'alpaca':
        session = Mock(spec=requests.Session)
        session.get.side_effect = fail
        call = lambda: AlpacaClient(session=session)._get('https://data.alpaca.markets/test-data', {})
    else:
        monkeypatch.setattr(requests, 'get', fail)
        if vendor == 'sec_json':
            call = lambda: sec_edgar._fetch_json('https://data.sec.gov/test-data')
        else:
            monkeypatch.setattr(sec_edgar, 'get_config', lambda: {'data_cache_dir': str(tmp_path)})
            monkeypatch.setattr(sec_edgar, '_fetch_json', lambda url: {'directory': {'item': [{'name': 'facts.xml'}]}})
            call = lambda: sec_edgar._filing_facts('123', '0000-00-000000', '2026-10-01', '10-Q')

    with pytest.raises(VendorUnavailableError) as failed:
        call()
    error = failed.value
    assert error.__cause__ is None and error.__context__ is None
    diagnostic = ''.join(traceback.format_exception(error)) + caplog.text
    assert 'SSLCertVerificationError verify_code=20 errno=1 reason=CERTIFICATE_VERIFY_FAILED' in diagnostic
    assert '代理环境=true' in diagnostic and 'CA来源=REQUESTS_CA_BUNDLE' in diagnostic
    for secret in ('proxy-canary', 'synthetic-name', 'synthetic-password', 'ca-path-canary',
                   'raw-canary', 'token-canary', 'cookie-canary'):
        assert secret not in diagnostic
