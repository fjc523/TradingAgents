"""动态工具与批次熔断行为。"""
import pytest
from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows import router
from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.yahoo_breaker import reset_yahoo_breaker
from tradingagents.dataflows.vendors.yahoo.common import yf_retry
from tradingagents.agents.analysts.news_analyst import available_news_tools


def test_register_vendor_method_routes_to_injected_implementation(monkeypatch):
    monkeypatch.setattr(router,'VENDOR_METHODS',{**router.VENDOR_METHODS,'get_news':dict(router.VENDOR_METHODS['get_news'])})
    router.register_vendor_method('get_news','sample',lambda *args:'样本新闻')
    with run_config({'tool_vendors':{'get_news':'sample'}}):
        assert router.route_to_vendor('get_news','TSLA','2026-09-01','2026-10-01')=='样本新闻'
    with pytest.raises(ValueError):router.register_vendor_method('nonexistent','sample',lambda:None)


def test_macro_tool_is_filtered_only_for_unconfigured_fred(monkeypatch):
    monkeypatch.delenv('FRED_API_KEY',raising=False)
    with run_config({'tool_vendors':{'get_macro_indicators':'fred'}}):
        assert 'get_macro_indicators' not in [tool.name for tool in available_news_tools()]
        monkeypatch.setenv('FRED_API_KEY','fixture')
        assert 'get_macro_indicators' in [tool.name for tool in available_news_tools()]
    monkeypatch.delenv('FRED_API_KEY',raising=False)
    with run_config({'tool_vendors':{'get_macro_indicators':'futu,fred'}}):
        assert 'get_macro_indicators' in [tool.name for tool in available_news_tools()]


def test_first_connection_failure_trips_until_next_batch():
    calls=[]
    def fail():calls.append(1);raise TimeoutError('timeout')
    with pytest.raises(VendorUnavailableError):yf_retry(fail)
    with pytest.raises(VendorUnavailableError):yf_retry(fail)
    assert len(calls)==1
    reset_yahoo_breaker()
    assert yf_retry(lambda:'恢复')=='恢复'
