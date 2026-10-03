"""财报时效、来源回退、历史截止和不推算Q4的固定输入验收。"""
from datetime import date

import pytest

from tradingagents.dataflows import router
from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.errors import StaleVendorDataError, VendorUnavailableError
from tradingagents.dataflows.vendors import sec_edgar


@pytest.fixture
def filings():
    return {'filings': {'recent': {
        'form': ['10-Q', '10-Q', '8-K'], 'reportDate': ['2026-07-26', '2026-04-26', '2026-08-20'],
        'filingDate': ['2026-08-26', '2026-05-27', '2026-08-20'], 'items': ['', '', '2.02,9.01'],
    }}}


@pytest.fixture
def facts():
    return {'facts': {'us-gaap': {
        'Assets': {'units': {'USD': [{'end': '2026-04-26', 'filed': '2026-05-27', 'val': 1000, 'form': '10-Q'}]}},
        'Revenues': {'units': {'USD': [{'start': '2026-01-26', 'end': '2026-04-26', 'filed': '2026-05-27', 'val': 1000, 'form': '10-Q'}]}},
        'NetCashProvidedByUsedInOperatingActivities': {'units': {'USD': [{'start': '2026-01-26', 'end': '2026-04-26', 'filed': '2026-05-27', 'val': 1000, 'form': '10-Q'}]}},
    }}}


def install(monkeypatch, facts, filings):
    monkeypatch.setattr(sec_edgar, 'cik_for', lambda ticker: '0000088941')
    def cached(url, name):
        if '/submissions/' in url:
            if isinstance(filings, Exception):
                raise filings
            return filings
        assert 'companyfacts' in url
        return facts
    monkeypatch.setattr(sec_edgar, '_cached_json', cached)


def test_latest_filing_filters_future_and_exposes_only_real_item202(monkeypatch, filings, facts):
    install(monkeypatch, facts, filings)
    current = sec_edgar.latest_periodic_filing('0000088941', '2026-10-02')
    assert current['reportDate'] == '2026-07-26' and current['last_earnings_filing_date'] == '2026-08-20'
    before = sec_edgar.latest_periodic_filing('0000088941', '2026-08-01')
    assert before['reportDate'] == '2026-04-26' and 'last_earnings_filing_date' not in before


@pytest.mark.parametrize('method', ['get_balance_sheet', 'get_income_statement', 'get_cashflow'])
def test_stale_each_statement_and_pre_filing_is_fresh(monkeypatch, facts, filings, method):
    install(monkeypatch, facts, filings)
    with pytest.raises(StaleVendorDataError) as caught:
        getattr(sec_edgar, method)('SMTC', 'quarterly', '2026-10-02')
    assert '2026-07-26' in str(caught.value) and '2026-08-26' in str(caught.value)
    assert caught.value.statement_metadata['latest_period'] == '2026-04-26'
    text = getattr(sec_edgar, method)('SMTC', 'quarterly', '2026-08-01')
    assert text.statement_metadata['stale'] is False and '⚠' not in text


def test_submissions_unavailable_uses_exact_135_day_boundary(monkeypatch, facts):
    install(monkeypatch, facts, VendorUnavailableError('固定403'))
    # 2026-04-26至2026-09-08恰135天；9月9为136天。
    assert (date(2026, 9, 8) - date(2026, 4, 26)).days == 135
    assert '未超过135天' in sec_edgar.get_balance_sheet('SMTC', 'quarterly', '2026-09-08')
    with pytest.raises(StaleVendorDataError, match='超过135天'):
        sec_edgar.get_balance_sheet('SMTC', 'quarterly', '2026-09-09')


@pytest.mark.parametrize('fallback', ['fresh', 'stale', 'unavailable', 'withheld'])
def test_router_checks_fallback_and_preserves_warned_sec(monkeypatch, facts, filings, fallback):
    install(monkeypatch, facts, filings)
    def other(*args, **kwargs):
        if fallback == 'unavailable':
            raise VendorUnavailableError('固定超时')
        if fallback == 'withheld':
            return '# Income Statement\nIncome Statement data is withheld for this date.'
        return '# 后备来源\n,' + ('2026-07-26' if fallback == 'fresh' else '2026-04-26') + '\nRevenue,999\n'
    monkeypatch.setitem(router.VENDOR_METHODS, 'get_income_statement', {'sec_edgar': sec_edgar.get_income_statement, 'yfinance': other})
    with run_config({'tool_vendors': {'get_income_statement': 'sec_edgar,yfinance'}}):
        text = router.route_to_vendor('get_income_statement', 'SMTC', 'quarterly', '2026-10-02')
    if fallback == 'fresh':
        assert '财报来源：yfinance' in text and '回退原因' in text
        assert text.statement_metadata['latest_period'] == '2026-07-26' and not text.statement_metadata['stale']
    else:
        assert text.startswith('⚠ 最新季报') and '2026-07-26' in text
        assert '2026-04-26' in text and text.statement_metadata['stale']
        assert text.statement_metadata['actual_source'] == 'sec_edgar'


def test_real_yahoo_pit_guard_is_not_bypassed(monkeypatch, facts, filings):
    from tradingagents.dataflows.vendors.yahoo import fundamentals
    install(monkeypatch, facts, filings)
    monkeypatch.setattr(fundamentals, 'withhold_undated_statements', lambda *a: '# Income Statement\nIncome Statement data is withheld for this date.')
    monkeypatch.setattr(fundamentals.yf, 'Ticker', lambda *a: pytest.fail('PIT防护后不读取当前报表'))
    monkeypatch.setitem(router.VENDOR_METHODS, 'get_income_statement', {'sec_edgar': sec_edgar.get_income_statement, 'yfinance': fundamentals.get_income_statement})
    with run_config({'tool_vendors': {'get_income_statement': 'sec_edgar,yfinance'}}):
        text = router.route_to_vendor('get_income_statement', 'SMTC', 'quarterly', '2026-10-02')
    assert text.startswith('⚠') and text.statement_metadata['actual_source'] == 'sec_edgar'


@pytest.mark.parametrize('method,tag', [('get_income_statement', 'Revenues'), ('get_cashflow', 'NetCashProvidedByUsedInOperatingActivities')])
def test_annual_facts_are_covered_but_quarterly_q4_not_invented(monkeypatch, facts, method, tag):
    latest = {'filings': {'recent': {'form': ['10-K'], 'reportDate': ['2026-06-30'], 'filingDate': ['2026-08-15']}}}
    annual_fact = {'start': '2025-07-01', 'end': '2026-06-30', 'filed': '2026-08-15', 'val': 9999, 'form': '10-K'}
    facts['facts']['us-gaap'][tag]['units']['USD'].append(annual_fact)
    install(monkeypatch, facts, latest)
    text = getattr(sec_edgar, method)('COHR', 'quarterly', '2026-10-02')
    assert '年度申报2026-06-30已收录' in text and '第四季度未单列' in text
    assert text.statement_metadata['latest_period'] == '2026-04-26'
    assert text.statement_metadata['annual_covered_period'] == '2026-06-30'
    assert text.statement_metadata['stale'] is False
    # 无关资产/收入tag的年度事实不能免除此报表的陈旧。
    other_method = 'get_cashflow' if method == 'get_income_statement' else 'get_income_statement'
    with pytest.raises(StaleVendorDataError):
        getattr(sec_edgar, other_method)('COHR', 'quarterly', '2026-10-02')


def test_assets_need_own_latest_period_and_annual_ignores_new_quarter(monkeypatch, facts, filings):
    facts['facts']['us-gaap']['Assets']['units']['USD'].append({'end': '2026-01-25', 'filed': '2026-03-23', 'val': 111, 'form': '10-K'})
    filings['filings']['recent']['form'].append('10-K')
    filings['filings']['recent']['reportDate'].append('2026-01-25')
    filings['filings']['recent']['filingDate'].append('2026-03-23')
    install(monkeypatch, facts, filings)
    annual = sec_edgar.get_balance_sheet('SMTC', 'annual', '2026-10-02')
    assert annual.statement_metadata['expected_period'] == '2026-01-25'
    assert annual.statement_metadata['stale'] is False
    with pytest.raises(StaleVendorDataError):
        sec_edgar.get_balance_sheet('SMTC', 'quarterly', '2026-10-02')


def test_submissions_cache_is_separate_and_24h_reused(monkeypatch, tmp_path, filings):
    calls = []
    monkeypatch.setattr(sec_edgar, '_fetch_json', lambda url: calls.append(url) or filings)
    with run_config({'data_cache_dir': str(tmp_path)}):
        first = sec_edgar.latest_periodic_filing('88941', '2026-10-02')
        second = sec_edgar.latest_periodic_filing('88941', '2026-08-01')
    assert first['reportDate'] == '2026-07-26' and second['reportDate'] == '2026-04-26'
    assert len(calls) == 1 and calls[0].endswith('CIK0000088941.json')
    assert (tmp_path/'sec_edgar/submissions_CIK0000088941.json').exists()
    assert not (tmp_path/'sec_edgar/CIK0000088941.json').exists()


def test_old_as_of_reads_official_archived_columnar_filings(monkeypatch):
    current = {'filings': {'recent': {'form': ['10-Q'], 'reportDate': ['2026-07-26'], 'filingDate': ['2026-08-26']},
                           'files': [{'name': 'older.json', 'filingFrom': '2010-01-01', 'filingTo': '2025-01-01'}]}}
    old = {'form': ['10-K'], 'reportDate': ['2019-12-31'], 'filingDate': ['2020-02-01']}
    calls = []
    monkeypatch.setattr(sec_edgar, '_cached_json', lambda url, name: calls.append((url,name)) or (old if 'older.json' in url else current))
    assert sec_edgar.latest_periodic_filing('88941', '2020-03-01')['reportDate'] == '2019-12-31'
    assert calls[-1][0] == 'https://data.sec.gov/submissions/older.json'


@pytest.mark.parametrize('body', ['', 'Revenue,\n', 'Revenue,NaN\n', 'Revenue,unavailable\n'])
def test_empty_or_placeholder_new_period_is_not_fresh(monkeypatch, facts, filings, body):
    install(monkeypatch, facts, filings)
    monkeypatch.setitem(router.VENDOR_METHODS, 'get_income_statement', {
        'sec_edgar': sec_edgar.get_income_statement,
        'yfinance': lambda *a: '# 固定后备\n,2026-07-26\n' + body})
    with run_config({'tool_vendors': {'get_income_statement': 'sec_edgar,yfinance'}}):
        text = router.route_to_vendor('get_income_statement', 'SMTC', 'quarterly', '2026-10-02')
    assert text.startswith('⚠') and text.statement_metadata['stale']


def test_period_parser_requires_actual_value_and_preserves_zero():
    from tradingagents.dataflows.statement_freshness import latest_statement_period
    assert latest_statement_period(',2026-04-26,2026-07-26\nRevenue,10,') == '2026-04-26'
    assert latest_statement_period(',2026-07-26\nRevenue,0') == '2026-07-26'
    assert latest_statement_period(',2026-07-26\nRevenue,inf') is None


@pytest.mark.parametrize('malformed', [[], None, {'filings': {'recent': []}}, {'filings': {'recent': {'form': '10-Q'}}}])
def test_malformed_submissions_is_typed_and_135_rule_still_applies(monkeypatch, facts, malformed):
    install(monkeypatch, facts, malformed)
    with pytest.raises(VendorUnavailableError):
        sec_edgar.latest_periodic_filing('88941', '2026-10-02')
    with pytest.raises(StaleVendorDataError, match='超过135天'):
        sec_edgar.get_balance_sheet('SMTC', 'quarterly', '2026-10-02')


def test_malformed_archive_is_typed_unavailable(monkeypatch):
    current = {'filings': {'recent': {'form': []}, 'files': [{'name': 'old.json', 'filingFrom': '2000-01-01'}]}}
    monkeypatch.setattr(sec_edgar, '_cached_json', lambda url, name: [] if 'old.json' in url else current)
    with pytest.raises(VendorUnavailableError, match='列式表结构非法'):
        sec_edgar.latest_periodic_filing('88941', '2020-01-01')


@pytest.mark.parametrize('table', [
    {'form': ['10-Q'], 'reportDate': ['2026-07-26']},
    {'form': ['10-Q'], 'filingDate': '2026-08-26', 'reportDate': ['2026-07-26']},
    {'form': ['10-Q'], 'filingDate': [], 'reportDate': ['2026-07-26']},
    {'form': ['10-Q'], 'filingDate': ['2026-08-26'], 'reportDate': []},
    {'form': ['10-Q'], 'filingDate': ['错误日期'], 'reportDate': ['2026-07-26']},
    {'form': ['10-Q'], 'filingDate': ['2026-08-26'], 'reportDate': ['']},
    {'form': ['10-Q'], 'filingDate': ['2026-08-26'], 'reportDate': ['错误日期']},
    {'form': ['10-Q'], 'filingDate': ['2026-08-26'], 'reportDate': ['2026-09-26']},
])
def test_required_submissions_dates_cannot_silently_bypass_age_guard(monkeypatch, facts, table):
    install(monkeypatch, facts, {'filings': {'recent': table}})
    with pytest.raises(VendorUnavailableError):
        sec_edgar.latest_periodic_filing('88941', '2026-10-02')
    with pytest.raises(StaleVendorDataError, match='超过135天'):
        sec_edgar.get_balance_sheet('SMTC', 'quarterly', '2026-10-02')


def test_nonperiodic_empty_report_date_is_allowed_and_archive_rules_are_same(monkeypatch):
    table = {'form': ['8-K', '10-Q'], 'filingDate': ['2026-08-20', '2026-08-26'],
             'reportDate': ['', '2026-07-26'], 'items': ['2.02', '']}
    monkeypatch.setattr(sec_edgar, '_cached_json', lambda *a: {'filings': {'recent': table}})
    assert sec_edgar.latest_periodic_filing('88941', '2026-10-02')['last_earnings_filing_date'] == '2026-08-20'
    current = {'filings': {'recent': {'form': []}, 'files': [{'name': 'bad-old.json', 'filingFrom': '2000-01-01'}]}}
    monkeypatch.setattr(sec_edgar, '_cached_json', lambda url, name: {'form': ['10-K'], 'reportDate': ['2019-12-31']}
                        if 'bad-old.json' in url else current)
    with pytest.raises(VendorUnavailableError, match='filingDate'):
        sec_edgar.latest_periodic_filing('88941', '2020-01-01')
