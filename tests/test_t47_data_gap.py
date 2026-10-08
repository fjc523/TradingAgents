"""T47定向合同；真实XML节选来自投研原件，无实时恢复声明。"""
from datetime import datetime, UTC
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import pytest

from tradingagents.dataflows.vendors import sec_edgar as sec, reddit
from tradingagents.dataflows.vendors.alpaca.client import AlpacaClient
from tradingagents.dataflows.errors import StaleVendorDataError

XML = Path(__file__).parent / 'fixtures/t47_smtc_instance.xml'
ACC = '0000088941-26-000034'

@pytest.mark.parametrize('count,token,truncated', [(50,'next',True),(19,None,False),(20,None,False),(21,None,True),(20,'next',True)])
def test_news_limit(count,token,truncated,monkeypatch):
    client = object.__new__(AlpacaClient)
    monkeypatch.setattr(client,'_get',lambda *a: {'news':[{'id':i,'created_at':'2026-10-08T12:00:00Z'} for i in range(count)],'next_page_token':token})
    result = client.get_news(['TSLA'],datetime(2026,10,8,tzinfo=UTC),datetime(2026,10,9,tzinfo=UTC),limit=20)
    assert result['truncated'] is truncated
    assert len(result['articles']) == min(count,20)

@pytest.fixture
def instance(tmp_path,monkeypatch):
    monkeypatch.setattr(sec,'get_config',lambda:{'data_cache_dir':str(tmp_path)})
    (tmp_path/'sec_edgar').mkdir()
    (tmp_path/'sec_edgar'/f'filing_{ACC}.xml').write_bytes(XML.read_bytes())
    return sec._filing_facts('0000088941',ACC,'2026-08-26','10-Q')

@pytest.mark.parametrize('tag,value,span',[
    ('RevenueFromContractWithCustomerExcludingAssessedTax',341871000,0),
    ('GrossProfit',183764000,0),('OperatingIncomeLoss',55772000,0),
    ('NetIncomeLoss',160121000,0),('EarningsPerShareDiluted',1.59,0),
    ('Assets',1647041000,0),('NetCashProvidedByUsedInOperatingActivities',105071000,1),
    ('PaymentsToAcquirePropertyPlantAndEquipment',15663000,1),
])
def test_real_xml_exact_values(instance,tag,value,span):
    values,unit=sec._as_of(instance,(tag,),'2026-10-08',((60,115),(150,200),(240,290)))
    assert values[('2026-07-26',span)] == value
    assert unit == ('USD/shares' if tag=='EarningsPerShareDiluted' else 'USD')
    assert all(f['filed']=='2026-08-26' and f['accn']==ACC for data in instance.values() for v in data['units'].values() for f in v)


def test_real_xml_tax_without_extending_statement_tags():
    root=ET.parse(XML).getroot(); ns={'x':'http://www.xbrl.org/2003/instance'}
    contexts={c.get('id'):c for c in root.findall('x:context',ns)}
    values=[]
    for node in root:
        if node.tag.endswith('}IncomeTaxExpenseBenefit'):
            c=contexts[node.get('contextRef')]
            if c.find('.//x:segment',ns) is None and c.findtext('x:period/x:endDate',namespaces=ns)=='2026-07-26':
                start=c.findtext('x:period/x:startDate',namespaces=ns)
                if (datetime(2026,7,26)-datetime.fromisoformat(start)).days in range(60,116): values.append(float(node.text))
    assert -101361000 in values
    assert 'IncomeTaxExpenseBenefit' not in {t for lines in sec._STATEMENTS.values() for _,tags in lines for t in tags}


def setup_statement(monkeypatch,instance,as_of):
    old={'Assets':{'units':{'USD':[{'end':'2026-04-26','val':1,'filed':'2026-05-27','form':'10-Q'}]}},
         'NetIncomeLoss':{'units':{'USD':[{'end':'2026-04-26','start':'2026-01-26','val':1,'filed':'2026-05-27','form':'10-Q'}]}},
         'NetCashProvidedByUsedInOperatingActivities':{'units':{'USD':[{'end':'2026-04-26','start':'2026-01-26','val':1,'filed':'2026-05-27','form':'10-Q'}]}}}
    monkeypatch.setattr(sec,'cik_for',lambda ticker:'0000088941')
    monkeypatch.setattr(sec,'_cached_json',lambda *a:{'facts':{'us-gaap':old}})
    filing={'form':'10-Q','reportDate':'2026-07-26' if as_of>='2026-08-26' else '2026-04-26','filingDate':'2026-08-26' if as_of>='2026-08-26' else '2026-05-27','isXBRL':1,'accessionNumber':ACC}
    monkeypatch.setattr(sec,'latest_periodic_filing',lambda *a:filing)
    fetch=Mock(return_value=instance);monkeypatch.setattr(sec,'_filing_facts',fetch)
    return old,fetch


def test_three_statements_and_pit(instance,monkeypatch):
    old,fetch=setup_statement(monkeypatch,instance,'2026-10-08')
    for call in (sec.get_balance_sheet,sec.get_income_statement,sec.get_cashflow):
        result=call('SMTC','quarterly','2026-10-08')
        assert result.statement_metadata['latest_period']=='2026-07-26'
        assert result.statement_metadata['actual_source']=='sec_edgar'
        assert result.statement_metadata['filing_instance']==ACC
    assert '2026-07-26 (6 months)' in sec.get_cashflow('SMTC','quarterly','2026-10-08')
    assert 'RevenueFromContractWithCustomerExcludingAssessedTax' not in old
    _,fetch=setup_statement(monkeypatch,instance,'2026-08-25')
    result=sec.get_balance_sheet('SMTC','quarterly','2026-08-25')
    assert result.statement_metadata['expected_period']=='2026-04-26'
    fetch.assert_not_called()


def test_covered_tsla_no_instance(instance,monkeypatch):
    _,fetch=setup_statement(monkeypatch,instance,'2026-10-08')
    monkeypatch.setattr(sec,'_cached_json',lambda *a:{'facts':{'us-gaap':instance}})
    assert sec.get_income_statement('TSLA','quarterly','2026-10-08').statement_metadata['latest_period']=='2026-07-26'
    fetch.assert_not_called()


def test_instance_missing_target_keeps_stale(instance,monkeypatch):
    _,fetch=setup_statement(monkeypatch,instance,'2026-10-08');fetch.return_value={}
    with pytest.raises(StaleVendorDataError): sec.get_balance_sheet('SMTC','quarterly','2026-10-08')


@pytest.fixture
def clock(monkeypatch):
    now=[0.0]; sleeps=[]
    def sleep(seconds): sleeps.append(seconds);now[0]+=seconds
    monkeypatch.setattr(reddit,'_clock',lambda:now[0]);monkeypatch.setattr(reddit,'_pace_sleep',sleep)
    monkeypatch.setattr(reddit.time,'sleep',sleep);monkeypatch.setattr(reddit,'_last_request_at',None)
    monkeypatch.setattr(reddit,'_next_request_at',0.0);monkeypatch.setattr(reddit.random,'uniform',lambda *a:0)
    return now,sleeps

@pytest.mark.parametrize('headers,wait',[({'x-ratelimit-reset':'35'},37),({'Retry-After':'5','x-ratelimit-reset':'35'},5),({'Retry-After':'120'},90),({'Retry-After':'bad','x-ratelimit-reset':'120'},90),({'Retry-After':'-1','x-ratelimit-reset':'nan'},60)])
def test_retry_header_priority(clock,monkeypatch,headers,wait):
    calls=[]
    def open_request(*a,**kw):
        calls.append(1)
        raise HTTPError('https://test',429,'rate',headers,None)
    monkeypatch.setattr(reddit,'urlopen',open_request)
    errors=[]
    assert reddit._fetch_subreddit_rss('TSLA','stocks',10,1,errors=errors) is None
    assert len(calls)==2 and clock[1]==[wait]
    if headers.get('x-ratelimit-reset')=='35': assert 'reset=35s' in errors[0]


def test_success_cooldown_not_capped(clock,monkeypatch):
    response=Mock();response.headers={'x-ratelimit-remaining':'0','x-ratelimit-reset':'120'}
    response.read.return_value=b'<feed xmlns="http://www.w3.org/2005/Atom"/>'
    response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=False)
    monkeypatch.setattr(reddit,'urlopen',lambda *a,**kw:response)
    assert reddit._fetch_subreddit_rss('TSLA','stocks',10,1)==[]
    assert reddit._fetch_subreddit_rss('SMTC','stocks',10,1)==[]
    assert clock[1]==[122]


def test_instance_selection_cache_and_dimension_units(tmp_path,monkeypatch):
    monkeypatch.setattr(sec,'get_config',lambda:{'data_cache_dir':str(tmp_path)})
    monkeypatch.setattr(sec,'_fetch_json',lambda *a:{'directory':{'item':[{'name':'FilingSummary.xml'},{'name':'test_cal.xml'},{'name':'test.xml'}]}})
    calls=[]
    response=Mock();response.content=XML.read_bytes()
    monkeypatch.setattr(sec.requests,'get',lambda url,**kw:calls.append(url) or response)
    first=sec._filing_facts('0000088941',ACC,'2026-08-26','10-Q')
    second=sec._filing_facts('0000088941',ACC,'2026-08-26','10-Q')
    assert first==second and len(calls)==1 and calls[0].endswith('/test.xml')
    # 同tag存在维度值时不能压过无维度原事实；原XML节选包含真实维度context。
    root=ET.parse(XML).getroot(); ns={'x':'http://www.xbrl.org/2003/instance'}
    dimension_ids={c.get('id') for c in root.findall('x:context',ns) if c.find('.//x:segment',ns) is not None}
    assert dimension_ids
    dimension_values={float(n.text) for n in root if n.tag.endswith('}RevenueFromContractWithCustomerExcludingAssessedTax') and n.get('contextRef') in dimension_ids}
    parsed={f['val'] for f in first['RevenueFromContractWithCustomerExcludingAssessedTax']['units']['USD']}
    assert dimension_values - parsed
    assert not (dimension_values - parsed) & parsed


def test_instance_http_failure_keeps_stale(instance,monkeypatch):
    from tradingagents.dataflows.errors import VendorUnavailableError
    _,fetch=setup_statement(monkeypatch,instance,'2026-10-08')
    fetch.side_effect=VendorUnavailableError('SEC原申报实例获取失败')
    with pytest.raises(StaleVendorDataError): sec.get_income_statement('SMTC','quarterly','2026-10-08')
