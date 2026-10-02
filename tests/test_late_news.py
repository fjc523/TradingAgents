"""补抓新闻的时间窗口、去重、阶段隔离与失败降级。"""
from datetime import datetime, timezone
import pytest
from tradingagents.dataflows.config import run_config
from tradingagents.graph.late_news import refresh_news, with_news_refresh, articles_from_response, render_late_news

NOW = datetime(2026,10,2,13,9,30,tzinfo=timezone.utc)


def state():
    return {'company_of_interest':'TSLA','trade_date':'2026-10-02',
            'news_last_fetched_at':'2026-10-02T13:00:02+00:00'}


def article(title='季度交付', stamp='2026-10-02T13:04:19Z', url='https://example.test/delivery'):
    return {'title':title, 'time_published':stamp, 'source':'Benzinga','summary':'季度交付增长', 'url':url}


def config(**overrides):
    return {'late_news_refresh':True,'_late_news_clock':lambda: NOW, **overrides}


def test_refresh_filters_deduplicates_and_caps(monkeypatch):
    from tradingagents.agents.tools import get_news
    rows=[article('旧新闻','2026-10-02T13:00:02Z'),article('未来新闻','2026-10-02T13:10:00Z'),article(),article(),
          article('季度交付',url='https://different.test'),article('同链接',url='https://example.test/delivery')]
    rows.extend(article(str(i),url=f'https://example.test/{i}') for i in range(20))
    monkeypatch.setattr(type(get_news), 'invoke', lambda *a,**k: {'feed':rows})
    with run_config(config()): update=refresh_news(state(),'portfolio')
    assert len(update['late_news']) == 10
    assert all(row['title'] not in {'旧新闻','未来新闻'} for row in update['late_news'])
    assert len({row['title'] for row in update['late_news']}) == 10
    assert len({row['url'] for row in update['late_news']}) == 10
    assert update['late_news'][0]['published_at'] == '2026-10-02T13:04:19Z'


@pytest.mark.parametrize('overrides',[{'late_news_refresh':False},{'news_cutoff_utc':'2026-10-02T12:31:00Z'}])
def test_disabled_and_replay_never_fetch(monkeypatch,overrides):
    from tradingagents.agents.tools import get_news
    monkeypatch.setattr(type(get_news),'invoke',lambda *a,**k: pytest.fail('不应补抓'))
    with run_config(config(**overrides)): assert refresh_news(state(),'research') == {}


def test_refresh_failure_keeps_window_and_does_not_block_node(monkeypatch):
    from tradingagents.agents.tools import get_news
    def fail(*args,**kwargs): raise RuntimeError('secret-key不应泄露')
    monkeypatch.setattr(type(get_news),'invoke',fail)
    wrapped=with_news_refresh(lambda s:{'investment_plan':'仍完成'},'research')
    with run_config(config()): result=wrapped(state(),{})
    assert result['investment_plan'] == '仍完成'
    assert result['late_news_errors'] == ['research阶段补抓失败：RuntimeError']
    assert 'news_last_fetched_at' not in result
    assert '数据限制' in render_late_news(result)


def test_portfolio_only_news_never_retroactively_enters_trader(monkeypatch):
    from tradingagents.agents.tools import get_news
    snapshots=[]
    monkeypatch.setattr(type(get_news),'invoke',lambda *a,**k:{'feed':[article()]})
    with run_config(config()):
        original=state()
        snapshots.append(render_late_news(original))
        result=with_news_refresh(lambda s:{'final_trade_decision':render_late_news(s)},'portfolio')(original,{})
    assert snapshots == [''] and not original.get('late_news')
    assert '季度交付' in result['final_trade_decision']
    assert '建议重跑' in result['final_trade_decision'] and '不得声称上游已评估' in result['final_trade_decision']


def test_existing_stage_dedup_and_missing_timestamp(monkeypatch):
    from tradingagents.agents.tools import get_news
    first={**article(),'published_at':'2026-10-02T13:04:19Z','stage':'research','fetched_at':NOW.isoformat()}
    original={**state(),'late_news':[first]}
    monkeypatch.setattr(type(get_news),'invoke',lambda *a,**k:{'feed':[article(),article('无时间',None,'https://example.test/none')]})
    with run_config(config()): result=refresh_news(original,'portfolio')
    assert result['late_news'] == [first]


@pytest.mark.parametrize('response',[{'feed':[article(stamp='20261002T130419')]},
    '### 季度交付 (source: Benzinga, created_at: 2026-10-02T13:04:19Z)\n交付增长\nLink: https://example.test/delivery'])
def test_formats_have_publication_time(response):
    rows=articles_from_response(response)
    assert len(rows)==1 and rows[0]['title']=='季度交付'
    assert rows[0]['published_at']


# ---- 经济数据补抓：数据源收录延迟时按标题去重，而非按发布时间 ----
from tradingagents.graph.late_news import refresh_macro, render_late_macro


def release(title='USA Nonfarm Payrolls For Sept. 29K Vs 89K Est.', created='2026-10-02T12:30:12Z'):
    return {'title':title,'name':'Nonfarm Payrolls For Sept.','actual':'29K','estimate':'89K','prior':None,'created_at':created}


def test_macro_published_before_previous_fetch_is_still_added(monkeypatch):
    from tradingagents.agents.tools import get_news
    monkeypatch.setattr(type(get_news),'invoke',lambda *a,**k:{'feed':[]})
    # 发布时间 08:30:12 早于初始取数 08:31:02，但当时数据源尚未收录，补抓时必须纳入
    with run_config(config(_late_macro_refresher=lambda:[release()], _late_macro_seen=[])):
        update=refresh_macro(state(),'research')
    assert [row['actual'] for row in update['late_macro']] == ['29K']
    assert update['late_macro'][0]['stage'] == 'research'
    assert update['late_macro'][0]['fetched_at'] == NOW.isoformat()


def test_macro_dedup_against_initial_context_and_previous_stage():
    seen_title='USA Unemployment Rate For September 4.2% Vs 4.1% Est.; 4.1% Prior'
    first={**release(),'stage':'research','fetched_at':NOW.isoformat()}
    rows=[release(), release(seen_title), release('USA Nonfarm Payrolls For Sept. Revises Prior From 162K To 133K')]
    with run_config(config(_late_macro_refresher=lambda:rows, _late_macro_seen=[seen_title])):
        update=refresh_macro({**state(),'late_macro':[first]},'portfolio')
    titles=[row['title'] for row in update['late_macro']]
    assert titles == [first['title'], 'USA Nonfarm Payrolls For Sept. Revises Prior From 162K To 133K']
    assert update['late_macro'][1]['stage'] == 'portfolio'


@pytest.mark.parametrize('overrides',[{'late_news_refresh':False},{'news_cutoff_utc':'2026-10-02T12:31:00Z'},{}])
def test_macro_disabled_replay_or_no_refresher_never_calls(overrides):
    refresher=None if not overrides else (lambda: pytest.fail('不应补抓经济数据'))
    with run_config(config(**overrides, _late_macro_refresher=refresher)):
        assert refresh_macro(state(),'research') == {}


def test_macro_failure_recorded_without_blocking_node(monkeypatch):
    from tradingagents.agents.tools import get_news
    monkeypatch.setattr(type(get_news),'invoke',lambda *a,**k:{'feed':[]})
    def fail(): raise TimeoutError('含凭据的错误正文不应泄露')
    wrapped=with_news_refresh(lambda s:{'investment_plan':'仍完成'},'research')
    with run_config(config(_late_macro_refresher=fail)): result=wrapped(state(),{})
    assert result['investment_plan'] == '仍完成'
    assert result['late_macro_errors'] == ['research阶段补抓经济数据失败：TimeoutError']
    assert '凭据' not in render_late_news(result) and '数据限制' in render_late_news(result)


def test_macro_rendered_for_downstream_and_portfolio_instruction(monkeypatch):
    from tradingagents.agents.tools import get_news
    monkeypatch.setattr(type(get_news),'invoke',lambda *a,**k:{'feed':[]})
    with run_config(config(_late_macro_refresher=lambda:[release()])):
        research=with_news_refresh(lambda s:{'investment_plan':render_late_news(s)},'research')(state(),{})
    assert '实际 29K，预期 89K' in research['investment_plan'] and '研究阶段补抓' in research['investment_plan']
    assert '建议重跑' not in research['investment_plan']
    portfolio_only=render_late_macro({'late_macro':[{**release(),'stage':'portfolio'}]})
    assert '上游未评估' in portfolio_only and '建议重跑' in portfolio_only
