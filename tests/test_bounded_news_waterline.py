"""预取、真实分析师工具节点和两次补抓之间的新闻水位回归。"""
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tradingagents.agents.tools import get_news
from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.vendors.alpaca import bounded_news, client, news
from tradingagents.graph.analyst_execution import AnalystNodeSpec
from tradingagents.graph.late_news import refresh_news
from tradingagents.graph.setup import _analyst_graph

T0 = datetime(2026, 10, 20, 12, 42, 3, tzinfo=timezone.utc)
T1 = T0 + timedelta(seconds=28)
T2 = T0 + timedelta(minutes=5)
T3 = T0 + timedelta(minutes=12)


def article(number, published):
    return {'id': number, 'headline': f'消息{number}', 'summary': '固定离线消息',
            'created_at': published.isoformat(), 'updated_at': published.isoformat(),
            'source': '离线来源', 'url': f'https://example.test/{number}'}


class Client:
    """只替换底层HTTP；按真实窗口筛选，记录调用及阶段单尝试参数。"""
    def __init__(self, articles, fail_research=False):
        self.articles = articles
        self.fail_research = fail_research
        self.calls = []

    def get_news(self, symbols, start, end, **kwargs):
        self.calls.append((start, end, kwargs))
        if self.fail_research and (start, end) == (T0, T2):
            raise ConnectionError('研究补抓失败')
        rows = [row for row in self.articles
                if start <= datetime.fromisoformat(row['created_at']) < end]
        return {'articles': rows[:50], 'truncated': len(rows) > 50,
                'next_page_token': '后续页' if len(rows) > 50 else None}


def analyst(state):
    """脚本化分析师只调用真实get_news工具，不调用模型。"""
    if isinstance(state['messages'][-1], ToolMessage):
        return {'messages': [AIMessage(content='新闻分析完成')],
                'news_report': state['messages'][-1].content}
    return {'messages': [AIMessage(content='', tool_calls=[{
        'name': 'get_news', 'id': '新闻工具调用',
        'args': {'start_date': '2026-10-13', 'end_date': '2026-10-20'},
    }])]}


def run_until_cached_tool(monkeypatch, articles, *, fail_research=False, policy='bounded-stock-news-v1'):
    fake = Client(articles, fail_research)
    monkeypatch.setattr(client, '_SINGLETON', fake)
    clock = [T0]
    settings = {'stock_news_policy': policy, '_analysis_market': 'US', '_analysis_type': 'stock',
                '_daily_run_id': str(uuid4()), '_analysis_symbol': 'TSLA',
                'late_news_refresh': True, 'news_cutoff_utc': None,
                '_late_news_clock': lambda: clock[0], 'data_vendors': {'news_data': 'alpaca'},
                'tool_vendors': {}, 'market_timezone': 'America/New_York'}
    state = {'company_of_interest': 'TSLA', 'trade_date': '2026-10-20',
             'messages': [HumanMessage(content='分析新闻')], 'late_news': [], 'late_news_errors': []}
    with run_config(settings):
        news.get_news('TSLA', '2026-10-13', '2026-10-20')
        initial_calls = len(fake.calls)
        clock[0] = T1
        graph = _analyst_graph(AnalystNodeSpec('news', 'News Analyst', 'news_report', (get_news,)), analyst, 2)
        state.update(graph.invoke(state))
    assert '固定离线消息' in state['news_report']
    return settings, clock, state, fake, initial_calls


@pytest.mark.parametrize('fail_research', [False, True])
def test_prefetch_cached_tool_and_real_refresh_keep_contiguous_window(monkeypatch, fail_research):
    gap = article(9001, T0 + timedelta(seconds=10))
    articles = [article(100, T0 - timedelta(days=1)), gap,
                article(9002, T2 - timedelta(minutes=1)), article(9003, T3 - timedelta(minutes=1))]
    settings, clock, state, fake, initial_calls = run_until_cached_tool(
        monkeypatch, articles, fail_research=fail_research)
    assert initial_calls == len(fake.calls) == 3
    assert state['news_last_fetched_at'] == T0.isoformat()
    with run_config(settings):
        clock[0] = T2
        research = refresh_news(state, 'research')
        state.update(research)
        if fail_research:
            assert 'news_last_fetched_at' not in research
            assert state['news_last_fetched_at'] == T0.isoformat()
        else:
            assert any(row['title'] == '消息9001' and row['stage'] == 'research' for row in state['late_news'])
        clock[0] = T3
        state.update(refresh_news(state, 'portfolio'))
    audit = bounded_news.audit_for(settings['_daily_run_id'], 'TSLA')
    stages = audit['stages']
    assert stages['research']['effective_start'] == stages['initial']['effective_end'] == T0.isoformat()
    assert fake.calls[3][:2] == (T0, T2)
    assert fake.calls[4][:2] == (T0 if fail_research else T2, T3)
    assert audit['http_attempts'] == len(fake.calls) == 5
    assert audit['cache_hits'] == 1
    assert all(call[2]['max_pages'] == 1 and call[2]['single_attempt'] is True for call in fake.calls)
    assert any(row['id'] == 9001 for row in stages['portfolio' if fail_research else 'research']['selected_articles'])
    assert {'消息9001', '消息9002', '消息9003'} <= {row['title'] for row in state['late_news']}
    assert stages['research']['coverage_status'] == ('failed' if fail_research else 'complete_within_provider')
    assert bool(state['late_news_errors']) is fail_research
    assert state['news_last_fetched_at'] == T3.isoformat()


def test_cached_tool_keeps_shared_late_article_budget(monkeypatch):
    articles = [article(100, T0 - timedelta(days=1))]
    articles += [article(200 + i, T0 + timedelta(seconds=10 + i)) for i in range(7)]
    articles += [article(300 + i, T2 + timedelta(seconds=10 + i)) for i in range(7)]
    settings, clock, state, fake, _ = run_until_cached_tool(monkeypatch, articles)
    with run_config(settings):
        for stage, now in [('research', T2), ('portfolio', T3)]:
            clock[0] = now
            state.update(refresh_news(state, stage))
    audit = bounded_news.audit_for(settings['_daily_run_id'], 'TSLA')
    assert audit['stages']['research']['selected_count'] == 7
    assert audit['stages']['portfolio']['selected_count'] == 3
    assert audit['stages']['portfolio']['budget_dropped'] == 4
    assert len(state['late_news']) == 10
    assert len(fake.calls) == audit['http_attempts'] == 5


def test_legacy_analyst_waterline_still_uses_tool_call_time(monkeypatch):
    settings, _, state, fake, initial_calls = run_until_cached_tool(
        monkeypatch, [article(100, T0 - timedelta(days=1))], policy='legacy')
    assert initial_calls == 1 and len(fake.calls) == 2
    assert state['news_last_fetched_at'] == T1.isoformat()
    assert bounded_news.audit_for(settings['_daily_run_id'], 'TSLA') is None
