"""未来美国事实与有界新闻的确定性离线回归。"""
from datetime import datetime, timedelta
import json
from uuid import uuid4
from zoneinfo import ZoneInfo

from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.vendors.alpaca import bounded_news
from tradingagents.memory.log import TradingMemoryLog

NOW = datetime(2026, 10, 20, 8, 41, tzinfo=ZoneInfo('America/New_York'))


def test_shared_budget_and_initial_subwindow():
    calls = []
    class Client:
        def get_news(self, symbols, start, end, **kwargs):
            calls.append((start, end))
            return {'articles': [{'id': str(len(calls)), 'headline': f'消息{len(calls)}',
                'created_at': (end - timedelta(minutes=1)).isoformat()}], 'truncated': False}
    config = {'_daily_run_id': str(uuid4()), '_analysis_symbol': 'A', '_late_news_clock': lambda: NOW}
    first = bounded_news.read('A', NOW - timedelta(days=3), NOW + timedelta(days=1), config, Client())
    assert '消息1' in first and len(calls) == 3
    subset = bounded_news.read('A', NOW - timedelta(hours=1), NOW + timedelta(days=1), config, Client())
    assert '消息1' not in subset and '消息3' in subset and len(calls) == 3


def test_research_failure_preserves_waterline(monkeypatch):
    from tradingagents.agents import tools
    from tradingagents.graph.late_news import refresh_news
    previous = NOW - timedelta(minutes=10)
    config = {'late_news_refresh': True, 'stock_news_policy': 'bounded-stock-news-v1', '_analysis_market': 'US',
        '_analysis_type': 'stock', '_daily_run_id': str(uuid4()), '_analysis_symbol': 'A', '_late_news_clock': lambda: NOW}
    class Client:
        def get_news(self, *args, **kwargs):
            raise RuntimeError('模拟传输失败')
    def invoke(args, config=None):
        return bounded_news.read('A', previous, NOW + timedelta(days=1), settings, Client())
    settings = config
    from types import SimpleNamespace
    monkeypatch.setattr(tools, 'get_news', SimpleNamespace(invoke=invoke))
    with run_config(config):
        result = refresh_news({'company_of_interest': 'A', 'trade_date': '2026-10-20', 'news_last_fetched_at': previous.isoformat()}, 'research')
    assert 'news_last_fetched_at' not in result
    assert result['late_news_errors']


def test_memory_suppresses_each_pre_w4_key(tmp_path):
    memory = tmp_path / 'memory.md'
    memory.write_text('[2026-10-12 | SPY | Hold | +1.0% | +1.0% | 5d | resolved:2026-10-18]\n\nDECISION:\n旧决定\n\n<!-- ENTRY_END -->\n\n')
    rows = []
    for day, rid in [('2026-10-12', 'old'), ('2026-10-13', 'winner')]:
        rows.append({'run_id': rid, 'symbol': 'SPY', 'trade_date': day, 'is_current': True, 'analysis_symbol': 'SPY',
            'finished_at': day + 'T08:42:00-04:00', 'price_data_end_date': '2026-10-09',
            'entry': {'date': '2026-10-13', 'basis': 'same_day_open'}, 'primary_metric': 'raw_return',
            'ratings': {'pm': 'Hold'}, 'windows': {'5': {'status': 'settled', 'settled_at': '2026-10-19T16:01:00-04:00', 'primary_return': .02}}})
    path = tmp_path / 'outcomes.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    log = TradingMemoryLog({'memory_log_path': str(memory), 'evaluation_outcomes_path': str(path), 'memory_fact_policy': 'w4-visible-v1'})
    result = log.get_past_context('SPY', as_of=NOW.isoformat())
    assert log.fact_audit['windows']['5']['n'] == 1
    assert log.fact_audit['windows']['5']['legacy_suppressed'] == ['2026-10-12|SPY']
    assert '旧memory 5日tag' not in result
    assert 'winner|SPY' in log.fact_audit['windows']['5']['selected_ids']
