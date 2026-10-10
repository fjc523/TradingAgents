"""T51 冻结截止：保留时区，日期标签与未知时刻保守可见。"""
import json
from datetime import datetime
import pytest
from tradingagents.memory.log import TradingMemoryLog
from tradingagents.graph.trading_graph import TradingAgentsGraph


@pytest.mark.parametrize('known,expected', [
    ('2026-10-08T12:30:00+00:00', True),
    ('2026-10-08T08:31:00-04:00', True),
    ('2026-10-08T23:37:35-04:00', False),
    ('2026-10-09T00:30:00+14:00', True),
    ('2026-10-07', True), ('2026-10-08', False),
    (None, False), ('', False), ('未知', False),
    ('2026-10-07T01:00:00', False),
])
def test_timestamp_visibility(known, expected):
    assert TradingMemoryLog._known_by(known, '2026-10-08T08:31:00-04:00') is expected


def test_explicit_cutoff_overrides_same_day_historical_test(monkeypatch):
    import tradingagents.graph.trading_graph as graph_module
    monkeypatch.setattr(graph_module, 'is_historical', lambda *_: False)
    graph = object.__new__(TradingAgentsGraph)
    cutoff = datetime.fromisoformat('2026-10-08T08:31:00-04:00')
    assert graph._memory_as_of('2026-10-08', cutoff) == cutoff.isoformat()
    assert graph._memory_as_of('2026-10-08') is None


@pytest.mark.parametrize('lessons', ['stats', 'text'])
def test_legacy_date_and_c2_cutoff_without_writing_original(tmp_path, lessons):
    memory = TradingMemoryLog({'memory_log_path': str(tmp_path/'memory.md'),
        'lesson_min_settled_same_ticker': 0 if lessons == 'text' else 10,
        'cross_ticker_lessons': lessons})
    memory.store_decision('SPY', '2026-10-01', 'Rating: Hold\n固定决策')
    memory.update_with_outcome('SPY', '2026-10-01', .01, .01, 5, '当日才知反思', resolution_date='2026-10-08')
    before = memory._log_path.read_bytes()
    assert memory.get_past_context('SPY', as_of='2026-10-08T08:31:00-04:00') == ''
    later_context = memory.get_past_context('SPY', as_of='2026-10-09T08:30:00-04:00')
    assert later_context
    if lessons == 'text':
        assert '当日才知反思' in later_context
    memory._outcomes_path = tmp_path/'outcomes.jsonl'
    row = dict(is_current=True, trade_date='2026-10-01', symbol='SPY', ratings={'pm': 'Hold'}, primary_metric='raw_return', windows={
        '5': dict(status='settled', settled_at='2026-10-08T23:37:35-04:00', primary_return=.02),
        '10': dict(status='settled', settled_at=None, primary_return=.03)})
    memory._outcomes_path.write_text(json.dumps(row)+'\n')
    ledger_before = memory._outcomes_path.read_bytes()
    assert memory._visible_facts('2026-10-08T08:31:00-04:00') == []
    facts = memory._visible_facts('2026-10-09T08:30:00-04:00')
    assert facts[0]['returns'] == {'5': .02}
    assert memory._log_path.read_bytes() == before
    assert memory._outcomes_path.read_bytes() == ledger_before


@pytest.mark.parametrize('known,expected', [
    ('2026-10-08', True), ('2026-10-09', False),
    ('2026-10-08T23:59:59.999999-04:00', True),
    ('2026-10-09T00:00:00-04:00', False),
    ('2026-10-09T04:00:00+00:00', False),
])
def test_date_cutoff_preserves_previous_day_boundary(known, expected):
    assert TradingMemoryLog._known_by(known, '2026-10-08') is expected
