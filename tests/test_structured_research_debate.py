"""固定桩在真实LangGraph验证并行屏障、可见性和旧路径，不代表真实投资效果。"""
import hashlib
import threading
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.schemas import ResearchPlan, ResearchCrux, render_research_plan
from tradingagents.dataflows.config import run_config
from tradingagents.graph import setup as graph_setup_module
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.propagation import Propagator
from tests.test_research_report_evidence import fixed_state


class ParallelModel:
    def __init__(self):
        self.barriers = {phase: threading.Barrier(2) for phase in ['opening', 'rebuttal']}
        self.prompts = {}
        self.lock = threading.Lock()

    def invoke(self, prompt):
        side = 'bull' if prompt.startswith('你是多头') else 'bear'
        phase = 'opening' if '本次只完成首轮' in prompt else 'rebuttal'
        with self.lock:
            self.prompts[(side, phase)] = prompt
        self.barriers[phase].wait(timeout=5)
        return SimpleNamespace(content=f'{side}_{phase}_固定输出')


@pytest.mark.parametrize('symbol', ['COHR', 'NVDA', 'QQQ', 'SMTC', 'TSLA'])
@pytest.mark.parametrize('mode', ['structured', 'legacy'])
def test_actual_graph_two_parallel_stages_and_single_stable_join(monkeypatch, symbol, mode):
    class LegacyModel:
        def __init__(self):
            self.prompts = []
        def invoke(self, prompt):
            self.prompts.append(prompt)
            return SimpleNamespace(content='旧首轮固定输出')
    llm = ParallelModel() if mode == 'structured' else LegacyModel()
    for factory, field in [('create_market_analyst', 'market_report'), ('create_sentiment_analyst', 'sentiment_report'),
                           ('create_news_analyst', 'news_report'), ('create_fundamentals_analyst', 'fundamentals_report')]:
        monkeypatch.setattr(graph_setup_module, factory, lambda model, config=None, key=field: lambda state: {'messages': [AIMessage('完成')], key: key + ':完整报告'})
    seen = []
    def manager(state):
        seen.append(dict(state))
        return {'investment_plan': '固定研究计划'}
    monkeypatch.setattr(graph_setup_module, 'create_research_manager', lambda *args: manager)
    monkeypatch.setattr(graph_setup_module, 'create_trader', lambda *args: lambda state: {'trader_investment_plan': '固定交易计划'})
    for factory, speaker in [('create_aggressive_debator', 'Aggressive'), ('create_conservative_debator', 'Conservative'), ('create_neutral_debator', 'Neutral')]:
        monkeypatch.setattr(graph_setup_module, factory, lambda model, config=None, label=speaker: lambda state: {'risk_debate_state': {**state['risk_debate_state'], 'count': state['risk_debate_state']['count'] + 1, 'latest_speaker': label}})
    monkeypatch.setattr(graph_setup_module, 'create_portfolio_manager', lambda *args: lambda state: {'final_trade_decision': '固定最终计划'})
    import tradingagents.graph.late_news as late
    monkeypatch.setattr(late, 'refresh_news', lambda *args: {})
    monkeypatch.setattr(late, 'refresh_macro', lambda *args: {})
    graph = graph_setup_module.GraphSetup(llm, llm, ConditionalLogic(), 2, {'debate_mode': mode}).setup_graph().compile()
    state = Propagator().create_initial_state(symbol, '2026-10-02', instrument_context='上下文')
    state['investment_debate_state']['history'] = '隐藏旧辩论'
    with run_config({'output_language': 'Chinese'}):
        result = graph.invoke(state)
    if mode == 'legacy':
        assert len(llm.prompts) == 2
        assert result['investment_debate_state']['count'] == 2
        assert result['risk_debate_state']['count'] == 3
        assert '隐藏旧辩论' in llm.prompts[0]
        assert '旧首轮固定输出' in llm.prompts[1]
        return
    assert len(llm.prompts) == 4
    assert len(seen) == 1
    for side in ['bull', 'bear']:
        opponent = 'bear' if side == 'bull' else 'bull'
        opening = llm.prompts[side, 'opening']
        rebuttal = llm.prompts[side, 'rebuttal']
        assert '隐藏旧辩论' not in opening
        assert '固定输出' not in opening
        assert f'{opponent}_opening_固定输出' in rebuttal
        assert f'{side}_opening_固定输出' not in rebuttal
        assert '对方首轮' not in opening
        assert '论证强度1–5' not in opening
        for field in ['market_report', 'fundamentals_report', 'sentiment_report', 'news_report']:
            assert field + ':完整报告' in opening
            assert field + ':完整报告' in rebuttal
    history = result['investment_debate_state']['history']
    ordered = ['bull_opening_固定输出', 'bear_opening_固定输出', 'bull_rebuttal_固定输出', 'bear_rebuttal_固定输出']
    assert [history.index(item) for item in ordered] == sorted(history.index(item) for item in ordered)
    assert result['investment_debate_state']['count'] == 4
    assert seen[0]['investment_debate_state']['history'] == history
    assert result['risk_debate_state']['count'] == 3
    assert result['final_trade_decision'] == '固定最终计划'


@pytest.mark.parametrize('factory, digest', [(create_bull_researcher, '62a364255d852418605a914509d97556b999f961265a2dfe17a31391fcfe8bc2'),
    (create_bear_researcher, '3c57e615287de40efd82c5088ada42a8582907abcb1e4a4090d1b2029dcc290d')])
def test_legacy_prompt_original_golden(factory, digest):
    class Capture:
        def invoke(self, prompt):
            self.prompt = prompt
            return SimpleNamespace(content='旧路径')
    llm = Capture()
    with run_config({'output_language': 'Chinese', 'allocation_bands': None, 'rating_probability_fields': False, 'sentiment_min_social_posts': 0}):
        factory(llm)(fixed_state())
    assert hashlib.sha256(llm.prompt.encode()).hexdigest() == digest


def test_cruxes_render_before_rating_and_old_record_compatibility():
    crux = ResearchCrux(bull_claim='多方1', bear_claim='空方1', evidence='报告数据1', winner='未决', reason='不足')
    plan = ResearchPlan(recommendation='Hold', rationale='理由', strategic_actions='等待', cruxes=[crux] * 3, evidence_check='无；核对数据1、2、3')
    text = render_research_plan(plan)
    assert text.index('分歧点') < text.index('Recommendation')
    assert '报告数据1' in text
    old = ResearchPlan(recommendation='Hold', rationale='旧理由', strategic_actions='旧动作')
    assert '未提供；不能视为已裁决' in render_research_plan(old)
    from tradingagents.agents.rating import output_flags
    short = ResearchPlan(recommendation='Hold', rationale='理由', strategic_actions='动作', cruxes=[crux] * 2)
    assert len(short.cruxes) == 2
    assert output_flags(short.model_dump(), {}, layer='rm')['cruxes_count'] == 2
