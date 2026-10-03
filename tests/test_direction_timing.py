"""方向声明、执行时机和关闭原版本兼容；固定桩不代表模型遵守质量。"""
import hashlib
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from tradingagents.graph.propagation import Propagator
from tradingagents.dataflows.config import run_config
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.agents.schemas import (TraderProposal, PortfolioDecision, load_trader_proposal,
    load_portfolio_decision, render_trader_proposal, render_pm_decision, price_plan_instruction)

OFF = dict(risk_layer_direction_lock=False, rating_timing_decoupled=False, price_plan_alt_target=False)
BASELINES = {
    'trader': 'becac8b08a195d6def633e5694aa7185138d97de9c7aa72bcc078e2adb8b019b',
    'trader_schema': '7c2cffa549a2f63ff51d75a3a0cbd456996a0720b1812c6ea5be465b16033840',
    'pm': 'ae03513f54b5be6f869d3682d5b1961e9eaa58efba3312908aec67da51fc930c',
    'pm_schema': '2d7c82fe4d41d7bf53c19a3578a396a596580255a964375bd17ef3d7229e0d7b',
    'aggressive': 'a85e9fbc32c7fd18039c185ac99dde62655ac92012f6e8690c16244e42f365d1',
    'conservative': '8f8207696fdb465fe2635675a2f3d04e713b8c96c110f1fa100bc3fcf5320093',
    'neutral': '3fb2519ead26ccf0d7dd954e69b0e59dfeffe93e0d2ef6c084e1a9517708fedd',
}
FACTORIES = dict(trader=create_trader, pm=create_portfolio_manager, aggressive=create_aggressive_debator,
                conservative=create_conservative_debator, neutral=create_neutral_debator)


def fixed_state():
    state = Propagator().create_initial_state('SMTC', '2026-10-02', instrument_context='上下文', past_context='历史')
    state.update(market_report='市场完整报告', sentiment_report='情绪完整报告', news_report='新闻完整报告',
                 fundamentals_report='基本面完整报告', investment_plan='研究经理完整计划与分歧', trader_investment_plan='交易员方案')
    state['risk_debate_state'].update(history='审阅历史', current_aggressive_response='激进意见',
                                    current_conservative_response='保守意见', current_neutral_response='中立意见')
    return state


class Capture:
    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, prompt):
        self.prompt = prompt
        if not hasattr(self, 'schema'):
            return SimpleNamespace(content='固定审阅输出')
        fields = {'direction_change': '否'} if 'direction_change' in self.schema.model_fields else {}
        if 'action' in self.schema.model_fields:
            return self.schema(action='Hold', reasoning='固定理由', **fields)
        return self.schema(rating='Hold', executive_summary='固定摘要', investment_thesis='固定论点', **fields)


@pytest.mark.parametrize('role', FACTORIES)
def test_closed_prompt_and_schema_original_git_golden(role):
    # 独立从a897742旧Git对象执行原源码捕获，期望不是调用新函数产生。
    llm = Capture()
    with run_config({'output_language': 'Chinese'}):
        FACTORIES[role](llm, OFF)(fixed_state())
    text = json.dumps(llm.prompt, sort_keys=True) if isinstance(llm.prompt, list) else llm.prompt
    assert hashlib.sha256(text.encode()).hexdigest() == BASELINES[role]
    if hasattr(llm, 'schema'):
        assert 'direction_change' not in llm.schema.model_fields
        assert hashlib.sha256(json.dumps(llm.schema.model_json_schema(), sort_keys=True).encode()).hexdigest() == BASELINES[role + '_schema']


@pytest.mark.parametrize('role', ['aggressive', 'conservative', 'neutral'])
def test_risk_only_execution_new_evidence_required(role):
    llm = Capture()
    FACTORIES[role](llm, {})(fixed_state())
    assert '允许反对交易方向' not in llm.prompt
    assert '方案与评级一致性' not in llm.prompt
    assert '研究经理未考虑的可核对新证据' in llm.prompt
    assert '不得仅凭盈亏比或入场点不足' in llm.prompt
    assert '研究经理完整计划与分歧' in llm.prompt


@pytest.mark.parametrize('risk,timing,alt', [(bool(i & 1),bool(i & 2),bool(i & 4)) for i in range(8)])
def test_private_switch_combinations_and_pm_order(risk, timing, alt):
    config = dict(risk_layer_direction_lock=risk, rating_timing_decoupled=timing, price_plan_alt_target=alt)
    for role in ['trader', 'pm']:
        llm = Capture(); node = FACTORIES[role](llm, config)
        FACTORIES[role](Capture(), OFF)
        node(fixed_state())
        assert ('direction_change' in llm.schema.model_fields) == (risk or timing)
        text = json.dumps(llm.prompt, ensure_ascii=False) if isinstance(llm.prompt,list) else llm.prompt
        assert ('目标方法：ATR替代' in text) == alt
        assert ('评级只表达未来5–20' in text) == timing
        if role == 'pm' and (risk or timing):
            assert text.index('研究经理完整计划与分歧') < text.index('交易员方案') < text.index('审阅历史')
            assert '方案必须与评级一致' not in text


@pytest.mark.parametrize('schema,values', [(TraderProposal,dict(action='Overweight',reasoning='理由')),
    (PortfolioDecision,dict(rating='Overweight',executive_summary='摘要',investment_thesis='论点'))])
def test_direction_new_generation_required_and_format(schema, values):
    assert 'direction_change' in schema.model_json_schema()['required']
    for invalid in [None, '', '是', '是：', '不变']:
        with pytest.raises(ValidationError):
            schema(**values, direction_change=invalid)
    with pytest.raises(ValidationError):
        schema(**values)
    assert schema(**values,direction_change='否').direction_change == '否'
    assert schema(**values,direction_change='是：新补抓新闻，来源X，事实Y改变需求').direction_change.startswith('是：')


def test_old_records_compatibility_without_false_declaration():
    trader = load_trader_proposal(dict(action='Hold',reasoning='旧理由'))
    pm = load_portfolio_decision(dict(rating='Hold',executive_summary='旧摘要',investment_thesis='旧论点'))
    assert '未提供；旧记录未声明' in render_trader_proposal(trader)
    assert '未提供；旧记录未声明' in render_pm_decision(pm)
    assert trader.direction_change is None and pm.direction_change is None


def test_price_rule_conditions_and_thresholds_not_relaxed():
    old = price_plan_instruction(OFF)
    assert hashlib.sha256(old.encode()).hexdigest() == 'dd917fcaa60b98b9d0402768ab76ef80186260298304847d2356eaf3311bdaf0'
    active = price_plan_instruction({})
    for phrase in ['最近上方阻力不足1ATR', '60日新高附近且上方无阻力', '区间上沿+2×ATR', '目标方法：ATR替代',
                   '阻力上方0–0.25ATR', '收盘站上阻力', '阻力下方1–1.5ATR', '1–2.5ATR', '盈亏比≥1.5', '不满足仍写不适用']:
        assert phrase in active
    assert '52周' not in active
    assert '不能作为改评级理由' in active
    assert '目标方法：ATR替代' not in old


def test_active_rating_definitions_are_direction_only_and_pm_waiting_two_prices():
    from tradingagents.agents.rating import RATING_DEFINITIONS, LEGACY_RATING_DEFINITIONS
    for phrase in ['积极建仓或加仓', '逐步提高配置', '维持现有配置', '把配置降到目标水平', '清仓或不建仓']:
        assert phrase not in RATING_DEFINITIONS
        assert phrase in LEGACY_RATING_DEFINITIONS
    for role in ['trader','pm']:
        llm = Capture();FACTORIES[role](llm, {})(fixed_state())
        prompt = '\n'.join(message['content'] for message in llm.prompt) if isinstance(llm.prompt,list) else llm.prompt
        assert RATING_DEFINITIONS in prompt
        assert 'Buy/Overweight没有合格入场点时保持评级' in prompt
        assert '不适用：等待回踩至 X 或突破 Y 确认' in prompt
        assert '不编造' in prompt or '不能编造' in prompt
        for phrase in ['积极建仓或加仓', '逐步提高配置', '维持现有配置', '把配置降到目标水平', '清仓或不建仓']:
            assert phrase not in prompt
    assert '5–20交易日' in PortfolioDecision.model_fields['time_horizon'].description
    from tradingagents.agents.managers.research_manager import create_research_manager
    from tests.test_research_report_evidence import Capture as ResearchCapture
    state=fixed_state();state['investment_debate_state']['history']='辩论'
    llm=ResearchCapture();create_research_manager(llm, {})(state)
    assert RATING_DEFINITIONS in llm.prompt
    assert '积极建仓或加仓' not in llm.prompt
