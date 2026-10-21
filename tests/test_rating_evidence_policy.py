"""新政策实际四角色提示与并存schema缓存隔离，不执行模型。"""
from copy import deepcopy
import importlib
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.rating import RATING_EVIDENCE_INSTRUCTION, rating_definitions
from tradingagents.agents.schemas import PortfolioDecision, decision_schema


class Captured(Exception):
    """在真实节点构造提示后、调用模型前退出。"""


def prompt_for(role, config, state=None):
    module_name, factory_name = {
        'rm': ('managers.research_manager', 'create_research_manager'),
        'trader': ('trader.trader', 'create_trader'),
        'pm': ('managers.portfolio_manager', 'create_portfolio_manager'),
    }[role]
    module = importlib.import_module('tradingagents.agents.' + module_name)
    from unittest.mock import patch
    captured = []
    def take(*args, **kwargs):
        captured.append(args[1] if role == 'pm' else args[2])
        raise Captured()
    sample = dict(company_of_interest='TSLA', trade_date='2026-10-21', asset_type='stock',
                  instrument_context='市场上下文\n标签来源：冻结来源', past_context='原历史',
                  recent_decisions_context='冻结最近三次表', market_report='市场报告',
                  fundamentals_report='财报', news_report='新闻', sentiment_report='',
                  investment_plan='研究经理计划', trader_investment_plan='交易员方案',
                  investment_debate_state={'history': '', 'count': 0}, risk_debate_state={'history': ''})
    sample.update(state or {})
    with patch.object(module, 'bind_structured', return_value=object()), patch.object(module, 'invoke_structured' if role == 'pm' else 'invoke_decision', take):
        node = getattr(module, factory_name)(MagicMock(), config=config)
        with pytest.raises(Captured):
            node(sample)
    prompt = captured[0]
    return '\n'.join(m['content'] for m in prompt) if isinstance(prompt, list) else prompt


@pytest.mark.parametrize('role', ['rm', 'trader', 'pm'])
@pytest.mark.parametrize('reads_reports', [True, False])
@pytest.mark.parametrize('debate_mode', ['structured', 'legacy'])
@pytest.mark.parametrize('timing,legs', [(True, True), (False, False)])
def test_actual_prompts_policy_diff(role, reads_reports, debate_mode, timing, legs):
    config = dict(research_manager_reads_reports=reads_reports, debate_mode=debate_mode,
                  rating_timing_decoupled=timing, price_plan_legs=legs, risk_layer_direction_lock=True)
    before = prompt_for(role, {**config, 'rating_evidence_policy': 'legacy'})
    after = prompt_for(role, {**config, 'rating_evidence_policy': 'anti-chase-v1'})
    assert RATING_EVIDENCE_INSTRUCTION not in before
    assert after.count(RATING_EVIDENCE_INSTRUCTION) == 1
    if role == 'rm':
        assert '冻结最近三次表' in after and '冻结最近三次表' not in before
        assert '相对上次已发布决策：' in after and '既有rationale仍限600字' in after
    else:
        assert '冻结最近三次表' not in after
        assert '方向变更仅指相对该recommendation' in after
        assert '新增数据不含回看<20日的价格/技术/VIX变化' in after
    assert '标签来源：冻结来源' in after
    assert 'prob_outperform_20d' in after


@pytest.mark.parametrize('order', [('legacy', 'anti-chase-v1', 'legacy'), ('anti-chase-v1', 'legacy', 'anti-chase-v1')])
@pytest.mark.parametrize('market', [None, {'quote_currency': 'HKD', 'benchmark_symbol': '^HSI'}])
def test_schema_policy_cache_keeps_original_metadata_and_identity(order, market):
    original = deepcopy(PortfolioDecision.model_fields['direction_change'])
    models = [decision_schema(PortfolioDecision, {'rating_evidence_policy': policy, 'market_context': market}, 'pm') for policy in order]
    assert models[0] is models[2] and models[0] is not models[1]
    for policy, model in zip(order, models):
        description = model.model_fields['direction_change'].description
        assert ('方向变更仅指相对研究经理recommendation' in description) == (policy != 'legacy')
        assert model.model_fields['direction_change'] is not PortfolioDecision.model_fields['direction_change']
    assert PortfolioDecision.model_fields['direction_change'].description == original.description


def test_direction_lock_literal_preserves_original_trailing_space():
    import ast
    from pathlib import Path
    source = Path(importlib.import_module('tradingagents.agents.trader.trader').__file__).read_text()
    values = [node.value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert '输出五档动作，与研究经理不同须说明原因；观点有分歧本身不是Hold的理由。 ' in values
    assert '默认沿用研究经理recommendation。只有研究经理未考虑的可核对新证据才允许改变方向；不能凭盈亏比、入场点不足或笼统风险改变。 ' in values
    assert any(v.startswith('默认沿用研究经理recommendation，方向变更仅指相对该recommendation。') and v.endswith(' ') for v in values)


def test_shared_rule_keeps_execution_and_correction_exceptions():
    text = rating_definitions({'rating_evidence_policy': 'anti-chase-v1'})
    for part in ('<20个交易日', '20日概率每轮按当前证据独立评估', '不得照抄上次概率',
                 '未越档的概率调整不额外要求新事实', '点位腿（含风险减配）与止损规则',
                 '上次评级依据为短期信号，按合格证据纠正', 'MACD参数含26', '不能为维持旧评级忽略反证'):
        assert part in text
    with pytest.raises(ValueError, match='未知'):
        rating_definitions({'rating_evidence_policy': 'unknown'})
