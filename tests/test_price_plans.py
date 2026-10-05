"""核对价格方案结构、缺失值兼容和报告渲染，不查询真实行情。"""

import pytest
from pydantic import ValidationError

from tradingagents.agents.schemas import (
    PortfolioDecision, TraderProposal, ResearchPlan, PRICE_PLAN_INSTRUCTION,
    render_pm_decision, render_trader_proposal, render_research_plan, load_portfolio_decision,
)


def test_final_decision_contains_reference_entry_and_add_conditions():
    decision = PortfolioDecision(direction_change="否",
        rating="Buy", executive_summary="等待回踩确认", investment_thesis="依据日线支撑",
        reference_price="100 USD，2026-10-01 收盘，非实时报价",
        entry_plan="98–100 USD；回踩企稳后建仓；跌破97失效；依据日线支撑",
        add_plan="突破102后回踩101–102 USD；放量确认；跌回100失效；依据压力位",
        reduce_plan="跌破97 USD减仓；依据支撑失效", target_allocation_pct=60,
    )
    text = render_pm_decision(decision)
    assert all(value in text for value in (decision.reference_price, decision.entry_plan, decision.add_plan))
    assert all(label in text for label in ("参考价格与时点", "建仓点位", "加仓点位", "减仓点位"))
    assert "**目标配置（标准仓位=100%）**: 60%" in text
    proposal = TraderProposal(direction_change="否", action="Buy", reasoning="有条件建仓", **decision.model_dump(include={"reference_price", "entry_plan", "add_plan", "reduce_plan", "target_allocation_pct"}))
    assert decision.entry_plan in render_trader_proposal(proposal)
    assert decision.reduce_plan in render_trader_proposal(proposal)


def test_legacy_decision_remains_valid_without_price_plans():
    decision = load_portfolio_decision(dict(rating="Hold", executive_summary="观察", investment_thesis="证据不足"))
    assert decision.reference_price is None and decision.entry_plan is None and decision.add_plan is None
    assert "等待可靠行情" in render_pm_decision(decision)
    assert "不得编造报价" in PRICE_PLAN_INSTRUCTION and "未核验时段" in PRICE_PLAN_INSTRUCTION
    assert "**目标配置（标准仓位=100%）**: 未提供" in render_pm_decision(decision)


def test_zero_allocation_and_research_handoff_use_same_unit():
    plan = ResearchPlan(recommendation="Sell", rationale="风险过高", strategic_actions="退出", target_allocation_pct=0)
    assert "**目标配置（标准仓位=100%）**: 0%" in render_research_plan(plan)
    with pytest.raises(ValidationError):
        ResearchPlan.model_validate({**plan.model_dump(), "target_allocation_pct": -1})


def test_price_schema_and_prompt_share_fixed_first_sentence():
    """三类方案首句格式在提示和实际schema中一致。"""
    assert "区间 X–Y 美元（依据：" in PRICE_PLAN_INSTRUCTION
    from tradingagents.agents.schemas import price_plan_instruction
    assert '不适用：原因' in price_plan_instruction({'price_plan_legs':False,'price_plan_target_rule':'d1'})
    for schema in (TraderProposal, PortfolioDecision):
        for name in ("entry_plan", "add_plan", "reduce_plan"):
            desc = schema.model_fields[name].description
            assert "区间 X–Y 美元（依据：" in desc and "不适用：原因" in desc


def test_price_rules_use_runtime_config_without_changing_compatibility_constant():
    from tradingagents.agents.schemas import price_plan_instruction
    from tradingagents.dataflows.config import set_config
    from tradingagents.default_config import DEFAULT_CONFIG
    assert price_plan_instruction(DEFAULT_CONFIG) == PRICE_PLAN_INSTRUCTION
    set_config({"price_plan_legs":False,"price_plan_target_rule":"d1","price_plan_stop_atr_min": 1.2, "price_plan_stop_atr_normal": [1.6, 2.1],
                "price_plan_stop_atr_max": 2.6, "price_plan_min_reward_risk": 2})
    text = price_plan_instruction()
    assert "1.2–2.6倍ATR" in text and "1.6–2.1倍" in text
    assert "建仓/加仓要求≥2" in text and "区间上沿" in text
    assert "1–2.5ATR" in PRICE_PLAN_INSTRUCTION
