"""执行腿软校验、实际生成顺序和新旧提示分支；不调用模型。"""
import pytest
from tradingagents.agents.schemas import (LegTraderProposal, LegPortfolioDecision, ResearchPlan,
    CruxResearchPlan, EvidenceResearchPlan, LegacyResearchPlan, decision_schema, render_pm_decision,
    price_plan_instruction, render_trader_proposal, load_trader_proposal)


@pytest.mark.parametrize('schema,base', [(LegTraderProposal, {'action':'Buy','reasoning':'证据'}),
    (LegPortfolioDecision, {'rating':'Buy','executive_summary':'摘要','investment_thesis':'证据'})])
def test_bad_legs_preserve_decision(schema, base):
    model = schema(**base, direction_change='是。新消息改变需求', buy_legs=[
        {'kind':'pullback','status':'executable','zone_low':'105','zone_high':'100',
         'stop_loss':'NaN','reason':{'错误':'类型'},'confirm_days':'3'},
        {'kind':['错'],'trigger_rule':False}, {}, '错误腿'], reduce_legs='错误列表')
    assert model.direction_change == '是：新消息改变需求'
    assert model.buy_legs[0].kind == '回踩' and model.buy_legs[0].zone_low == 105
    assert model.buy_legs[0].stop_loss is None and model.buy_legs[0].reason is None
    assert model.buy_legs[0].confirm_days is None and len(model.buy_legs) == 3
    assert model.reduce_legs is None
    assert 'buy_legs:too_many_legs' in model.leg_validation_flags
    assert 'buy_legs.0:zone_inverted' in model.leg_validation_flags
    assert 'leg_validation_flags' not in schema.model_json_schema()['properties']


@pytest.mark.parametrize('value', ['是','是：', '是。。。', '', None, {'错':'类型'}])
def test_direction_invalid_is_retained_and_flagged(value):
    model=LegTraderProposal(action='Hold', reasoning='观察', direction_change=value)
    assert any(flag.startswith('direction_change:') for flag in model.leg_validation_flags)


@pytest.mark.parametrize('base', [ResearchPlan, CruxResearchPlan, EvidenceResearchPlan, LegacyResearchPlan])
def test_rm_has_no_legs_and_probability_before_rating(base):
    schema=decision_schema(base, {}, 'rm')
    props=list(schema.model_json_schema()['properties'])
    assert 'buy_legs' not in props and 'leg_validation_flags' not in schema.model_fields
    assert props[props.index('recommendation')-3:props.index('recommendation')]==[
        'prob_outperform_5d','prob_outperform_20d','expected_return_20d_range']
    model=schema(recommendation='Overweight',rationale='依据',strategic_actions='观察',prob_outperform_20d='62%')
    assert model.prob_outperform_20d == .62


def test_new_prompt_removes_dual_template_and_named_r36():
    text=price_plan_instruction({})
    assert '等待回踩至 X 或突破 Y 确认' not in text
    assert '每项首句固定' not in text and '区间上沿+2×ATR' not in text
    for phrase in ('U+0.05ATR','U+3ATR','具名支撑','仅观察','不得强制同时给回踩和突破'):
        assert phrase in text
    old=price_plan_instruction({'price_plan_legs':False,'price_plan_target_rule':'d1'})
    assert '区间上沿+2×ATR' in old and '等待回踩至 X 或突破 Y 确认' in old
    assert 'R36' not in old and 'buy_legs' not in old


def test_render_execution_after_reduction_only_for_nonempty_legs():
    old=load_trader_proposal({'action':'Hold','reasoning':'旧记录'})
    assert '**执行条件**' not in render_trader_proposal(old)
    new=LegTraderProposal(action='Overweight',reasoning='证据',direction_change='否',
        buy_legs=[{'kind':'突破','status':'待触发','trigger_rule':'收盘站上','trigger_price':201.99,'confirm_days':1,
                   'zone_low':201.99,'zone_high':204.9,'stop_loss':190.33,'first_target':239.87}])
    text=render_trader_proposal(new)
    assert text.index('**减仓点位**') < text.index('**执行条件**') < text.index('**目标配置')
    assert '收盘站上201.99（1日）后' in text


@pytest.mark.parametrize('layer', ['trader', 'pm'])
def test_direction_disabled_legs_loader_roundtrip(layer):
    # R37-W12-1：公开读取入口不能把合法无声明记录变成缺必填字段错误。
    from tradingagents.agents.schemas import (LegacyTraderProposal, LegacyPortfolioDecision,
        load_portfolio_decision)
    base=LegacyTraderProposal if layer=='trader' else LegacyPortfolioDecision
    loader=load_trader_proposal if layer=='trader' else load_portfolio_decision
    payload={'action':'Hold','reasoning':'观察'} if layer=='trader' else {
        'rating':'Hold','executive_summary':'观察','investment_thesis':'依据'}
    schema=decision_schema(base, {'price_plan_legs':True,
        'risk_layer_direction_lock':False,'rating_timing_decoupled':False}, layer)
    model=schema(**payload,buy_legs=[{'kind':'回踩','status':'仅观察','zone_low':'99'}])
    assert 'direction_change' not in type(model).model_fields
    loaded=loader(model.model_dump(mode='json'))
    assert loaded.direction_change is None and loaded.buy_legs[0].zone_low==99
    assert loader(loaded.model_dump(mode='json')).direction_change is None
    assert '未提供；旧记录未声明' in (render_trader_proposal(loaded) if layer=='trader'
        else render_pm_decision(loaded))


@pytest.mark.parametrize('reads,mode', [(False,'legacy'),(True,'legacy'),(False,'structured'),(True,'structured')])
def test_reordered_rm_retains_detailed_semantics(reads,mode):
    # R37-W12-4：只重排，不缩减原证据、反驳理由与复评要求。
    from tests.test_research_report_evidence import Capture, fixed_state
    from tradingagents.agents.managers.research_manager import create_research_manager
    llm=Capture()
    create_research_manager(llm,{'research_manager_reads_reports':reads,'debate_mode':mode})(fixed_state())
    for text in ('决定性论据前3条','被驳回论据及理由','复评触发条件'):
        assert text in llm.prompt
    assert llm.prompt.index('- **概率**') < llm.prompt.index('- **Recommendation**') < llm.prompt.index('- **Rationale**')
    if reads:
        for text in ('辩手引用不符处','双方都遗漏但影响结论的事实','无则写“无”并列已核对的2–3个数据点'):
            assert text in llm.prompt
        assert llm.prompt.index('- **引用核对**') < llm.prompt.index('- **概率**')
        assert '冲突本身不构成Hold理由；独立比较证据，选择证据占优方，仅均衡或不足时Hold，不受先后发言影响。' in llm.prompt
    else:
        assert '配置以单标的标准仓位100%为参考单位' in llm.prompt
