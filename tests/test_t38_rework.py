"""T38实际提示单份长规则、金额格式与成对括号的离线回归。"""
import json
import pytest
from tradingagents.agents.price_plan_legs import soft_direction,render_legs,BuyLeg
from tradingagents.agents.schemas import LegTraderProposal
from tests.test_direction_timing import Capture,FACTORIES,fixed_state


@pytest.mark.parametrize('text',['是（证据）','是(证据)','是：（证据）','是 (证据)'])
def test_o3_paired_parentheses(text):
    assert soft_direction(text)=='是：证据'


def test_o3_internal_or_unpaired_parentheses_keep_evidence():
    assert soft_direction('是（新增公告（来源X））')=='是：新增公告（来源X）'
    assert soft_direction('是（证据')=='是：（证据'
    assert soft_direction('是（ ）')=='是（ ）'
    assert soft_direction('是')=='是'


def test_o1_render_prices_two_decimal_but_days_and_allocation_keep_units():
    model=LegTraderProposal(action='Buy',reasoning='依据',direction_change='否',buy_legs=[{
        'status':'待触发','kind':'突破','trigger_rule':'收盘站上','trigger_price':12345.67,'confirm_days':2,
        'zone_low':12345.67,'zone_high':12345.678,'stop_loss':12300,'first_target':12500}],
        reduce_legs=[{'kind':'风险减配','trigger_price':12300,'post_allocation_pct':60}])
    text=render_legs(model)
    assert '12345.67（连续2日）' in text and '12345.67–12345.68' in text
    assert '止损12300.00' in text and '目标12500.00' in text and '减至60%' in text
    assert model.buy_legs[0].zone_high==12345.678


@pytest.mark.parametrize('role',['trader','pm'])
@pytest.mark.parametrize('lock',[False,True])
@pytest.mark.parametrize('dec',[False,True])
@pytest.mark.parametrize('tolerance',[0,7.5,10])
def test_o4_o5_actual_prompt_single_rules_all_constraints_schema_reference(role,lock,dec,tolerance):
    llm=Capture();FACTORIES[role](llm,{'allocation_tolerance_pct':tolerance,'risk_layer_direction_lock':lock,
        'rating_timing_decoupled':dec})(fixed_state())
    prompt='\n'.join(m['content'] for m in llm.prompt) if isinstance(llm.prompt,list) else llm.prompt
    schema=json.dumps(llm.schema.model_json_schema(),ensure_ascii=False)
    for phrase in ('建仓=无仓','不得为了与建仓不同而另造价位','目标内持仓不得仅因阻力受阻减仓',
        '不得强制同时给回踩和突破','同价加仓','风险减配必须具名失效位','与目标相差' if tolerance else '容差关闭'):
        assert prompt.count(phrase)==1
    assert '等待回踩至' not in prompt
    for field in ('entry_plan','add_plan','reduce_plan'):
        description=llm.schema.model_json_schema()['properties'][field]['description']
        assert len(description)<100 and '遵循提示词' in description and '建仓=无仓' not in description
    assert '建仓=无仓' not in schema
    expected=f'与目标相差<{tolerance:g}个百分点视为达标、不动' if tolerance else '容差关闭，按目标精确比较'
    assert expected in prompt and 'stop_anchor、target_anchor必须使用输入锚点表的键名' in prompt
