"""第二轮概率生成顺序、关闭逐字兼容及pending来源保护。"""
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tradingagents.agents.schemas import ResearchPlan, CruxResearchPlan, EvidenceResearchPlan, LegacyResearchPlan, PortfolioDecision, decision_schema
from tradingagents.llm_clients.codex_exec.schemas import schema_for
from tradingagents.memory.log import TradingMemoryLog
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.graph.propagation import Propagator


@pytest.mark.parametrize('base,layer',[(ResearchPlan,'rm'),(CruxResearchPlan,'rm'),(EvidenceResearchPlan,'rm'),(LegacyResearchPlan,'rm'),(PortfolioDecision,'pm')])
def test_actual_generation_order(base,layer):
    schema=decision_schema(base,{},layer)
    for payload in [schema.model_json_schema(),schema_for(schema)]:
        order=list(payload['properties'])
        probability_fields=['prob_outperform_5d','prob_outperform_20d','expected_return_20d_range']
        rating='recommendation' if layer=='rm' else 'rating'
        assert order[order.index(rating)-3:order.index(rating)]==probability_fields
        for evidence in ('cruxes','evidence_check'):
            if evidence in order:assert order.index(evidence)<order.index(probability_fields[0])
    assert [name for name in schema.model_fields if name != 'leg_validation_flags']==list(schema.model_json_schema()['properties'])


# 修改前d868bdb的JSON全文SHA256，同其他开关逐字比较，不能仅比较字段集合。
@pytest.mark.parametrize('base,layer,prices,timing,digest',[
    (ResearchPlan,'rm',False,False,'f52e6fe10eaf0522029ff5f502d357fccafe141fe9f8af160504118f635a351c'),
    (ResearchPlan,'rm',True,True,'f52e6fe10eaf0522029ff5f502d357fccafe141fe9f8af160504118f635a351c'),
    (CruxResearchPlan,'rm',True,True,'1380a7b73837a44601ff56f10358114d29541b86543507239c1a9ff5bc5e265b'),
    (EvidenceResearchPlan,'rm',True,True,'c06edc55a3f6b7a25ceb280da6b7eee044186976d5b889d41f9e90bdfe6cafa3'),
    (PortfolioDecision,'pm',False,False,'c9b26307d7639ac8815e25c973d8df5b94e70e6d956a97e148d44a8fd1201076'),
    (PortfolioDecision,'pm',False,True,'97630436f7b7c7dff35dff17b15b0dda1e16871e50b7cce2c53251f9e0d21b8e'),
    (PortfolioDecision,'pm',True,False,'28190b7f322706f38115b7a5d660721fc216a9b9a1b64632aae877d57236b750'),
    (PortfolioDecision,'pm',True,True,'28190b7f322706f38115b7a5d660721fc216a9b9a1b64632aae877d57236b750'),
])
def test_probability_disabled_exact_baseline(base,layer,prices,timing,digest):
    schema=decision_schema(base,{'price_plan_legs':False,'rating_probability_fields':False,'price_plan_evaluation_enabled':prices,'rating_timing_decoupled':timing},layer)
    encoded=json.dumps(schema.model_json_schema(),ensure_ascii=False).encode()
    assert hashlib.sha256(encoded).hexdigest()==digest


@pytest.mark.parametrize('factory',[create_research_manager,create_portfolio_manager])
def test_probability_prompt_has_one_rule(factory):
    llm=Mock();llm.with_structured_output.side_effect=NotImplementedError
    llm.invoke.return_value=SimpleNamespace(content='**Rating**: Hold')
    state=Propagator().create_initial_state('QQQ','2026-10-02')
    state.update(investment_plan='研究决定',trader_investment_plan='交易决定')
    factory(llm,{})(state)
    prompt=llm.invoke.call_args[0][0]
    assert '证据不足明确未知' not in prompt
    assert '给出评级' in prompt and '0–1' in prompt and '向0.5收缩' in prompt
    assert '关键输入缺失' in prompt and '原因' in prompt


def test_pending_live_replacement_and_settled_bytes(tmp_path):
    path=tmp_path/'memory.md';log=TradingMemoryLog({'memory_log_path':str(path)})
    log.store_decision('A','2026-10-02','Rating: Buy\n旧决定')
    log.store_decision('B','2026-10-01','Rating: Hold\n已结算')
    log.update_with_outcome('B','2026-10-01',.1,.05,5,'反思')
    settled=path.read_text().split(log._SEPARATOR)[1]
    log.store_decision('A','2026-10-02','Rating: Sell\n新决定',replace_pending=True)
    assert len(log.load_entries())==2
    assert log.load_entries()[0]['rating']=='Sell'
    assert log.load_entries()[0]['decision']=='Rating: Sell\n新决定'
    assert path.read_text().split(log._SEPARATOR)[1]==settled
    before=path.read_bytes()
    log.store_decision('B','2026-10-01','Rating: Buy\n不可覆盖',replace_pending=True)
    log.store_decision('A','2026-10-02','不可得：输入缺失',replace_pending=True)
    log.store_decision('A','2026-10-02','Rating: Buy\n未知来源')
    assert path.read_bytes()==before


@pytest.mark.parametrize('mode,historical,status,replace',[
    ('live',True,'success',True),('backfill',False,'success',False),
    (None,False,'success',True),(None,True,'success',False),
    ('live',False,'failed',False),('live',False,'unavailable',False)])
def test_graph_replacement_source_gate(monkeypatch,mode,historical,status,replace):
    graph=object.__new__(TradingAgentsGraph);graph.config={'analysis_mode':mode}
    graph._log_state=Mock();graph.memory_log=Mock()
    monkeypatch.setattr('tradingagents.graph.trading_graph.is_historical',lambda _:historical)
    graph.record_decision('A','2026-10-02',{'final_trade_decision':'Rating: Sell','status':status})
    assert graph.memory_log.store_decision.call_args.kwargs['replace_pending'] is replace
