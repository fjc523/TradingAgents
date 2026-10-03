"""研究经理读取完整报告与关闭兼容的固定输入验证。"""
import hashlib
import json

import pytest
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.schemas import LegacyResearchPlan, ResearchPlan, render_research_plan
from tradingagents.dataflows.config import run_config


class Capture:
    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, prompt):
        self.prompt = prompt
        return self.schema(recommendation='Hold', rationale='依据不足', strategic_actions='等待')


def fixed_state(symbol='SMTC'):
    return dict(company_of_interest=symbol, instrument_context='上下文', asset_type='stock', past_context='历史',
                market_report='市场完整报告', sentiment_report='情绪完整报告', news_report='新闻完整报告',
                fundamentals_report='基本面完整报告', investment_debate_state=dict(history='辩论',
                current_response='对方论点', bull_history='多方历史', bear_history='空方历史', count=2))


def test_disabled_original_prompt_and_schema_golden():
    # 基线由改动前72f46ce的原类和原提示独立捕获，不由新实现生成期望值。
    llm = Capture()
    with run_config({'output_language': 'Chinese'}):
        create_research_manager(llm, {'research_manager_reads_reports': False, 'debate_mode': 'legacy', 'rating_timing_decoupled': False})(fixed_state())
    assert hashlib.sha256(llm.prompt.encode()).hexdigest() == '847d24dc44c628678f142703ed7546b7860976286e516bdf40a09d3d75b6709d'
    assert hashlib.sha256(json.dumps(llm.schema.model_json_schema(), sort_keys=True).encode()).hexdigest() == 'b2acf3986011ff5bed4da1454ccd3e9699e30f63588320b2e1b8ba551fd1ab25'
    assert 'evidence_check' not in llm.schema.model_fields


@pytest.mark.parametrize('symbol', ['SMTC', 'QQQ', 'TSLA'])
def test_full_reports_in_order_without_truncation(symbol):
    state = fixed_state(symbol)
    keys = ['market_report', 'fundamentals_report', 'sentiment_report', 'news_report']
    for key in keys:
        state[key] = key + '完整内容' * 4000 + ':末尾保留'
    llm = Capture()
    create_research_manager(llm, {'research_manager_reads_reports': True, 'debate_mode': 'legacy'})(state)
    positions = [llm.prompt.index(state[key]) for key in keys]
    assert positions == sorted(positions)
    assert llm.prompt.index('辩论记录') > positions[-1]
    assert llm.prompt.index('历史教训') > llm.prompt.index('辩论记录')
    assert '双方都遗漏' in llm.prompt
    assert llm.schema.model_fields['evidence_check'].metadata


def test_missing_report_explicit_and_old_record_honest():
    state = fixed_state()
    state.pop('fundamentals_report')
    llm = Capture()
    result = create_research_manager(llm)(state)
    assert '(No fundamentals report in this run:' in llm.prompt
    assert '未提供；不能视为已核对' in result['investment_plan']
    old = LegacyResearchPlan(recommendation='Hold', rationale='旧理由', strategic_actions='旧动作')
    assert '引用核对' not in render_research_plan(old)


def test_evidence_length_limit():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ResearchPlan(recommendation='Hold', rationale='理由', strategic_actions='动作', evidence_check='字' * 201)


@pytest.mark.parametrize('reads,mode,fields', [(False,'legacy',set()), (True,'legacy',{'evidence_check'}),
    (False,'structured',{'cruxes'}), (True,'structured',{'evidence_check','cruxes'})])
def test_private_config_schema_isolation(reads, mode, fields):
    llm = Capture()
    node = create_research_manager(llm, {'research_manager_reads_reports': reads, 'debate_mode': mode})
    # 后建另一个实例不应污染已建节点的模式。
    create_research_manager(Capture(), {'research_manager_reads_reports': not reads, 'debate_mode': 'legacy'})
    node(fixed_state())
    assert set(llm.schema.model_fields) & {'evidence_check','cruxes'} == fields
