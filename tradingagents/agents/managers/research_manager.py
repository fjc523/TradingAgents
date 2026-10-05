"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import decision_schema

from tradingagents.dataflows.social_result import lesson_reference_instruction

from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction, report_or_absent
from tradingagents.agents.rating import rating_definitions, allocation_instruction, output_flags, flags_for_text, direction_flags
from tradingagents.agents.schemas import ALLOCATION_INSTRUCTION, ResearchPlan, LegacyResearchPlan, EvidenceResearchPlan, CruxResearchPlan, render_research_plan
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
    invoke_decision,
)


def create_research_manager(llm, config=None):
    if config is None:
        from tradingagents.dataflows.config import get_config
        config = get_config()
    rating_guidance = rating_definitions(config)
    reads_reports = config.get('research_manager_reads_reports', True)
    structured_debate = config.get('debate_mode', 'structured') == 'structured'
    schema = (ResearchPlan if reads_reports else CruxResearchPlan) if structured_debate else (EvidenceResearchPlan if reads_reports else LegacyResearchPlan)
    schema = decision_schema(schema, config, "rm")
    structured_llm = bind_structured(llm, schema, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state, profile='research_manager')
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]

        prompt = f"""As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.

{instrument_context}

---

{rating_guidance}

The debate always contains conflicting arguments; deciding which side is stronger is the job, so conflict alone is not a reason to Hold. Commit to the side with the stronger case, sized by how decisively it wins. Choose Hold only when the evidence is still balanced after that weighing, or too thin to support a call; do not manufacture a direction to appear decisive. Weigh the bull and bear cases on their merits, independent of which side spoke first or last.

---

**历史教训：**\n{state.get("past_context", "") or "无已结算教训"}

**Debate History:**
{history}

## Output

Write these sections, in this order, starting with the recommendation on its own line:

- **Recommendation**: exactly one of Buy / Overweight / Hold / Underweight / Sell
- **Rationale**: ≤600字，依次为决定性论据前3条、被驳回论据及理由、关键不确定性、复评触发条件
- **Strategic Actions**: 交易员具体行动，配置以单标的标准仓位100%为参考单位

{allocation_instruction(config)}

{NO_EXTERNAL_TOOLS}""" + get_language_instruction(labelled=True)

        if reads_reports:
            reports = '\n\n'.join(f'**{label}：**\n{report_or_absent(state.get(key, ""), source)}'
                for key, label, source in [('market_report', '市场报告', 'market'),
                                           ('fundamentals_report', '基本面报告', 'fundamentals'),
                                           ('sentiment_report', '情绪报告', 'sentiment'),
                                           ('news_report', '新闻报告', 'news')])
            prompt = f"""你是研究经理与辩论裁判，直接核对完整报告，再评估双方证据。
{instrument_context}

{reports}

**辩论记录：**
{history}

**历史教训：**
{state.get('past_context', '') or '无已结算教训'}

## Output
## 输出要求
{rating_guidance}
冲突本身不构成Hold理由；独立比较证据，选择证据占优方，仅均衡或不足时Hold，不受先后发言影响。
- **Recommendation**：Buy / Overweight / Hold / Underweight / Sell
- **Rationale**：≤600字，决定性论据前3条、被驳回论据及理由、不确定性、复评触发条件。
- **Strategic Actions**：给交易员的行动要求。
- **引用核对**：≤200字，列辩手引用不符处及双方都遗漏但影响结论的事实；无则写“无”并列已核对的2–3个数据点。
{allocation_instruction(config)}
{NO_EXTERNAL_TOOLS}""" + get_language_instruction(labelled=True)

        if structured_debate:
            prompt = prompt.replace('starting with the recommendation on its own line:', 'starting with 3–5 evidence-based cruxes, then the recommendation:')
            prompt += '\n先输出3–5个分歧点裁决（cruxes），每项列多方主张、空方主张、决定性报告证据、胜方/未决及理由，再输出评级。不得凭发言顺序判胜负。'

        if config.get("rating_probability_fields", True):
            prompt += "\n输出可选prob_outperform_5d、prob_outperform_20d（0–1）与expected_return_20d_range，每项单列；给出评级必须给0–1概率数值，证据薄弱向0.5收缩；只有关键输入缺失且连评级都不能给时才写不可得及原因。"
        if config.get('price_plan_legs', True):
            prompt = prompt.replace('输出可选prob_outperform_', '输出prob_outperform_')
            # 新分支实际输出列表先证据和概率；旧分支完整保留原顺序。
            start = prompt.index('## Output')
            end = prompt.index(allocation_instruction(config), start)
            original_lines = prompt[start:end].splitlines()
            sections = {}
            retained = []
            for line in original_lines:
                key = next((name for name in ('Recommendation', 'Rationale', 'Strategic Actions', '引用核对')
                            if line.startswith('- **' + name + '**')), None)
                if key:
                    sections[key] = line
                else:
                    retained.append('按以下顺序输出：' if line.startswith('Write these sections, in this order,') else line)
            # 保留原列表全部语义以及评级指南/证据裁决要求，仅移动原有条目。
            ordered = []
            if structured_debate:
                ordered.append('- **分歧点裁决**：先列3–5个分歧点与证据裁决。')
            if '引用核对' in sections:
                ordered.append(sections['引用核对'])
            if config.get('rating_probability_fields', True):
                ordered.append('- **概率**：prob_outperform_5d、prob_outperform_20d（0–1）与expected_return_20d_range。')
            ordered.extend(sections[name] for name in ('Recommendation', 'Rationale', 'Strategic Actions') if name in sections)
            output = '\n'.join(retained + ordered) + '\n'
            prompt = prompt[:start] + output + prompt[end:]
            prompt += '\n自由文本输出顺序：分歧点 → 引用核对 → 概率 → 评级（Recommendation） → 理由和行动；先核对证据再给评级。'
        prompt += lesson_reference_instruction(config)
        investment_plan, structured = invoke_decision(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

        new_investment_debate_state = {
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
            "structured_research_plan": structured,
            "decision_flags": {**state.get("decision_flags", {}), "rm": {**(output_flags(structured,config,layer="rm") if structured is not None else flags_for_text(investment_plan,config,layer="rm")), **direction_flags(state,structured,layer="rm",text=investment_plan)}},
        }

    return research_manager_node
