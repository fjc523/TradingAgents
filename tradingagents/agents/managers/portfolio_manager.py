"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call. Its rating is the run's
``final_rating``, and the decision is rendered to markdown as
``final_trade_decision`` for the memory log, CLI display and saved reports.
When a provider does not expose structured output, the agent falls back to
free-text generation and the rating is read from that text.
"""

from __future__ import annotations

from tradingagents.agents.schemas import decision_schema

from tradingagents.dataflows.social_result import lesson_reference_instruction

from tradingagents.dataflows.social_result import social_absence_instruction

import re

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
)
from tradingagents.agents.rating import rating_definitions, allocation_instruction, output_flags, flags_for_text, direction_flags, parse_rating
from tradingagents.agents.schemas import ALLOCATION_INSTRUCTION, price_plan_instruction, PortfolioDecision, LegacyPortfolioDecision, render_pm_decision
from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured, invoke_structured


def create_portfolio_manager(llm, config=None):
    if config is None:
        from tradingagents.dataflows.config import get_config
        config = get_config()
    config = dict(config)
    direction_lock = config.get('risk_layer_direction_lock', True)
    schema = PortfolioDecision if direction_lock else LegacyPortfolioDecision
    schema = decision_schema(schema, config, "pm")
    structured_llm = bind_structured(llm, schema, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state, profile='portfolio_manager')
        portfolio_context = get_portfolio_context_from_state(state)

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        market_report = (state.get("market_report") or "").strip()
        table = re.search(r"(?m)^.*关键价位表.*\n(?:\s*\n)?((?:\|[^\n]*\n?)+)", market_report)
        price_context = table[1] if table else market_report
        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

{portfolio_context}

---

{rating_definitions(config)}

默认沿用交易员点位，修改须说明理由；方案必须与评级一致。Underweight或Sell不新建仓，建仓方案首句写不适用及原因。执行摘要≤4句，投资论点≤800字，每个价格方案≤200字。
市场关键价位（提取失败时为完整市场报告）：
{price_context}

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

Ground every conclusion in specific evidence from the analysts. The risk debate always contains conflicting stances; deciding which is stronger is the job, so conflict alone is not a reason to Hold. Commit to the stronger case, sized by how decisively it wins. Choose Hold only when the evidence is still balanced after that weighing, or too thin to support a call; do not force a direction to appear decisive. Weigh the analysts on their merits, independent of speaking order.

## Output

Write these sections, in this order, starting with the rating on its own line:

- **Rating**: exactly one of Buy / Overweight / Hold / Underweight / Sell
- **Executive Summary**: the call and how to act on it
- **Investment Thesis**: the evidence that decided it, and what would change it

{price_plan_instruction(config)}
{allocation_instruction(config)}

{NO_EXTERNAL_TOOLS}{get_language_instruction(labelled=True)}"""

        if direction_lock:
            prompt = f"""你是组合经理，综合执行风险并给出最终方向评级与执行方案。
{instrument_context}

**研究经理完整计划与分歧点（方向锚点）：**
{research_plan}

**交易员方案：**
{trader_plan}

**风险审阅历史：**
{history}

{portfolio_context}
市场关键价位（提取失败时为完整市场报告）：
{price_context}
{lessons_line}

默认沿用研究经理recommendation，不能以交易员改后评级或审阅意见多数替代研究判断。只有研究经理未考虑的新信息才可改变方向：late news补抓、新增数据或审阅人列出的可核对新事实，须注明来源、事实及方向影响。盈亏比、入场点不足和笼统“风险较高”不是依据。
默认沿用交易员点位，修改须说明理由。Underweight或Sell不新建仓。执行摘要≤4句、投资论点≤800字、每项点位≤200字。
{rating_definitions(config)}
{price_plan_instruction(config)}
{allocation_instruction(config)}
## Output
## 输出要求
- **Rating**：Buy / Overweight / Hold / Underweight / Sell
- **Executive Summary**：方向、配置及如何执行
- **Investment Thesis**：决定性事实及复评条件
- **方向变更**：direction_change必填“否”或“是：具体新证据”
{NO_EXTERNAL_TOOLS}{get_language_instruction(labelled=True)}"""

        if config.get("price_plan_evaluation_enabled", True):
            prompt += "\n输出可选stop_loss与first_target绝对价格，保留点位依据；无可靠值留空，不将模糊目标当第一目标。"
        if config.get('rating_timing_decoupled', True):
            if not direction_lock:
                prompt=prompt.replace('方案必须与评级一致。','')
                prompt += '\n'+rating_definitions(config)
            prompt += '\nBuy/Overweight没有合格入场点时保持评级；entry_plan首句写“不适用：等待回踩至 X 或突破 Y 确认”，X/Y均为输入中的具体回踩价和突破价；缺锚点须明确说明缺失，不能编造价位。'

        # The typed rating is the decision; the rendered text only carries it.
        # Read back from text, a rating the thesis quotes could replace it.
        if config.get("rating_probability_fields", True):
            prompt += "\n输出可选prob_outperform_5d、prob_outperform_20d（0–1）与expected_return_20d_range，每项单列；证据不足明确未知。"
        prompt += lesson_reference_instruction(config)
        prompt += social_absence_instruction(config)
        decision = invoke_structured(structured_llm, prompt, "Portfolio Manager")
        if decision is not None:
            final_trade_decision = render_pm_decision(decision)
            final_rating = decision.rating.value
        else:
            final_trade_decision = llm.invoke(prompt).content
            final_rating = parse_rating(final_trade_decision)

        new_risk_debate_state = {
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
            "structured_pm_decision": decision.model_dump(mode="json") if decision is not None else None,
            "decision_flags": {**state.get("decision_flags", {}), "pm": {**(output_flags(decision.model_dump(mode="json"),config,layer="pm") if decision is not None else flags_for_text(final_trade_decision,config,layer="pm")), **direction_flags(state,decision.model_dump(mode="json") if decision is not None else None,layer="pm",text=final_trade_decision)}},
            "final_rating": final_rating,
        }

    return portfolio_manager_node
