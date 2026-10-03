"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

from tradingagents.agents.schemas import decision_schema

from langchain_core.messages import AIMessage

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
)
from tradingagents.agents.rating import rating_definitions, output_flags, flags_for_text
from tradingagents.agents.schemas import ALLOCATION_INSTRUCTION, price_plan_instruction, TraderProposal, LegacyTraderProposal, render_trader_proposal
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
    invoke_decision,
)


def create_trader(llm, config=None):
    if config is None:
        from tradingagents.dataflows.config import get_config
        config = get_config()
    config = dict(config)
    direction_lock = config.get('risk_layer_direction_lock', True) or config.get('rating_timing_decoupled', True)
    schema = TraderProposal if direction_lock else LegacyTraderProposal
    schema = decision_schema(schema, config, "trader")
    structured_llm = bind_structured(llm, schema, "Trader")

    def trader_node(state):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state, profile='trader')
        investment_plan = state["investment_plan"]
        # The research plan digests the debate but loses exact price structure;
        # give the Trader the technical market report so entry/stop levels are
        # grounded in real ATR / support-resistance / current price (#1167). The
        # report is empty when the user did not select the market analyst, so
        # only offer it (and the grounding instruction) when it has content.
        market_report = (state["market_report"] or "").strip()
        portfolio_context = get_portfolio_context_from_state(state)

        if market_report:
            grounding = (
                "Ground concrete price levels (entry, stop-loss, position sizing) in the technical "
                "market report's price structure -- current price, support/resistance, ATR, and "
                "volatility -- and use the research plan for direction and strategy. "
            )
            report_section = f"Technical Market Report:\n{market_report}\n\n"
        else:
            grounding = ""
            report_section = ""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "输出五档动作，与研究经理不同须说明原因；观点有分歧本身不是Hold的理由。 "
                    + grounding
                    # Entry/stop are numeric price fields. Asking for concrete
                    # levels invites a percentage ("15%"), which is not a price
                    # and fails the structured parse (#1288).
                    + "State entry price and stop-loss as absolute price levels in the "
                    "instrument's quote currency (for example 189.5), never a percentage "
                    "or a range; convert a percentage distance to the price level it "
                    "implies, or omit the field if you cannot state a number. "
                    + price_plan_instruction(config)
                    + rating_definitions(config)
                    + ALLOCATION_INSTRUCTION
                    + NO_EXTERNAL_TOOLS
                    + get_language_instruction(labelled=True)
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the research team's investment plan for {company_name}. "
                    f"{instrument_context}\n\n"
                    f"{report_section}"
                    f"{portfolio_context}\n\n"
                    f"Proposed Investment Plan:\n{investment_plan}\n\n"
                    "Make an informed, strategic trading decision.\n\n"
                    "## Output\n\n"
                    "Write these sections, in this order, starting with the action "
                    "on its own line:\n\n"
                    "- **Action**: exactly one of Buy / Overweight / Hold / Underweight / Sell.\n"
                    "- **Reasoning**: why, against the plan and the price structure\n"
                    "- **Entry Price**, **Stop Loss**, **Position Sizing**: when you can state them"
                ),
            },
        ]

        if direction_lock:
            messages[0]['content'] = messages[0]['content'].replace(
                '输出五档动作，与研究经理不同须说明原因；观点有分歧本身不是Hold的理由。 ',
                '默认沿用研究经理recommendation。只有研究经理未考虑的可核对新证据才允许改变方向；不能凭盈亏比、入场点不足或笼统风险改变。 ')
            messages[1]['content'] += '\n- **方向变更**：direction_change必填“否”或“是：具体新证据”，包含来源、事实及方向影响。'

        if config.get("price_plan_evaluation_enabled", True):
            messages[1]["content"] += "\n输出first_target绝对价格；无可靠第一目标留空，不编造。"
        trader_plan, structured = invoke_decision(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "structured_trader_proposal": structured,
            "decision_flags": {**state.get("decision_flags", {}), "trader": output_flags(structured,config,layer="trader") if structured is not None else flags_for_text(trader_plan,config,layer="trader")},
        }

    return trader_node
