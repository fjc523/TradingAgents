from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
    opponent_argument_or_opening,
    report_or_absent,
)


def create_aggressive_debator(llm):
    def aggressive_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        aggressive_history = risk_debate_state.get("aggressive_history", "")

        current_conservative_response = opponent_argument_or_opening(
            risk_debate_state.get("current_conservative_response", ""), "conservative analyst"
        )
        current_neutral_response = opponent_argument_or_opening(
            risk_debate_state.get("current_neutral_response", ""), "neutral analyst"
        )

        market_research_report = report_or_absent(state["market_report"], "market")
        sentiment_report = report_or_absent(state["sentiment_report"], "sentiment")
        news_report = report_or_absent(state["news_report"], "news")
        fundamentals_report = report_or_absent(state["fundamentals_report"], "fundamentals")
        instrument_context = get_instrument_context_from_state(state)
        portfolio_context = get_portfolio_context_from_state(state)

        trader_decision = state["trader_investment_plan"]

        prompt = f"""你是激进价格方案审阅人，使用个人投资者和单标的标准仓位100%口径，审阅交易员方案。是否错失上行、止损过紧、配置过低；不要替交易员辩护。
每人≤600字，依次输出：最大问题或遗漏；具体修改（目标配置、区间、止损/失效价、触发条件）；与交易员的分歧点。每项修改引用证据；无需修改时给明确理由。允许反对交易方向，不以立场替代数据。
交易员方案：{trader_decision}
{instrument_context}
{portfolio_context}
市场报告：{market_research_report}
情绪报告：{sentiment_report}
新闻报告：{news_report}
基本面报告：{fundamentals_report}
讨论历史：{history}
其他视角：{current_conservative_response}
{current_neutral_response}
""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Aggressive Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": aggressive_history + "\n" + argument,
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Aggressive",
            "current_aggressive_response": argument,
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return aggressive_node
