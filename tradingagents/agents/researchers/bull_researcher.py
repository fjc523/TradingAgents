from tradingagents.dataflows.social_result import social_absence_instruction
from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    opponent_argument_or_opening,
    report_or_absent,
)


def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = opponent_argument_or_opening(
            investment_debate_state.get("current_response", ""), "bear analyst"
        )
        market_research_report = report_or_absent(state["market_report"], "market")
        sentiment_report = report_or_absent(state["sentiment_report"], "sentiment")
        news_report = report_or_absent(state["news_report"], "news")
        fundamentals_report = report_or_absent(state["fundamentals_report"], "fundamentals")
        instrument_context = get_instrument_context_from_state(state, profile='bull_researcher')
        asset_type = state.get("asset_type", "stock")
        focus = (
            "关注基金成分集中度、市场广度、资金流、指数估值和追踪风险。"
            if asset_type == "etf" else
            "关注公司最新季度趋势、竞争地位、估值、财报与行业事件。"
            if asset_type == "stock" else "关注资产自身供需、流动性和波动，不假设存在公司财务。"
        )
        prompt = f"""你是多头研究员，围绕决策周期陈述本方论点，不给交易结论。{focus}
输出依次为：最强3条论据（注明来源报告）；承认对方有效论点；什么证据推翻本方观点；论证强度1–5。每次发言≤900字。
数字必须来自输入，推算须标注“推算”及输入数据。对方未发言时，预判并回应空方最可能提出的论点，明确写这是预判，不捏造已说过的观点。
{instrument_context}
市场报告：{market_research_report}
情绪报告：{sentiment_report}
新闻报告：{news_report}
基本面报告：{fundamentals_report}
辩论历史：{history}
对方最新论点：{current_response}
""" + get_language_instruction()

        prompt += social_absence_instruction()
        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
