from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
    opponent_argument_or_opening,
    report_or_absent,
)


def create_neutral_debator(llm, config=None):
    if config is None:
        from tradingagents.dataflows.config import get_config
        config = get_config()
    direction_lock = config.get('risk_layer_direction_lock', True)
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_aggressive_response = opponent_argument_or_opening(
            risk_debate_state.get("current_aggressive_response", ""), "aggressive analyst"
        )
        current_conservative_response = opponent_argument_or_opening(
            risk_debate_state.get("current_conservative_response", ""), "conservative analyst"
        )

        market_research_report = report_or_absent(state["market_report"], "market")
        sentiment_report = report_or_absent(state["sentiment_report"], "sentiment")
        news_report = report_or_absent(state["news_report"], "news")
        fundamentals_report = report_or_absent(state["fundamentals_report"], "fundamentals")
        instrument_context = get_instrument_context_from_state(state, profile='neutral_debator')
        portfolio_context = get_portfolio_context_from_state(state)

        trader_decision = state["trader_investment_plan"]

        prompt = f"""你是中立价格方案审阅人，使用个人投资者和单标的标准仓位100%口径，审阅交易员方案。方案与评级一致性、盈亏比、决策周期匹配。
每人≤600字，依次输出：最大问题或遗漏；具体修改（目标配置、区间、止损/失效价、触发条件）；与交易员的分歧点。每项修改引用证据；无需修改时给明确理由。允许反对交易方向，不以立场替代数据。
交易员方案：{trader_decision}
{instrument_context}
{portfolio_context}
市场报告：{market_research_report}
情绪报告：{sentiment_report}
新闻报告：{news_report}
基本面报告：{fundamentals_report}
讨论历史：{history}
其他视角：{current_aggressive_response}
{current_conservative_response}
""" + get_language_instruction()

        if direction_lock:
            prompt = prompt.replace('允许反对交易方向，不以立场替代数据。',
                '只审目标配置、区间、止损/失效价、事件与跳空风险。若认为方向有误，只能列研究经理未考虑的可核对新证据并注明来源，由组合经理决定；不得仅凭盈亏比或入场点不足主张改变方向。')
            prompt = prompt.replace('方案与评级一致性、盈亏比、决策周期匹配。',
                '点位方案是否符合止损ATR范围、盈亏比及目标方法规则、执行周期是否匹配。')
            prompt = prompt.replace('是否错失上行、止损过紧、配置过低；不要替交易员辩护。',
                '是否错失合格突破或回踩执行机会、止损过紧、目标配置不当。')
            prompt = prompt.replace('交易员方案：', '研究经理完整方向计划：' + state.get('investment_plan', '未提供，不能假称研究经理已考虑') + '\n交易员方案：', 1)

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
