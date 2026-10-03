from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.turn import take_turn
from tradingagents.agents.context import ANALYST_INSTRUCTION, get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.tools import (
    get_global_news,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
)

# 分析师与工具节点共享此工具列表。
TOOLS = (
    get_news,
    get_global_news,
    get_macro_indicators,
    get_prediction_markets,
)


def available_news_tools():
    """只有未配置FRED时不绑定宏观工具，富途来源保留。"""
    import os
    from tradingagents.dataflows.router import get_vendor
    from tradingagents.dataflows.config import get_config
    chain = get_vendor("macro_data", "get_macro_indicators")
    if chain == "default":
        chain = "fred"
    names = [name.strip() for name in chain.split(",") if name.strip()]
    return tuple(tool for tool in TOOLS if tool.name != "get_macro_indicators" or names != ["fred"] or os.environ.get("FRED_API_KEY"))


def create_news_analyst(llm):
    tools = available_news_tools()
    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "公司" if asset_type == "stock" else "基金或资产"
        instrument_context = get_instrument_context_from_state(state, profile='news_analyst')

        system_message = (
            f"你是新闻分析师，首先调用get_news(start_date, end_date)获取本{asset_label}的新闻，"
            f"再按需调用其他可用工具：{', '.join(tool.name for tool in tools if tool.name != 'get_news')}。"
            "按公司事件→行业/竞争对手→相关宏观的优先级组织，只分析对决策周期的含义。"
            "不要重复罗列注入上下文已有的宏观发布数据。正文≤1200字，另附事件表"
            "（日期、来源、事件或观点、重要性、方向、是否可能已反映在价格中）和"
            "决策周期内的催化剂日历。无可靠日期时说明未确认。"
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    ANALYST_INSTRUCTION,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        result, report = take_turn(prompt, llm, tools, state["messages"])

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
