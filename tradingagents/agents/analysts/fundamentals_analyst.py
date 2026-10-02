from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.turn import take_turn
from tradingagents.agents.context import ANALYST_INSTRUCTION, get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.tools import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
)

# The tools this analyst is offered; its tool node is built from the same tuple.
TOOLS = (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_insider_transactions,
)


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        focus = (
            "基金口径：成分集中度、市场广度、资金流与指数估值；不把ETF当作经营公司，不要求品牌或护城河。"
            if state.get("asset_type", "stock") == "etf"
            else "公司口径：下次财报是否在决策周期内、最新季度同比/环比和利润率、自由现金流与流动性、估值及内部人交易。"
        )
        system_message = (
            "你是基本面分析师，关注中短期决策周期内的约束。" + focus
            + "使用get_fundamentals、get_balance_sheet、get_cashflow、get_income_statement和get_insider_transactions。"
            "每项写明报告期；自由现金流注明口径，估值注明所用价格、日期和来源；"
            "内部人交易区分计划性交易与Form 144拟售，不把拟售当实际成交。"
            "财报在周期内时首段提示事件风险；日历未找到时写未找到已确认的财报日。"
            "正文≤1500字，末尾可附关键约束表，不以一周作为基本面观察期，不重复日历/宏观已有明细。"
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
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in TOOLS]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        result, report = take_turn(prompt, llm, TOOLS, state["messages"])

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
