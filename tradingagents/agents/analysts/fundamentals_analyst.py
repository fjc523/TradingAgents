from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.turn import take_turn
from tradingagents.agents.context import ANALYST_INSTRUCTION, get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.tools import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
    get_earnings_expectations,
)

# The tools this analyst is offered; its tool node is built from the same tuple.
TOOLS = (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_insider_transactions,
)


def available_tools(config=None, asset_type="stock"):
    """关闭时返回原工具集，ETF/指数不查询一致预期。"""
    if config is None:
        from tradingagents.dataflows.config import get_config
        config = get_config()
    return TOOLS + (get_earnings_expectations,) if config.get('earnings_expectations_enabled', True) and asset_type == 'stock' else TOOLS


def create_fundamentals_analyst(llm, config=None):
    if config is None:
        from tradingagents.dataflows.config import get_config
        config = get_config()
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state, profile='fundamentals_analyst')

        focus = (
            "基金口径：成分集中度、市场广度、资金流与指数估值；不把ETF当作经营公司，不要求品牌或护城河。"
            if state.get("asset_type", "stock") == "etf"
            else "公司口径：下次财报是否在决策周期内、最新季度同比/环比和利润率、自由现金流与流动性、估值及内部人交易。"
        )
        system_message = (
            "你是基本面分析师，关注中短期决策周期内的约束。" + focus
            + "使用get_fundamentals、get_balance_sheet、get_cashflow、get_income_statement和get_insider_transactions。"
            "财报表头含⚠或陈旧提示时，报告第一段必须说明最新季报未取得及实际表体期末，指出哪些同比/环比结论不可靠。"
            "年度申报已收录而Q4未单列时明确区别年度期与季度表期，不推算Q4。"
            "每项写明报告期；自由现金流注明口径，估值注明所用价格、日期和来源；"
            "内部人交易区分计划性交易与Form 144拟售，不把拟售当实际成交。"
            "财报在周期内时首段提示事件风险；日历未找到时写未找到已确认的财报日。"
            "正文≤1500字，末尾可附关键约束表，不以一周作为基本面观察期，不重复日历/宏观已有明细。"
            + get_language_instruction()
        )

        tools = available_tools(config, state.get("asset_type", "stock"))
        if get_earnings_expectations in tools:
            system_message += "必须调用get_earnings_expectations；数据可用时单列‘预期与修正’小节，引用修正方向及幅度；不可得或回放不可用时明确原因，不用当前预期填历史。"

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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
