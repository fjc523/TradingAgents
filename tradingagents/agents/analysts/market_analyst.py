from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.turn import take_turn
from tradingagents.agents.context import ANALYST_INSTRUCTION, get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.tools import get_indicators, get_stock_data, get_verified_market_snapshot

# The tools this analyst is offered; its tool node is built from the same tuple.
TOOLS = (
    get_stock_data,
    get_indicators,
    get_verified_market_snapshot,
)


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        system_message = (
            "你是市场分析师。首先调用get_verified_market_snapshot取得日线快照；"
            "快照未含的指标或需要更长序列时再按需调用get_indicators，最多4轮工具。"
            "可选指标：close_10_ema、close_20_sma、close_50_sma、close_200_sma、macd、macds、macdh、rsi、boll、boll_ub、boll_lb、atr、vwma。"
            "不要求先下载完整行情CSV，不重复查询已有数值；快照与其他数据冲突时报告差异，不编造调和值。"
            "正文≤1500字，写明最新完整日线日期P、收盘价、ATR14及来源，分析决策周期内趋势、波动和量价。"
            "历史验证、反复支撑或涨跌幅必须有具体日期/价位证据。报告末尾附关键价位表，"
            "列为价位、类型（支撑/阻力/均线/布林/前高前低）、来源日期、距参考现价几个ATR；"
            "每个价位必须出自快照或价位锚点，缺失时注明不可计算。"
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
            "market_report": report,
        }

    return market_analyst_node
