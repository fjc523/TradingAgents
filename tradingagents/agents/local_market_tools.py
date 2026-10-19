"""新市场工具只读取本次运行已核验的市场快照，不引入美国默认链。"""

import json
from langchain_core.tools import tool
from tradingagents.dataflows.config import get_config


def _read(key, ticker=None):
    config = get_config()
    bundle = config.get("local_market_bundle") or {}
    if ticker and ticker != bundle.get("symbol"):
        return "DATA_UNAVAILABLE：请求证券与本次唯一身份不一致"
    return json.dumps(bundle.get(key, {"status": "unavailable", "reason": "本次未取得可靠资料"}), ensure_ascii=False, default=str)


@tool
def get_verified_market_snapshot(ticker: str, trade_date: str = "") -> str:
    """读取本证券完整日线快照，含P、币种、来源、指标周期与原始证据。"""
    return _read("market_snapshot", ticker)


@tool
def get_stock_data(ticker: str, start_date: str = "", end_date: str = "") -> str:
    """读取本次P截止的完整价格收益日线；不前填缺数或加入未完成K线。"""
    return _read("prices", ticker)


@tool
def get_indicators(ticker: str, symbol: str = "", curr_date: str = "", look_back_days: int = 200) -> str:
    """读取已核验市场指标与锚点，原始周期和参数随快照给出。"""
    return _read("market_snapshot", ticker)


@tool
def get_news(ticker: str, start_date: str = "", end_date: str = "") -> str:
    """读取本次中文别名新闻和官方公告检索结果、发布时间精度及覆盖限制。"""
    return _read("news", ticker)


@tool
def get_fundamentals(ticker: str, curr_date: str = "") -> str:
    """读取最新官方已披露财务关键字段、可比期间、逐表单位、披露及原文证据。"""
    return _read("financials", ticker)


@tool
def get_balance_sheet(ticker: str, freq: str = "interim", curr_date: str = "") -> str:
    """读取实际披露频率的合并资产负债表；不把中期改为季度。"""
    return _read("financials", ticker)


@tool
def get_income_statement(ticker: str, freq: str = "interim", curr_date: str = "") -> str:
    """读取实际已披露损益与同期比较，不使用未核验季度和自动年化。"""
    return _read("financials", ticker)


@tool
def get_cashflow(ticker: str, freq: str = "interim", curr_date: str = "") -> str:
    """读取经营现金流、资本开支与明确推导的FCF，不补造缺失数值。"""
    return _read("financials", ticker)


TOOLS = {
    "market": (get_stock_data, get_indicators, get_verified_market_snapshot),
    "news": (get_news,),
    "fundamentals": (get_fundamentals, get_balance_sheet, get_income_statement, get_cashflow),
}
