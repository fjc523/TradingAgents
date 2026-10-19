"""角色自有指令的市场契约；不接收报告、历史或工具事实。"""


def instruction_template(text, config):
    """只在模板构造处替换固定的美国市场占位语义。"""
    market = (config or {}).get("market_context")
    if not market:
        return text
    currency = market["quote_currency"]
    benchmark = market["benchmark_symbol"]
    return (text.replace("美元", currency)
            .replace("个股/行业ETF主口径为对SPY超额，宽基/指数为绝对收益", f"本标的主口径为对{benchmark}的本币价格超额收益（拆合股校正、不含现金分红）")
            .replace("最新季度趋势", "最新已披露官方财报实际期间趋势"))


def market_instruction(config):
    """在角色提示构造时明确本次分析口径。"""
    market = (config or {}).get("market_context")
    if not market:
        return ""
    return (f"\n本次市场契约：{market['market']}，报价与点位币种{market['quote_currency']}，"
            f"比较基准{market['benchmark_symbol']}；主口径为本币、拆合股校正且不含现金分红的价格超额收益。"
            "财报频率、截止日、单位及比较期间只按输入官方披露，不把年度/中期强称最新季度；无历史PIT不伪造。"
            "执行约束UNKNOWN时所有买入与减仓腿均仅观察，保留未知原因，不声称可执行。")
