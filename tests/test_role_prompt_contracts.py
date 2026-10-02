"""核对实际角色提示的证据交接、篇幅与评级边界，不调用网络和模型。"""

import importlib
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage


def state(asset_type="stock"):
    return {
        "company_of_interest": "SPY" if asset_type == "etf" else "TSLA",
        "asset_type": asset_type, "trade_date": "2026-10-02", "messages": [],
        "market_report": "## 关键价位表\n|价位|类型|\n|---|---|\n|350|SMA20|\n\n不应重复的长说明",
        "fundamentals_report": "最新季度数据", "news_report": "新闻样本", "sentiment_report": "情绪样本",
        "investment_plan": "研究方案", "trader_investment_plan": "交易员方案",
        "past_context": "历史教训唯一标记",
        "investment_debate_state": {"history": "研究辩论", "bull_history": "", "bear_history": "", "current_response": "", "count": 0},
        "risk_debate_state": {"history": "风险讨论", "aggressive_history": "", "conservative_history": "", "neutral_history": "",
                              "current_aggressive_response": "", "current_conservative_response": "", "current_neutral_response": "", "latest_speaker": "", "count": 0},
    }


@pytest.mark.parametrize("name,factory,required", [
    ("market_analyst", "create_market_analyst", ("首先调用get_verified_market_snapshot", "关键价位表", "ATR14", "≤1500")),
    ("fundamentals_analyst", "create_fundamentals_analyst", ("自由现金流", "报告期", "Form 144", "≤1500")),
])
def test_analyst_actual_brief(name, factory, required, monkeypatch):
    module = importlib.import_module("tradingagents.agents.analysts." + name)
    captured = []
    def take_turn(prompt, *args):
        captured.append(prompt.invoke({"messages": []}).to_messages()[0].content)
        return AIMessage(content="样本"), "样本"
    monkeypatch.setattr(module, "take_turn", take_turn)
    getattr(module, factory)(object())(state())
    assert all(text in captured[0] for text in required)
    assert "**Rating**:" not in captured[0]
    assert "another assistant" not in captured[0]


@pytest.mark.parametrize("side", ["bull", "bear"])
def test_research_branches_and_no_trade_call(side):
    module = importlib.import_module(f"tradingagents.agents.researchers.{side}_researcher")
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="观点")
    factory = getattr(module, f"create_{side}_researcher")
    for asset_type, focus in (("stock", "竞争地位"), ("etf", "成分集中度")):
        factory(llm)(state(asset_type))
        text = llm.invoke.call_args.args[0]
        assert focus in text and "论证强度1–5" in text and "推算" in text and "≤900" in text
        assert "新闻报告" in text and "**Rating**:" not in text
        if asset_type == "etf":
            assert "品牌" not in text and "竞争地位" not in text


@pytest.mark.parametrize("side", ["aggressive", "conservative", "neutral"])
def test_risk_reviews_concrete_changes_instead_of_defending(side):
    module = importlib.import_module(f"tradingagents.agents.risk_mgmt.{side}_debator")
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="修改建议")
    getattr(module, f"create_{side}_debator")(llm)(state())
    text = llm.invoke.call_args.args[0]
    assert all(value in text for value in ("价格方案审阅人", "具体修改", "分歧点", "≤600", "标准仓位100%"))
    assert "the firm's assets" not in text and "**Rating**:" not in text
    assert "FINAL TRANSACTION PROPOSAL:" not in text


def test_decision_handoff_contains_lessons_and_price_table():
    from tradingagents.agents.managers.research_manager import create_research_manager
    from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
    for factory in (create_research_manager, create_portfolio_manager):
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError
        llm.invoke.return_value = AIMessage(content="**Rating**: Hold")
        factory(llm)(state())
        text = llm.invoke.call_args.args[0]
        assert "历史教训唯一标记" in text
        if factory is create_portfolio_manager:
            assert "|350|SMA20|" in text and "不应重复的长说明" not in text
            assert "默认沿用交易员点位" in text and "修改须说明理由" in text
        else:
            assert "被驳回论据" in text and "复评触发条件" in text


def test_sentiment_is_explicit_about_missing_social_views():
    from tradingagents.agents.analysts.sentiment_analyst import _build_system_message
    text = _build_system_message(ticker="TSLA", start_date="2026-09-25", end_date="2026-10-02",
                                 news_block="新闻样本", stocktwits_block="不可用", reddit_block="无关帖子")
    assert "confidence必须为low" in text and "本评分仅反映新闻语气" in text
    assert "新闻语气" in text and "社交情绪" in text and "≤1000" in text
