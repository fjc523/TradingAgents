"""Guard the news analyst prompt against tool-signature drift (#1116).

The prompt used to advertise ``get_news(query, ...)``, tricking the LLM into
hallucinating free-text query calls. The instrument now comes from the run's
state, so the model passes only the window.
"""
import inspect

import pytest

import tradingagents.agents.analysts.news_analyst as na
from tradingagents.agents.tools import get_news


@pytest.mark.unit
def test_get_news_takes_the_window_not_a_query():
    arg_names = set(get_news.tool_call_schema.model_json_schema()["properties"])
    assert arg_names == {"start_date", "end_date"}


@pytest.mark.unit
def test_news_prompt_matches_get_news_signature():
    src = inspect.getsource(na)
    assert "get_news(start_date, end_date)" in src
    assert "get_news(query" not in src


def test_news_prompt_requests_news_before_macro_and_two_tables(monkeypatch):
    """核对真正交给模型的新闻提示，而非只检索源码。"""
    from langchain_core.messages import AIMessage
    captured = []
    def turn(prompt, llm, tools, messages):
        captured.append(prompt.invoke({"messages": messages}).to_messages()[0].content)
        return AIMessage(content="新闻样本"), "新闻样本"
    monkeypatch.setattr(na, "take_turn", turn)
    state = {"trade_date": "2026-10-02", "company_of_interest": "TSLA", "asset_type": "stock", "messages": []}
    na.create_news_analyst(object())(state)
    text = captured[0]
    assert "首先调用get_news(start_date, end_date)" in text
    assert "事件表" in text and "催化剂日历" in text
    assert "不要重复罗列" in text and "工具失败时换一个" in text
    assert "**Rating**:" not in text and "不要输出评级或交易动作" in text
    assert "another assistant" not in text
