"""Alpaca 新闻源适配上游 ticker/global news 接口。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.date_window import in_window
from tradingagents.dataflows.vendors.alpaca.client import get_shared_client


def _format(title: str, result: dict, start: datetime, end: datetime) -> str:
    articles = result["articles"]
    if not articles:
        return f"Alpaca 在 {start.isoformat()} 至 {end.isoformat()} 未返回新闻"
    lines = [f"## {title}\n"]
    for item in articles:
        created = item.get("created_at", "unknown time")
        source = item.get("source", "Alpaca")
        lines.append(f"### {item.get('headline', 'No title')} (source: {source}, created_at: {created})")
        if item.get("summary"):
            lines.append(item["summary"])
        if item.get("url"):
            lines.append(f"Link: {item['url']}")
        if item.get("revised_after_cutoff"):
            lines.append("内容可能已在截止后修订。")
        lines.append("")
    if result.get("truncated"):
        lines.append(f"结果已截断，仅含最新 {len(articles)} 条。")
    return "\n".join(lines)


def _read(
    title: str, symbols: list[str] | None, start: datetime, end: datetime, limit: int | None,
) -> str:
    config = get_config()
    cutoff = config.get("news_cutoff_utc")
    if cutoff:
        parsed = datetime.fromisoformat(str(cutoff).replace("Z", "+00:00"))
        parsed = parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
        end = min(end, parsed)
    if end <= start:
        return "当前点时窗口内没有可用的 Alpaca 新闻"
    result = get_shared_client().get_news(symbols, start, end, limit=limit)
    rows = [
        item for item in result["articles"]
        if in_window(_parse_created(item.get("created_at")), start, end - timedelta(days=1))
    ]
    result["articles"] = rows
    return _format(title, result, start, end)


def _parse_created(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    config = get_config()
    market_tz = ZoneInfo(config.get("market_timezone") or "America/New_York")
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=market_tz)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=market_tz) + timedelta(days=1)
    return _read(
        f"{ticker} Alpaca news", [ticker], start, end, config.get("news_article_limit", 20)
    )


def get_global_news(
    as_of_date: str,
    look_back_days: int | None = None,
    limit: int | None = None,
) -> str:
    config = get_config()
    days = config.get("global_news_lookback_days", 7) if look_back_days is None else look_back_days
    count = config.get("global_news_article_limit", 10) if limit is None else limit
    market_tz = ZoneInfo(config.get("market_timezone") or "America/New_York")
    end = datetime.strptime(as_of_date, "%Y-%m-%d").replace(tzinfo=market_tz) + timedelta(days=1)
    start = end - timedelta(days=days)
    return _read("Global Alpaca news", None, start, end, count)
