"""进程级共享 Alpaca HTTP 客户端和请求限流器。"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from datetime import UTC, date, datetime, time as day_time, timedelta
from typing import Any, Callable

import requests

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import VendorNotConfiguredError, VendorUnavailableError

_DATA_URL = "https://data.alpaca.markets"
logger = logging.getLogger(__name__)
_SINGLETON = None
_SINGLETON_LOCK = threading.Lock()


class _SlidingWindowLimiter:
    def __init__(self, requests_per_minute: int, *, clock=time.monotonic, sleeper=time.sleep):
        self.limit = max(1, int(requests_per_minute))
        self.requests = deque()
        self.clock = clock
        self.sleeper = sleeper
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = self.clock()
                while self.requests and now - self.requests[0] >= 60:
                    self.requests.popleft()
                if len(self.requests) < self.limit:
                    self.requests.append(now)
                    return
                delay = self.requests[0] + 60 - now
            self.sleeper(delay)


class AlpacaClient:
    """共用 HTTP session，确保上游新闻和主项目上下文走同一限流入口。"""

    def __init__(
        self,
        *,
        requests_per_minute: int | None = None,
        session: requests.Session | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        config = get_config()
        self.requests_per_minute = int(
            requests_per_minute or config.get("alpaca_requests_per_minute", 180)
        )
        self.session = session or requests.Session()
        self._sleep = sleeper
        self._clock = clock
        self._limiter = _SlidingWindowLimiter(
            self.requests_per_minute, clock=monotonic, sleeper=sleeper
        )
        self._rate_limit_remaining: int | None = None

    @property
    def rate_limit_remaining(self) -> int | None:
        """最近一次响应中 Alpaca 返回的剩余请求数；尚无响应时为 None。"""
        return self._rate_limit_remaining

    @property
    def headers(self) -> dict[str, str]:
        key = os.environ.get("APCA_API_KEY_ID")
        secret = os.environ.get("APCA_API_SECRET_KEY")
        if not key or not secret:
            raise VendorNotConfiguredError(
                "缺少 Alpaca 凭据：请设置 APCA_API_KEY_ID 与 APCA_API_SECRET_KEY"
            )
        return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        headers = self.headers
        for attempt in range(4):
            self._limiter.acquire()
            try:
                response = self.session.get(url, params=params, headers=headers, timeout=20)
            except requests.RequestException as exc:
                raise VendorUnavailableError(f"Alpaca 请求失败：{type(exc).__name__}") from exc
            remaining = response.headers.get("X-Ratelimit-Remaining")
            try:
                self._rate_limit_remaining = int(remaining) if remaining is not None else None
            except (TypeError, ValueError):
                self._rate_limit_remaining = None
            if response.status_code == 429:
                logger.warning("Alpaca 返回 HTTP 429，等待限流重置后重试")
                if attempt >= 3:
                    raise VendorUnavailableError("Alpaca HTTP 429，重试三次后仍受限")
                reset = response.headers.get("X-Ratelimit-Reset")
                try:
                    delay = max(0, float(reset) - self._clock()) if reset else 1.0
                except ValueError:
                    delay = 1.0
                self._sleep(delay)
                continue
            try:
                response.raise_for_status()
                body = response.json()
            except (requests.RequestException, ValueError) as exc:
                raise VendorUnavailableError(
                    f"Alpaca HTTP {response.status_code}：{type(exc).__name__}"
                ) from exc
            if not isinstance(body, dict):
                raise VendorUnavailableError("Alpaca 返回值不是 JSON 对象")
            return body
        raise VendorUnavailableError("Alpaca 请求重试已耗尽")

    def get_bars(
        self,
        symbols: list[str],
        start: date | str,
        end: date | str,
        *,
        feed: str = "sip",
        adjustment: str = "all",
        timeframe: str = "1Day",
    ) -> dict[str, list[dict[str, Any]]]:
        if feed not in {"sip", "iex"}:
            raise ValueError("Alpaca bars feed must be 'sip' or 'iex'")
        start_dt = _as_datetime(start)
        end_dt = _as_datetime(end)
        if not isinstance(end, datetime) and len(str(end)) == 10:
            end_dt += timedelta(days=1)
        if feed == "sip" and timeframe.lower() not in {"1day", "1d"} \
                and end_dt > datetime.now(UTC) - timedelta(minutes=15):
            raise ValueError("免费 Alpaca 套餐不能请求最近 15 分钟的 SIP 数据")
        params: dict[str, Any] = {
            "symbols": ",".join(symbols), "start": start_dt.isoformat(),
            "end": end_dt.isoformat(), "timeframe": timeframe,
            "feed": feed, "adjustment": adjustment, "sort": "asc", "limit": 10000,
        }
        rows: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in symbols}
        token = None
        while True:
            if token:
                params["page_token"] = token
            body = self._get(f"{_DATA_URL}/v2/stocks/bars", params)
            for symbol, bars in (body.get("bars") or {}).items():
                rows.setdefault(symbol, []).extend(bars or [])
            token = body.get("next_page_token")
            if not token:
                break
        return rows

    def get_snapshots(
        self, symbols: list[str], *, feed: str = "overnight",
    ) -> dict[str, dict[str, Any]]:
        if feed not in {"overnight", "iex"}:
            raise ValueError("Alpaca snapshot feed must be 'overnight' or 'iex'")
        body = self._get(
            f"{_DATA_URL}/v2/stocks/snapshots",
            {"symbols": ",".join(symbols), "feed": feed},
        )
        return body

    def get_news(
        self,
        symbols: list[str] | None,
        start: datetime,
        end: datetime,
        *,
        limit: int | None = None,
        max_pages: int = 20,
    ) -> dict[str, Any]:
        if limit is not None and limit <= 0:
            return {"articles": [], "truncated": False}
        start_utc, end_utc = _utc(start), _utc(end)
        params: dict[str, Any] = {
            "start": start_utc.isoformat().replace("+00:00", "Z"),
            "end": end_utc.isoformat().replace("+00:00", "Z"),
            "limit": 50,
            "sort": "desc",
        }
        if symbols:
            params["symbols"] = ",".join(symbols)
        articles: list[dict[str, Any]] = []
        token = None
        truncated = False
        pages = 0
        while pages < max_pages:
            if token:
                params["page_token"] = token
            body = self._get(f"{_DATA_URL}/v1beta1/news", params)
            pages += 1
            for item in body.get("news", []) or []:
                created = _parse_timestamp(item.get("created_at"))
                if created is None or not (start_utc <= created < end_utc):
                    continue
                article = dict(item)
                updated = _parse_timestamp(item.get("updated_at"))
                article["revised_after_cutoff"] = updated is not None and updated > end_utc
                articles.append(article)
            token = body.get("next_page_token")
            if limit is not None and len(articles) >= limit:
                break
            if not token:
                break
        else:
            truncated = bool(token)
        articles.sort(key=lambda item: _parse_timestamp(item.get("created_at")) or datetime.min.replace(tzinfo=UTC), reverse=True)
        if limit is not None:
            articles = articles[:limit]
        return {"articles": articles, "truncated": truncated}


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _parse_timestamp(value) -> datetime | None:
    if not value:
        return None
    try:
        return _utc(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except (ValueError, TypeError):
        return None


def _as_datetime(value: date | datetime | str) -> datetime:
    if isinstance(value, datetime):
        return _utc(value)
    if isinstance(value, date):
        return datetime.combine(value, day_time.min, tzinfo=UTC)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if len(str(value)) == 10:
        parsed += timedelta(days=0)
    return _utc(parsed)


def get_shared_client() -> AlpacaClient:
    """返回唯一的进程级 Alpaca 客户端。"""
    global _SINGLETON
    if _SINGLETON is None:
        with _SINGLETON_LOCK:
            if _SINGLETON is None:
                _SINGLETON = AlpacaClient()
    return _SINGLETON
