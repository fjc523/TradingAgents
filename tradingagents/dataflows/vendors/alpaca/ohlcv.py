"""通过进程共享客户端读取Alpaca SIP复权日线。"""
import pandas as pd
from tradingagents.dataflows.vendors.alpaca.client import get_shared_client


def load_alpaca_ohlcv(symbol, start, end):
    rows = []
    token = None
    client = get_shared_client()
    while True:
        params = {"symbols": symbol, "start": start, "end": end + "T23:59:59Z",
                  "feed": "sip", "adjustment": "all", "timeframe": "1Day", "limit": 10000}
        if token:
            params["page_token"] = token
        response = client._get("https://data.alpaca.markets/v2/stocks/bars", params)
        rows.extend(response.get("bars", {}).get(symbol, []))
        token = response.get("next_page_token")
        if not token:
            break
    data = pd.DataFrame(rows).rename(columns={"t":"Date","o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
    if not data.empty:
        data["Date"] = pd.to_datetime(data["Date"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    return data
