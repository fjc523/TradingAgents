"""Company statements as they were filed, from SEC EDGAR.

Every other fundamentals vendor serves a period's current value and cuts the
statement at the fiscal period end. That is two claims a run should not make: a
period that has ended is not public until the company files, weeks later, and a
figure that was later restated is not what investors saw at the time.

EDGAR reports every fact with the date it was filed, so a run dated ``as_of_date``
serves exactly what was on file by then, restatements included at the vintage
that was current: Apple's 2008 total assets read 39.6B until the 2010 amendment
restated them to 36.2B.

Access needs no key or account, only a User-Agent identifying the caller, which
SEC requires and refuses requests without. US filers only: anything absent from
EDGAR's ticker map falls through to the next configured vendor.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

from tradingagents import __version__
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError, StaleVendorDataError
from tradingagents.dataflows.statement_freshness import StatementResult
from tradingagents.dataflows.files import replace_file

logger = logging.getLogger(__name__)

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_SUBMISSIONS_ARCHIVE_URL = "https://data.sec.gov/submissions/{name}"

_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# A filing history only changes when something new is filed, so one fetch per
# company per day serves every date a run asks about.
_CACHE_TTL_SECONDS = 24 * 60 * 60

# Line items, each with the tags filers use for it, best first. First match wins
# and values are never summed across tags: a company reporting revenue under two
# tags would otherwise be counted twice.
_STATEMENTS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "balance_sheet": [
        ("Total Assets", ("Assets",)),
        ("Current Assets", ("AssetsCurrent",)),
        ("Cash and Equivalents", ("CashAndCashEquivalentsAtCarryingValue",)),
        ("Total Liabilities", ("Liabilities",)),
        ("Current Liabilities", ("LiabilitiesCurrent",)),
        ("Stockholders Equity", ("StockholdersEquity",
                                 "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")),
    ],
    "income_statement": [
        ("Revenue", ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                     "SalesRevenueNet")),
        ("Cost of Revenue", ("CostOfRevenue", "CostOfGoodsAndServicesSold")),
        ("Gross Profit", ("GrossProfit",)),
        ("Operating Income", ("OperatingIncomeLoss",)),
        ("Net Income", ("NetIncomeLoss",)),
        ("Diluted EPS", ("EarningsPerShareDiluted",)),
    ],
    "cashflow": [
        ("Operating Cash Flow", ("NetCashProvidedByUsedInOperatingActivities",
                                 "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")),
        ("Investing Cash Flow", ("NetCashProvidedByUsedInInvestingActivities",)),
        ("Financing Cash Flow", ("NetCashProvidedByUsedInFinancingActivities",)),
        ("Capital Expenditure", ("PaymentsToAcquirePropertyPlantAndEquipment",
                                 "PaymentsToAcquireProductiveAssets")),
    ],
}

# A statement's figures cover a span: a quarter is about 90 days, a year about
# 365. One filing reports both the quarter and the year to date under the same
# end date, so a match on the end date alone can report half a year as a quarter.
_SPANS = {"quarterly": (60, 115), "annual": (300, 400)}

# A 10-Q's cash flows are often filed only year to date. A quarterly table takes
# the quarter where filed, else the span to date, named in the column; it never
# subtracts one filing from another, which would give a figure no filing states.
_YEAR_TO_DATE = ((150, 200, 6), (240, 290, 9))

# A fiscal year is a period an annual report covers. A 10-Q balance has no span
# to reject, and some filers' 10-Qs report twelve-month totals that pass the span
# check, so either would read as a fiscal year. The value is still the latest
# filing of any form: a recast after a split or spin-off counts from its filing.
_ANNUAL_FORMS = ("10-K", "20-F", "40-F")


def _user_agent() -> str:
    """Who SEC sees. No account or key exists; callers identify themselves.

    www.sec.gov, which serves the ticker map, refuses a User-Agent carrying no
    contact address: a client name alone or with a project URL gets 403, one
    with an address gets 200. So the default carries a placeholder address and
    the package version. Set SEC_EDGAR_USER_AGENT to your own name and address
    so SEC can reach you about your traffic rather than the project.
    """
    configured = os.getenv("SEC_EDGAR_USER_AGENT", "").strip()
    return configured or f"TradingAgents/{__version__} (contact@example.com)"




def _fetch_json(url: str) -> dict:
    """Read a public EDGAR document, respecting SEC's identification rule."""
    try:
        response = requests.get(url, headers={"User-Agent": _user_agent()}, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        # Every failure here is "this vendor cannot serve it now", so the router
        # moves on instead of seeing a transport exception it has no rule for.
        raise VendorUnavailableError(f"SEC EDGAR request failed ({status or type(exc).__name__})") from exc
    except ValueError as exc:
        raise VendorUnavailableError("SEC EDGAR returned an unreadable response") from exc


def _cached_json(url: str, name: str) -> dict:
    path = Path(get_config()["data_cache_dir"]) / "sec_edgar" / name
    if path.exists() and time.time() - path.stat().st_mtime < _CACHE_TTL_SECONDS:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            pass  # a truncated file is a miss, not a failure
    data = _fetch_json(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    replace_file(path, lambda temp: Path(temp).write_text(json.dumps(data), encoding="utf-8"))
    return data


def cik_for(ticker: str) -> str | None:
    """The filer's CIK, or None when the ticker is not a US filer."""
    table = _cached_json(_TICKERS_URL, "company_tickers.json")
    wanted = ticker.strip().upper()
    for entry in table.values():
        if entry.get("ticker", "").upper() == wanted:
            return f"{int(entry['cik_str']):010d}"
    return None


def _submission_rows(table, as_of_date):
    """必需日期列对齐且合法；非周期申报允许空reportDate，但不能缺filingDate。"""
    if not isinstance(table, dict) or not isinstance(table.get('form'), list):
        raise VendorUnavailableError('SEC submissions列式表结构非法')
    forms = table['form']
    for key in ('filingDate', 'reportDate'):
        values = table.get(key, [] if not forms else None)
        if not isinstance(values, list) or len(values) != len(forms):
            raise VendorUnavailableError(f'SEC submissions必需日期列{key}缺失、类型错误或长度不对齐')
    rows = []
    for i, form in enumerate(forms):
        row = {key: values[i] for key, values in table.items() if isinstance(values, list) and i < len(values)}
        try:
            filed = date.fromisoformat(row['filingDate'])
            report = row['reportDate']
            if form in ('10-Q', '10-K') or report:
                report_date = date.fromisoformat(report)
                if form in ('10-Q', '10-K') and report_date > filed:
                    raise ValueError('报告期晚于提交日')
        except (ValueError, KeyError, TypeError) as exc:
            raise VendorUnavailableError('SEC submissions必需日期值非法') from exc
        if filed <= date.fromisoformat(as_of_date):
            rows.append(row)
    return rows


def _latest_filing(cik, as_of_date, forms=('10-Q', '10-K')):
    payload = _cached_json(_SUBMISSIONS_URL.format(cik=cik), f'submissions_CIK{cik}.json')
    if not isinstance(payload, dict):
        raise VendorUnavailableError('SEC submissions根结构非法')
    filings = payload.get('filings')
    if not isinstance(filings, dict) or not isinstance(filings.get('recent'), dict):
        raise VendorUnavailableError('SEC submissions缺少filings.recent列式数据')
    rows = _submission_rows(filings['recent'], as_of_date)
    candidates = [row for row in rows if row.get('form') in forms and row.get('reportDate')]
    # 历史截止早于recent覆盖时，只补取可能包含历史申报的官方归档。
    if not candidates:
        files = filings.get('files', [])
        if not isinstance(files, list) or any(not isinstance(row, dict) for row in files):
            raise VendorUnavailableError('SEC submissions归档目录结构非法')
        archives = sorted(files, key=lambda row: row.get('filingTo', ''), reverse=True)
        for archive in archives:
            if archive.get('filingFrom', '') > as_of_date or not archive.get('name'):
                continue
            name = archive['name']
            table = _cached_json(_SUBMISSIONS_ARCHIVE_URL.format(name=name), f'submissions_{name}')
            old = _submission_rows(table, as_of_date)
            rows.extend(old)
            candidates.extend(row for row in old if row.get('form') in forms and row.get('reportDate'))
            if candidates:
                break
    if not candidates:
        return None
    latest = max(candidates, key=lambda row: (row['filingDate'], row['reportDate']))
    result = {key: latest[key] for key in ('form', 'reportDate', 'filingDate')}
    # Item 2.02只能证明财报发布8-K申报日，明确不等同推测的财报召开时间。
    earnings = [row['filingDate'] for row in rows if row.get('form') == '8-K'
                and '2.02' in {item.strip() for item in str(row.get('items', '')).split(',')}]
    if earnings:
        result['last_earnings_filing_date'] = max(earnings)
    return result


def latest_periodic_filing(cik, as_of_date):
    """分析截止前已申报的最新10-Q/10-K；不读未来申报。"""
    return _latest_filing(str(cik).zfill(10), as_of_date)


def _span_index(fact: dict, spans: tuple[tuple[int, int], ...]) -> int | None:
    """Which of ``spans`` a duration fact covers (0 for an instant fact), or None."""
    if "start" not in fact:
        return 0
    days = (date.fromisoformat(fact["end"]) - date.fromisoformat(fact["start"])).days
    return next((i for i, (low, high) in enumerate(spans) if low <= days <= high), None)


def _as_of(facts: dict, tags: tuple[str, ...], as_of_date: str, spans: tuple[tuple[int, int], ...],
           forms: tuple[str, ...] = ()) -> tuple[dict, str]:
    """({(period end, span index): value}, unit) for the tags the filer reports, as known then.

    A period reported more than once takes its latest filing on or before the
    date, so an amendment counts from the day it was filed and not before. The
    unit comes from the filing: most lines are USD, earnings per share are
    USD/shares, and scaling those alike would print a real figure as zero. A
    duration fact (revenue, cash flow) must cover one of ``spans``; an instant
    fact (a balance) has no span and serves any.
    """
    values: dict[tuple[str, int], float] = {}
    chosen_unit = "USD"
    # Tags are tried in order and a period keeps the first one that reports it:
    # filers renamed lines over the years, so one tag covers only part of the
    # history. Values are never added across tags, which would double count.
    for tag in tags:
        for unit, unit_values in ((facts.get(tag) or {}).get("units", {})).items():
            latest: dict[tuple[str, int], dict] = {}
            covered: set[tuple[str, int]] = set()   # periods a filing of ``forms`` reports
            for fact in unit_values:
                index = _span_index(fact, spans)
                key = (fact["end"], index)
                if fact["filed"] > as_of_date or index is None or key in values:
                    continue
                if not forms or fact.get("form", "").startswith(forms):
                    covered.add(key)
                seen = latest.get(key)
                if seen is None or fact["filed"] >= seen["filed"]:
                    latest[key] = fact
            latest = {key: fact for key, fact in latest.items() if key in covered}
            if latest:
                chosen_unit = unit
                values.update({key: fact["val"] for key, fact in latest.items()})
    return dict(sorted(values.items())), chosen_unit


def _statement(kind: str, ticker: str, freq: str, as_of_date: str, title: str) -> str:
    as_of_date = as_of_date or datetime.now().strftime("%Y-%m-%d")
    cik = cik_for(ticker)
    if cik is None:
        raise NoMarketDataError(ticker, ticker, "not a US SEC filer")

    facts = _cached_json(_FACTS_URL.format(cik=cik), f"CIK{cik}.json")
    us_gaap = (facts.get("facts") or {}).get("us-gaap")
    if not us_gaap:
        raise NoMarketDataError(ticker, ticker, "US filer with no us-gaap facts")

    quarterly = freq.lower() == "quarterly"
    if quarterly:
        spans = (_SPANS["quarterly"], *((low, high) for low, high, _ in _YEAR_TO_DATE))
        names = ["", *(f" ({months} months)" for _, _, months in _YEAR_TO_DATE)]
    else:
        spans, names = (_SPANS["annual"],), [""]
    forms = () if quarterly else _ANNUAL_FORMS
    lines = {label: _as_of(us_gaap, tags, as_of_date, spans, forms) for label, tags in _STATEMENTS[kind]}
    # Each row takes the shortest span it reports for a period, and a column
    # holds one span of one period, so a row filed only to date keeps its figure
    # beside a row filed by quarter.
    chosen = {label: {} for label in lines}
    for label, (values, _) in lines.items():
        for end, index in values:
            chosen[label][end] = min(index, chosen[label].get(end, index))
    periods = sorted({(end, index) for spans_of in chosen.values() for end, index in spans_of.items()})
    if not periods:
        raise NoMarketDataError(ticker, ticker, f"no {freq} {title.lower()} filed by {as_of_date}")

    header = (
        f"# {title} for {ticker.upper()} ({freq}), USD in millions unless the row says otherwise\n"
        f"# SEC EDGAR facts filed on or before {as_of_date}, at the values filed then\n\n"
    )
    rows = [",".join([""] + [end + names[index] for end, index in periods])]
    for label, (values, unit) in lines.items():
        # Every row spans the same columns, or a reader lines the table up wrong.
        if not values:
            rows.append(",".join([label] + ["unavailable (not tagged by this filer)"] * len(periods)))
            continue
        name = label if unit == "USD" else f"{label} ({unit})"
        # Plain numbers: a thousands separator would split the CSV field.
        cells = []
        for end, index in periods:
            value = values.get((end, index)) if chosen[label].get(end) == index else None
            cells.append("" if value is None else f"{value / 1e6:.0f}" if unit == "USD" else f"{value:.2f}")
        rows.append(",".join([name] + cells))
    text = header + "\n".join(rows) + "\n"
    latest_period = max(end for end, _ in periods)
    metadata = {'actual_source': 'sec_edgar', 'latest_period': latest_period,
                'stale': False, 'reason': '', 'as_of_date': as_of_date}
    filing, unavailable = None, None
    try:
        filing = latest_periodic_filing(cik, as_of_date) if quarterly else _latest_filing(cik, as_of_date, ('10-K',))
    except VendorUnavailableError as exc:
        unavailable = str(exc)
    expected = filing['reportDate'] if filing else None
    if filing:
        metadata.update(expected_period=expected, filing_date=filing['filingDate'], filing_form=filing['form'])
        if filing.get('last_earnings_filing_date'):
            metadata['last_earnings_filing_date'] = filing['last_earnings_filing_date']
            text = f"# 上次财报发布8-K申报日：{filing['last_earnings_filing_date']}（Item 2.02）\n" + text
    annual_covered = False
    if quarterly and filing and filing['form'] == '10-K' and latest_period < expected and kind != 'balance_sheet':
        # 同一报表自己的年度事实才证明已收录10-K；绝不借无关tag免陈旧。
        annual_periods = []
        for _, tags in _STATEMENTS[kind]:
            values, _ = _as_of(us_gaap, tags, as_of_date, (_SPANS['annual'],), _ANNUAL_FORMS)
            annual_periods.extend(end for end, _ in values)
        annual_covered = bool(annual_periods and max(annual_periods) >= expected)
        if annual_covered:
            metadata['annual_covered_period'] = expected
            metadata['reason'] = f'年度申报{expected}已收录，第四季度未单列，不推算Q4；季度表体截至{latest_period}'
            text = f"# {metadata['reason']}\n" + text
    age = (date.fromisoformat(as_of_date) - date.fromisoformat(latest_period)).days
    stale = bool(expected and latest_period < expected and not annual_covered) or bool(unavailable and age > 135)
    if stale:
        if expected:
            reason = f"SEC XBRL汇总未收录{filing['form']} {expected}（{filing['filingDate']}提交），表体截至{latest_period}"
            minimum_period = expected
        else:
            reason = f'SEC submissions不可用（{unavailable}）；表体期末{latest_period}距分析日{age}天，超过135天，最新申报无法核验'
            minimum_period = (date.fromisoformat(as_of_date) - timedelta(days=135)).isoformat()
        metadata.update(stale=True, reason=reason)
        raise StaleVendorDataError(reason, original_text=text, statement_metadata=metadata, minimum_period=minimum_period)
    if unavailable:
        metadata['reason'] = f'SEC submissions不可用（{unavailable}）；表体期末{latest_period}距分析日{age}天，未超过135天；最新申报未核验'
        text = f"# {metadata['reason']}\n" + text
    return StatementResult(text, metadata)


def get_balance_sheet(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """Balance sheet as filed on or before ``as_of_date``."""
    return _statement("balance_sheet", ticker, freq, as_of_date, "Balance Sheet")


def get_income_statement(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """Income statement as filed on or before ``as_of_date``.

    A fourth quarter is never derived: filers report it only inside the annual
    figure, and subtracting three separately filed quarters would invent a number
    with no filing date behind it.
    """
    return _statement("income_statement", ticker, freq, as_of_date, "Income Statement")


def get_cashflow(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """Cash flow statement as filed on or before ``as_of_date``."""
    return _statement("cashflow", ticker, freq, as_of_date, "Cash Flow Statement")
