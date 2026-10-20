"""每次标的运行独立的有界新闻预算，不把可变状态放入deepcopy配置。"""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
import re
import threading
import unicodedata
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from zoneinfo import ZoneInfo

_STAGE = ContextVar('bounded_news_stage', default=None)
_STATES = {}
_LOCK = threading.RLock()
ET = ZoneInfo('America/New_York')


def enabled(config):
    return config.get('stock_news_policy') == 'bounded-stock-news-v1' and config.get('_analysis_market') == 'US' and config.get('_analysis_type') == 'stock'


@contextmanager
def stage_context(stage, previous, started):
    token = _STAGE.set((stage, previous, started))
    try:
        yield
    finally:
        _STAGE.reset(token)


def stamp(value):
    try:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return result.astimezone(timezone.utc) if result.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


def audit_for(run_id, symbol):
    with _LOCK:
        state = _STATES.get((run_id, symbol))
        return copy.deepcopy(state['audit']) if state else None


def identity(row):
    if row.get('id') is not None:
        return 'id:' + str(row['id'])
    if row.get('url'):
        url = urlsplit(str(row['url']))
        query = [(k, v) for k, v in parse_qsl(url.query) if not k.lower().startswith('utm_') and k.lower() not in ('fbclid', 'gclid')]
        return 'url:' + urlunsplit((url.scheme, url.netloc, url.path, urlencode(query), ''))
    return 'text:' + str((row.get('source'), unicodedata.normalize('NFKC', str(row.get('headline', ''))).casefold().strip(), row.get('created_at')))


def read(ticker, start, nominal_end, config, client):
    """initial只查询一次三段，两个补抓阶段分别独占一次HTTP。"""
    stage_info = _STAGE.get()
    stage, previous, started = stage_info if stage_info else ('initial', None, None)
    clock = config.get('_late_news_clock') or config.get('_daily_live_information_clock')
    started = started or (clock() if callable(clock) else datetime.now(timezone.utc))
    if started.tzinfo is None:
        raise ValueError('新闻请求时刻必须带时区')
    started = started.astimezone(timezone.utc)
    start, nominal_end = start.astimezone(timezone.utc), nominal_end.astimezone(timezone.utc)
    cutoff = stamp(config.get('news_cutoff_utc'))
    end = min([nominal_end, started] + ([cutoff] if cutoff else []))
    key = (config.get('_daily_run_id'), config.get('_analysis_symbol') or ticker)
    with _LOCK:
        state = _STATES.setdefault(key, {'responses': {}, 'articles': {}, 'late_count': 0,
            'audit': {'policy': 'bounded-stock-news-v1', 'stages': {}, 'http_attempts': 0,
                'legacy_prompt_bytes': 'NOT_MEASURABLE', 'new_prompt_bytes': 0, 'module_path': __file__, 'pid': os.getpid()}})
        if stage in state['responses']:
            if stage == 'initial':
                cached_meta = state['audit']['stages'][stage]
                if start.isoformat() != cached_meta['effective_start'] or end.isoformat() != cached_meta['effective_end']:
                    selected = state.get('rendered', {}).get(stage, [])
                    cached = [text for row, text in selected if (published := stamp(row.get('created_at'))) and start <= published < end]
                    return '\n'.join(cached) + '\n缓存子窗口筛选；未重新请求供应商，缓存外覆盖未知。'
            return state['responses'][stage]
        if stage != 'initial' and previous:
            start = max(start, previous.astimezone(timezone.utc))
        meta = {'stage': stage, 'request_started_at': started.isoformat(), 'nominal_end': nominal_end.isoformat(),
            'effective_start': start.isoformat(), 'effective_end': end.isoformat(),
            'end_truncated_by': 'request_started_at' if end == started else 'cutoff' if cutoff and end == cutoff else None,
            'segments': [], 'raw_count': 0, 'visibility_excluded': 0, 'budget_dropped': 0, 'revised_count': 0}
        state['audit']['stages'][stage] = meta
        pieces = []
        if end > start:
            slots = 3 if stage == 'initial' else 1
            width = (end - start) / slots
            pieces = [(start + width * i, start + width * (i + 1)) for i in range(slots)]
        rows = []
        for lo, hi in pieces:
            segment = {'start': lo.isoformat(), 'end': hi.isoformat(), 'status': 'pending'}
            meta['segments'].append(segment)
            state['audit']['http_attempts'] += 1
            try:
                result = client.get_news([ticker], lo, hi, limit=50, max_pages=1, single_attempt=True)
                raw_rows = result.get('articles', [])
                segment.update(status='success', raw_count=min(50, len(raw_rows)), saturated=bool(result.get('truncated')),
                    next_page_token=result.get('next_page_token'), over_page=bool(result.get('over_page') or len(raw_rows) > 50))
                meta['raw_count'] += min(50, len(raw_rows))
                for article in raw_rows[:50]:
                    published = stamp(article.get('created_at'))
                    if not published or not lo <= published < hi or (stage != 'initial' and not start < published):
                        meta['visibility_excluded'] += 1
                        continue
                    row = dict(article)
                    row.pop('content', None)
                    updated = stamp(row.get('updated_at'))
                    if updated and updated > end:
                        row.pop('headline', None)
                        row.pop('summary', None)
                        row['revised_after_cutoff'] = True
                        meta['revised_count'] += 1
                    if len(str(row.get('summary', ''))) > 800:
                        row.pop('summary', None)
                        row['summary_omitted'] = True
                    rows.append(row)
            except Exception as exc:
                segment.update(status='failed', error_type=type(exc).__name__)
        unique = {}
        for row in sorted(rows, key=lambda x: (stamp(x.get('updated_at')) or stamp(x['created_at']), identity(x))):
            unique[identity(row)] = row
        meta['deduplicated_count'] = len(unique)
        # initial已呈现的同一版本不再占补抓额度；内容修订仍按截止证据排除。
        if stage != 'initial':
            unique = {k: v for k, v in unique.items() if k not in state['articles']}
        by_day = defaultdict(list)
        for k, row in unique.items():
            by_day[stamp(row['created_at']).astimezone(ET).date().isoformat()].append((k, row))
        for group in by_day.values():
            group.sort(key=lambda x: (stamp(x[1]['created_at']), x[0]), reverse=True)
        ordered = []
        for day in sorted(by_day):
            ordered.append(by_day[day].pop(0))
        ordered.extend(sorted([x for group in by_day.values() for x in group], key=lambda x: (stamp(x[1]['created_at']), x[0]), reverse=True))
        cap = 50 if stage == 'initial' else max(0, 10 - state['late_count'])
        byte_cap = 24000 if stage == 'initial' else 6000
        selected, blocks, rendered, used = [], [], [], 0
        for k, row in ordered:
            text = (f"### {row.get('headline', '截止前标题版本不可证')} (source: {row.get('source', 'Alpaca')}, created_at: {row['created_at']})\n"
                + (str(row.get('summary')) + '\n' if row.get('summary') else '')
                + ('摘要因长度未纳入\n' if row.get('summary_omitted') else '')
                + ('截止后修订；标题与摘要不可用\n' if row.get('revised_after_cutoff') else '')
                + (f"Link: {row['url']}\n" if row.get('url') else ''))
            size = len(text.encode())
            if len(selected) >= cap or used + size > byte_cap - 1024:
                meta['budget_dropped'] += 1
                continue
            used += size
            blocks.append(text)
            rendered.append((row, text))
            selected.append((k, row))
        successful = sum(s['status'] == 'success' for s in meta['segments'])
        failed = any(s['status'] == 'failed' for s in meta['segments'])
        partial = any(s.get('saturated') or s.get('next_page_token') or s.get('over_page') for s in meta['segments']) or meta['visibility_excluded'] or meta['revised_count']
        status = ('not_applicable' if not pieces else 'failed' if not successful and failed else
            'budget_exhausted' if meta['budget_dropped'] else 'partial' if failed or partial else 'complete_within_provider')
        obtained = Counter(stamp(row['created_at']).astimezone(ET).date().isoformat() for row in unique.values())
        retained = Counter(stamp(row['created_at']).astimezone(ET).date().isoformat() for _, row in selected)
        days = []
        day = start.astimezone(ET).date()
        while end > start and day <= (end - timedelta(microseconds=1)).astimezone(ET).date():
            days.append(day.isoformat())
            day += timedelta(days=1)
        meta.update(coverage_status=status, selected_count=len(selected), obtained_by_day=dict(obtained), selected_by_day=dict(retained),
            no_return_days=[d for d in days if not obtained[d]], selected_articles=[row for _, row in selected],
            earliest=min((r['created_at'] for r in unique.values()), default=None), latest=max((r['created_at'] for r in unique.values()), default=None))
        notice = f"新闻覆盖：{stage} / {status}；有效窗口 {start.isoformat()} 至 {end.isoformat()}；取得 {len(unique)}、入选 {len(selected)}、预算淘汰 {meta['budget_dropped']}；无返回不等于没有新闻。"
        if partial:
            notice += ' 结果已截断或存在覆盖缺口。'
        result_text = '\n'.join(blocks) + '\n' + notice
        assert len(result_text.encode()) <= byte_cap
        meta['prompt_bytes'] = len(result_text.encode())
        state['audit']['new_prompt_bytes'] += meta['prompt_bytes']
        if stage != 'initial':
            state['late_count'] += len(selected)
        state['articles'].update(selected)
        state.setdefault('rendered', {})[stage] = rendered
        state['responses'][stage] = result_text
        return result_text
