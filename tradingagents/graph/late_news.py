"""决策节点前补抓带发布时间的新增新闻，不重跑上游角色。"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import json
import re
from tradingagents.dataflows.config import get_config


def current_time():
    clock = get_config().get('_late_news_clock')
    return clock() if clock else datetime.now(timezone.utc)


def parse_time(value):
    if not value:
        return None
    try:
        text = str(value)
        if re.fullmatch(r'\d{8}T\d{6}', text):
            return datetime.strptime(text, '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
        result = datetime.fromisoformat(text.replace('Z', '+00:00'))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result
    except (TypeError, ValueError):
        return None


def articles_from_response(value):
    """解析现有Alpaca/Yahoo文本与Alpha Vantage JSON，忽略无发布时间的条目。"""
    if isinstance(value, str):
        try: raw = json.loads(value)
        except ValueError: raw = None
    else: raw = value
    if isinstance(raw, dict):
        rows = raw.get('feed', raw.get('articles', []))
        return [{'title':row.get('headline', row.get('title','')), 'summary':row.get('summary',''),
                 'published_at':row.get('created_at', row.get('time_published')), 'source':row.get('source',''),
                 'url':row.get('url','')} for row in rows]
    rows = []
    for part in re.split(r'(?m)^### ', str(value))[1:]:
        header, _, body = part.partition('\n')
        match = re.match(r'(.*?) \(source: (.*?), created_at: ([^)]+)\)$', header)
        if not match: continue
        link = re.search(r'(?m)^Link: (.+)$', body)
        rows.append({'title':match[1], 'source':match[2], 'published_at':match[3],
                     'summary': re.sub(r'(?m)^Link: .*$', '', body).strip(), 'url':link[1] if link else ''})
    return rows


def refresh_news(state, stage, config=None):
    settings = get_config()
    if not settings.get('late_news_refresh') or settings.get('news_cutoff_utc'):
        return {}
    previous = parse_time(state.get('news_last_fetched_at') or settings.get('_late_news_initial_as_of'))
    if previous is None: return {}
    started = current_time()
    stamp = started.isoformat()
    update = {'news_last_fetched_at':stamp}
    try:
        from tradingagents.agents.tools import get_news
        response = get_news.invoke({'ticker':state['company_of_interest'], 'start_date':state['trade_date'],
                                    'end_date':state['trade_date'], 'trade_date':state['trade_date']}, config=config)
        finished = current_time()
        existing = list(state.get('late_news', []))
        urls = {row['url'] for row in existing if row.get('url')}
        titles = {row['title'].casefold() for row in existing}
        fresh = []
        rows = sorted(articles_from_response(response), key=lambda row:parse_time(row['published_at']) or datetime.min.replace(tzinfo=timezone.utc))
        for row in rows:
            published = parse_time(row['published_at'])
            title = row['title'].casefold()
            if not published or not previous < published <= finished: continue
            if title in titles or (row['url'] and row['url'] in urls): continue
            fresh.append({**row, 'stage':stage, 'fetched_at':finished.isoformat()})
            titles.add(title)
            if row['url']: urls.add(row['url'])
            if len(existing) + len(fresh) >= 10: break
        update['late_news'] = (existing + fresh)[:10]
    except Exception as exc:
        update['late_news_errors'] = [*state.get('late_news_errors', []),
                                      f'{stage}阶段补抓失败：{type(exc).__name__}']
        # 失败保留原查询起点，下一节点仍可补抓遗漏窗口。
        update.pop('news_last_fetched_at', None)
    return update


def refresh_macro(state, stage):
    """决策节点前重新取当日经济数据标题，按标题去重。

    数据源存在收录延迟：发布时间早于上次查询的标题可能在之后才可见，因此不按
    发布时间与上次查询比较，而是与初始上下文及已纳入的标题去重。取数函数由调用
    项目通过 ``_late_macro_refresher`` 提供，返回带 ``title`` 的字典列表。
    """
    settings = get_config()
    refresher = settings.get('_late_macro_refresher')
    if not settings.get('late_news_refresh') or settings.get('news_cutoff_utc') or not callable(refresher):
        return {}
    existing = list(state.get('late_macro', []))
    seen = {str(title).casefold() for title in settings.get('_late_macro_seen', [])}
    seen |= {row['title'].casefold() for row in existing}
    try:
        rows = refresher()
        fetched = current_time().isoformat()
        fresh = []
        for row in rows:
            title = str(row.get('title', ''))
            if not title or title.casefold() in seen:
                continue
            fresh.append({**row, 'stage': stage, 'fetched_at': fetched})
            seen.add(title.casefold())
        return {'late_macro': (existing + fresh)[:20]}
    except Exception as exc:
        return {'late_macro_errors': [*state.get('late_macro_errors', []),
                                      f'{stage}阶段补抓经济数据失败：{type(exc).__name__}']}


def with_news_refresh(node, stage):
    def wrapped(state, config):
        update = {**refresh_news(state, stage, config), **refresh_macro(state, stage)}
        return {**update, **node({**state, **update})}
    return wrapped


def render_late_macro(state):
    rows, errors = state.get('late_macro', []), state.get('late_macro_errors', [])
    if not rows and not errors: return ''
    lines = ['\n\n## 分析期间新增经济数据（数据源收录后补抓）']
    for row in rows:
        label = '组合经理阶段补抓，上游未评估' if row.get('stage') == 'portfolio' else '研究阶段补抓'
        if row.get('actual'):
            detail = f"实际 {row['actual']}，预期 {row.get('estimate') or '—'}，前值 {row.get('prior') or '—'}"
            lines.append(f"- {row.get('created_at') or '时间未知'} · {row.get('name') or row['title']}：{detail}（{label}）")
        else:
            lines.append(f"- {row.get('created_at') or '时间未知'} · {row['title']}（{label}）")
    if any(row.get('stage') == 'portfolio' for row in rows):
        lines.append('组合经理须逐条说明上游未评估的经济数据是否改变评级、目标配置或点位及理由；需要重新辩论时写“建议重跑”，不得声称上游已评估。')
    if errors: lines.extend(['数据限制：', *errors])
    return '\n'.join(lines)


def render_late_news(state):
    macro = render_late_macro(state)
    rows, errors = state.get('late_news', []), state.get('late_news_errors', [])
    if not rows and not errors: return macro
    timestamp = parse_time(state.get('news_last_fetched_at'))
    cutoff = timestamp.astimezone(ZoneInfo('America/New_York')).strftime('%H:%M') if timestamp else '未记录'
    lines = [f'\n\n## 分析期间新增消息（截至 {cutoff} ET）']
    for row in rows:
        lines.append(f"- {row['published_at']} · {row['source']} · {row['title']}\n{row['summary']}\n{row['url']}")
        if row['stage'] == 'portfolio':
            lines.append('仅组合经理阶段纳入：上游未评估。组合经理须逐条说明是否改变评级、目标配置或点位及理由；需要重新辩论时写“建议重跑”，不得声称上游已评估。')
    if errors: lines.extend(['数据限制：', *errors])
    return '\n'.join(lines) + macro
