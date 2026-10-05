"""当前一致预期快照；历史回放不能使用今日的预期与修正。"""
from datetime import datetime, timezone
import math

import yfinance as yf

from tradingagents.dataflows.date_window import is_historical
from tradingagents.dataflows.vendor_observer import report_vendor

class ExpectationsResult(str):
    """来源状态与正文同时保留，避免外层观察者把缺值当成功。"""
    def __new__(cls, text, outcome, reason=None):
        value=super().__new__(cls,text)
        value.source_outcome=outcome
        value.source_reason=reason
        return value


FIELDS = ('earnings_estimate', 'revenue_estimate', 'eps_trend', 'eps_revisions', 'earnings_history', 'earnings_dates')


def number(value):
    """只接受来源中的有限数值，不补造缺失项。"""
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _cell(value):
    value = number(value)
    return f'{value:g}' if value is not None else '不可得'


def _value(frame, index, column):
    try:
        return frame.loc[index, column]
    except (KeyError, AttributeError, TypeError):
        return None


def eps_change(actual, base):
    """绝对基数避免负分母翻向；近零仅给美元差额。"""
    actual,base=number(actual),number(base)
    if actual is None or base is None:return '不可计算'
    difference=actual-base
    if abs(base)<0.05:return f'{difference:+.5g}美元（基数过小，百分比不可比）'
    return f'{difference/abs(base)*100:+.2f}%'


def render_expectations(frames, *, queried_at, errors=None):
    """把实际字段渲染成表，历史趋势只是当前快照中的回看值。"""
    lines = [f'### 一致预期（yfinance 当前快照；查询于 {queried_at}）',
             '非历史时点数据库；更新频率及数据延迟未获保证。', '',
             '|项目|本季|下季|', '|---|---|---|']
    for label, field, column in [('EPS均值','earnings_estimate','avg'), ('EPS分析师数','earnings_estimate','numberOfAnalysts'),
                                 ('营收均值','revenue_estimate','avg'), ('营收分析师数','revenue_estimate','numberOfAnalysts')]:
        lines.append('|' + '|'.join([label, *[_cell(_value(frames.get(field), period, column)) for period in ('0q','+1q')]]) + '|')
    for days in (7,30,60,90):
        cells=[]
        for period in ('0q','+1q'):
            previous=number(_value(frames.get('eps_trend'),period,f'{days}daysAgo'))
            current=number(_value(frames.get('eps_trend'),period,'current'))
            cells.append(f'{_cell(previous)}；变化{eps_change(current,previous)}')
        lines.append('|'+'|'.join([f'{days}天前EPS及变化',*cells])+'|')
    for label,column in [('近30天上调数','upLast30days'),('近30天下调数','downLast30days')]:
        lines.append('|'+'|'.join([label,*[_cell(_value(frames.get('eps_revisions'),period,column)) for period in ('0q','+1q')]])+'|')
    lines += ['', '变化%=(当前EPS−过去EPS)/|过去EPS|×100；基数绝对值<0.05仅报美元差额，缺值不计算。', '', '|财报季度|实际EPS|预期EPS|惊喜%／美元差额（实际−预期，绝对基数）|','|---|---|---|---|']
    history=frames.get('earnings_history')
    if history is not None and not history.empty:
        for index,row in history.sort_index().tail(4).iterrows():
            surprise=eps_change(row.get('epsActual'),row.get('epsEstimate'))
            lines.append('|'+'|'.join([str(index),_cell(row.get('epsActual')),_cell(row.get('epsEstimate')),surprise])+'|')
    else:
        lines.append('|不可得|不可得|不可得|不可得|')
    dates=frames.get('earnings_dates'); next_date=None
    if dates is not None:
        try:
            now=datetime.fromisoformat(queried_at)
            future=[stamp for stamp in dates.index if stamp.to_pydatetime().astimezone(timezone.utc)>=now]
            next_date=str(min(future)) if future else None
        except (TypeError,ValueError,AttributeError):
            pass
    lines.append(f'下次财报日：{next_date or "不可得（当前来源未提供未来日期）"}')
    if errors:
        lines.append('字段不可得原因：'+'；'.join(f'{name}：{reason}' for name,reason in errors.items()))
    return '\n'.join(lines)


def get_earnings_expectations(ticker, curr_date):
    """查询六组真实字段，单组失败不伪造其余值。"""
    from tradingagents.dataflows.config import get_config
    config=get_config()
    if config.get("analysis_mode") == "backfill" or (config.get("news_cutoff_utc") and config.get("analysis_mode") != "live") or is_historical(curr_date):
        reason='回放不可用：来源只提供当前值'
        report_vendor('get_earnings_expectations','yfinance','no_data',error=reason,symbol=ticker)
        return ExpectationsResult(reason, 'no_data', reason)
    obj=yf.Ticker(ticker); frames={}; errors={}
    for field in FIELDS:
        try:
            value=getattr(obj,field)
            frames[field]=value
            if value is None or value.empty:
                errors[field]='来源为空'
        except Exception as exc:
            errors[field]=type(exc).__name__
    stamp=datetime.now(timezone.utc).isoformat()
    report_vendor('get_earnings_expectations','yfinance','success' if len(errors)<len(FIELDS) else 'no_data',
                  error='；'.join(f'{key}：{value}' for key,value in errors.items()) or None,symbol=ticker)
    return ExpectationsResult(render_expectations(frames,queried_at=stamp,errors=errors), "success" if len(errors)<len(FIELDS) else "no_data", "；".join(f"{key}：{value}" for key,value in errors.items()) or None)
