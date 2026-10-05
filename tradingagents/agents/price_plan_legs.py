"""可选执行腿的软归一与持仓语义；不替模型生成价格。"""
import math
import re
import unicodedata

from pydantic import BaseModel, Field


class BuyLeg(BaseModel):
    """买入条件按腿描述，所有字段可缺失。"""
    kind: str | None = Field(default=None, description='回踩或突破')
    status: str | None = Field(default=None, description='可执行、待触发或仅观察')
    zone_low: float | None = None
    zone_high: float | None = None
    trigger_rule: str | None = Field(default=None, description='触及区间或收盘站上')
    trigger_price: float | None = None
    confirm_days: int | None = Field(default=None, description='收盘连续确认1或2日')
    stop_loss: float | None = None
    stop_anchor: str | None = Field(default=None, description='具名支撑及缓冲，例如P_Low−0.03ATR')
    first_target: float | None = None
    target_method: str | None = Field(default=None, description='阻力位或ATR替代')
    target_anchor: str | None = None
    reason: str | None = Field(default=None, description='≤120字，依据或未通过检查')


class ReduceLeg(BaseModel):
    """超配回落与风险减配分别说明。"""
    kind: str | None = Field(default=None, description='超配回落或风险减配')
    zone_low: float | None = None
    zone_high: float | None = None
    trigger_rule: str | None = Field(default=None, description='进入区间受阻、收盘跌破、立即或其他条件')
    trigger_price: float | None = None
    post_allocation_pct: float | None = Field(default=None, description='风险减配必填减后配置；超配回落为目标配置')
    reason: str | None = None


_ENUMS = {
    'buy_legs': {'kind': {'回踩', '突破'}, 'status': {'可执行', '待触发', '仅观察'},
                 'trigger_rule': {'触及区间', '收盘站上'}, 'target_method': {'阻力位', 'ATR替代'}},
    'reduce_legs': {'kind': {'超配回落', '风险减配'},
                    'trigger_rule': {'进入区间受阻', '收盘跌破', '立即', '其他条件'}},
}
_SYNONYMS = {'pullback': '回踩', 'breakout': '突破', 'executable': '可执行', 'pending': '待触发',
             'observe': '仅观察', '观察': '仅观察', '触及': '触及区间', '收盘突破': '收盘站上',
             '最近阻力': '阻力位', 'atr': 'ATR替代', '超配减仓': '超配回落', '风险减仓': '风险减配'}
_NUMERIC = {'zone_low', 'zone_high', 'trigger_price', 'confirm_days', 'stop_loss', 'first_target', 'post_allocation_pct'}


def soft_direction(value):
    """接受任意标点/空白加证据，异常原文留给软flag。"""
    if not isinstance(value, str):
        return '' if value is None else str(value)
    stripped = value.strip()
    if stripped.startswith('否'):
        return '否'
    if stripped.startswith('是'):
        tail = stripped[1:]
        if tail and (tail[0].isspace() or unicodedata.category(tail[0]).startswith('P')):
            evidence = tail.lstrip(' \t\r\n:：,，。.;；—–-（(“”"')
            if evidence.strip('）) \t\r\n'):
                return '是：' + evidence.strip()
    return value


def normalize_legs(payload):
    """错误字段置空、倒置/多腿保留并诊断，永不因新字段抛错。"""
    if not isinstance(payload, dict):
        return payload
    result = dict(payload)
    flags = []
    for name, enums in _ENUMS.items():
        values = result.get(name)
        if values is None:
            continue
        if not isinstance(values, list):
            flags.append(name + ':invalid_list')
            result[name] = None
            continue
        if len(values) > 2:
            flags.append(name + ':too_many_legs')
        legs = []
        model = BuyLeg if name == 'buy_legs' else ReduceLeg
        for index, value in enumerate(values):
            prefix = f'{name}.{index}'
            if isinstance(value, model):
                value = value.model_dump()
            if not isinstance(value, dict):
                flags.append(prefix + ':invalid_leg')
                continue
            leg = {}
            for field in model.model_fields:
                raw = value.get(field)
                normalized = raw
                if raw is not None:
                    if field in _NUMERIC:
                        try:
                            normalized = float(raw) if not isinstance(raw, bool) else None
                            if normalized is None or not math.isfinite(normalized):
                                raise ValueError()
                            if field == 'confirm_days':
                                if normalized not in (1, 2):
                                    raise ValueError()
                                normalized = int(normalized)
                        except (ValueError, TypeError, OverflowError):
                            normalized = None
                    elif field in enums:
                        normalized = _SYNONYMS.get(raw.strip().lower(), raw.strip()) if isinstance(raw, str) else None
                        if normalized not in enums[field]:
                            normalized = None
                    elif not isinstance(raw, str):
                        normalized = None
                    if normalized is None:
                        flags.append(prefix + '.' + field + ':invalid_value')
                leg[field] = normalized
            if leg.get('zone_low') is not None and leg.get('zone_high') is not None and leg['zone_low'] > leg['zone_high']:
                flags.append(prefix + ':zone_inverted')
            if len(leg.get('reason') or '') > 120 and name == 'buy_legs':
                flags.append(prefix + ':reason_long')
            legs.append(leg)
        result[name] = legs
    if 'direction_change' in result:
        raw = result['direction_change']
        normalized = soft_direction(raw)
        if normalized != '否' and not (normalized.startswith('是：') and any(unicodedata.category(char)[0] not in ('P', 'Z', 'C') for char in normalized[2:])):
            flags.append('direction_change:missing_evidence' if str(raw).strip() == '是' else 'direction_change:invalid_format')
        result['direction_change'] = normalized
    result['leg_validation_flags'] = flags
    return result


LEGS_INSTRUCTION = (
    '建仓=无仓；加仓=已持有且低于目标T，仅补至目标差额，价格相同时写“同建仓”，不得为了与建仓不同而另造价位。'
    '与目标相差小于配置容差视为达标、不动；超配回落只减超出部分并减回T。'
    '风险减配必须具名失效位、减后配置并复评；Buy/Overweight目标内持仓不得仅因阻力受阻减仓。'
    'Underweight/Sell不新建仓、不加仓，目标为上限。'
    'Buy/Overweight没有可执行或待触发的合格买入腿时保持评级；只列真实存在的分支，各写状态；'
    '不合格的写“仅观察”并列出未通过的检查；不得强制同时给回踩和突破。'
    '买入腿buy_legs状态为可执行（触及区间）、待触发（收盘站上，连续1–2日确认后执行）、仅观察（只复评）；'
    '减仓腿reduce_legs类别为超配回落、风险减配。每腿独立填写区间/触发/止损具名锚点/第一目标方法锚点及依据。'
    '三段每段≤200字，首句：可执行“区间 X–Y 美元（依据：…）”；待触发“待触发：收盘站上 Z 后 区间 X–Y 美元（依据：…）”；'
    '观察“不适用：仅观察 X（原因）”或“不适用：无合格买点（原因）”；评级限制“不适用：评级 Underweight 不新建仓”；'
    '同价加仓“同建仓：…（仅补至目标差额）”；减仓“超配回落：区间 X–Y 美元（依据：…）”或“风险减配：收盘跌破 S 降至 P%”。'
)


def render_legs(model):
    """旧记录没有非空腿时不新增执行段。"""
    buy = getattr(model, 'buy_legs', None) or []
    reduce = getattr(model, 'reduce_legs', None) or []
    if not buy and not reduce:
        return ''
    def val(value):
        return f'{value:g}' if isinstance(value, (int, float)) else str(value or '未提供')
    lines = ['', '', '**执行条件**']
    for leg in buy:
        zone = val(leg.zone_low) + '–' + val(leg.zone_high)
        trigger = (f'收盘站上{val(leg.trigger_price)}（{val(leg.confirm_days)}日）后 ' if leg.trigger_rule == '收盘站上' else '触及区间 ')
        lines.append(f'- 买入·{val(leg.kind)}｜{val(leg.status)}｜{trigger}{zone}｜止损{val(leg.stop_loss)}（{val(leg.stop_anchor)}）｜目标{val(leg.first_target)}（{val(leg.target_method)}）｜{leg.reason or ""}')
    for leg in reduce:
        lines.append(f'- {val(leg.kind)}｜{val(leg.trigger_rule)}{val(leg.trigger_price)}｜{val(leg.zone_low)}–{val(leg.zone_high)}｜减至{val(leg.post_allocation_pct)}%｜{leg.reason or ""}')
    return '\n'.join(lines)
