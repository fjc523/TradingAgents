"""Shared 5-tier rating vocabulary and a deterministic heuristic parser.

The same five-tier scale (Buy, Overweight, Hold, Underweight, Sell) is used by:
- The Research Manager (investment plan recommendation)
- The Portfolio Manager (final position decision; its free-text fallback is read here)
- The memory log (rating tag stored alongside each decision entry)

Centralising it here avoids drift between those call sites.

``extract_rating`` returns ``None`` when no rating can be found, and every
caller turns that into ``REVIEW`` rather than a tradeable position: a decision
nobody can read is not a Hold, and a Hold recorded in its place is quoted back to
the next run as a call that was never made (#1170).
"""

from __future__ import annotations

import re
import unicodedata

# Canonical, ordered 5-tier scale (most bullish to most bearish).
RATINGS_5_TIER: tuple[str, ...] = (
    "Buy", "Overweight", "Hold", "Underweight", "Sell",
)

RATING_DEFINITIONS = """五档评级定义（单标的标准仓位=100%，不固定映射比例）：
- **Buy**：决策周期内明确看多，积极建仓或加仓。
- **Overweight**：偏多，逐步提高配置。
- **Hold**：维持现有配置，等待触发条件。
- **Underweight**：偏空，把配置降到目标水平。
- **Sell**：明确看空，清仓或不建仓。
观点有分歧本身不是选择Hold的理由；权衡证据后仍均衡或证据不足才使用Hold。"""

# 关闭方向/时机解耦时保留原定义逐字文本。
LEGACY_RATING_DEFINITIONS = RATING_DEFINITIONS
RATING_TIMING_INSTRUCTION = (
    '评级只表达未来5–20个交易日的方向及主口径收益判断，点位方案只表达执行时机。'
    '当前无合格入场点或盈亏比不足仅写入entry_plan/add_plan首句“不适用：等待…”及具体价位，不能作为调整评级理由。'
)
RATING_DEFINITIONS = """五档评级定义（未来5–20交易日的方向及主口径收益判断）：
- **Buy**：明确看多，预期显著正向主口径收益。
- **Overweight**：偏多，预期正向主口径收益。
- **Hold**：方向与主口径收益判断中性、证据均衡或不足。
- **Underweight**：偏空，预期负向主口径收益。
- **Sell**：明确看空，预期显著负向主口径收益。
观点有分歧本身不是选择Hold的理由；权衡证据后仍均衡或证据不足才使用Hold。""" + '\n' + RATING_TIMING_INSTRUCTION


def rating_definitions(config=None):
    """按图私有配置选定义，不改变五档枚举和解析口径。"""
    if config is None:
        from tradingagents.dataflows.config import get_config
        config = get_config()
    text = RATING_DEFINITIONS if config.get('rating_timing_decoupled', True) else LEGACY_RATING_DEFINITIONS
    if config.get('allocation_bands', DEFAULT_ALLOCATION_BANDS) is not None:
        text = text.replace('不固定映射比例', '配置按allocation_bands区间')
        text += '\n目标配置须满足allocation_bands区间（见统一配置说明），与入场时机分开。'
    if config.get('rating_probability_fields', True):
        text += '\n' + PROBABILITY_INSTRUCTION
    return text


# Signal emitted when the model's decision has no recognizable rating. It is not
# a tradeable position: it flags output that needs a human/re-run rather than
# silently degrading to Hold. Callers that map the signal onto the 5-tier enum
# (e.g. ``PortfolioRating(signal)``) should guard with ``is_review`` first.
RATING_REVIEW = "REVIEW"

_RATING_SET = {r.lower() for r in RATINGS_5_TIER}

# Matches "Rating: X" / "rating - X" / "Rating — **X**" — tolerates markdown
# bold wrappers and any dash or colon a model writes as the separator. "rating"
# must start a word, so "Operating margin: Sell-side" is not a label.
_RATING_LABEL_RE = re.compile(r"(?<![a-z])rating\b[^:\-\u2010-\u2015]*[:\-\u2010-\u2015][\s*]*(\w+)",
                              re.IGNORECASE)

# The same label opening its own line ("**Rating**: X", "## Final Rating - X",
# "Our rating: X"): the shape the Portfolio Manager is asked to write its
# decision in. Only emphasis and heading marks may precede it, so a list item,
# table row or blockquote quoting someone else's rating is not one.
_RATING_LINE_RE = re.compile(
    r"[\s*_#]*(?:\w+\s+)?rating[^\w:\-\u2010-\u2015]*[:\-\u2010-\u2015][\s*]*(\w+)",
    re.IGNORECASE,
)

# A line presenting the scale rather than a decision ("Rating Scale: Buy, ...").
_RATING_SCALE_RE = re.compile(r"rating\s*(scale|options|legend)", re.IGNORECASE)

def extract_rating(text: str) -> str | None:
    """Extract a 5-tier rating from its label, or ``None`` if there is none.

    Reads an explicit "Rating: X" label (tolerant of markdown bold) in the
    NFKC-normalized text, so fullwidth punctuation like ``Rating：Overweight``
    matches as ASCII does: the first one opening its own line, else the last
    one anywhere.
    """
    if not text:
        return None
    norm = unicodedata.normalize("NFKC", text)

    # A decision is asked to open with its rating on its own line, so the first
    # such line is the call; later ones may quote someone else's ("Consensus
    # rating: Buy"). Without one, the last label anywhere wins: prose states its
    # rating after discussing the alternatives. Lines presenting the scale itself
    # are a legend the model echoed, not a call.
    on_own_line = anywhere = None
    for line in norm.splitlines():
        if _RATING_SCALE_RE.search(line):
            continue
        m = _RATING_LINE_RE.match(line)
        if on_own_line is None and m and m.group(1).lower() in _RATING_SET:
            on_own_line = m.group(1).capitalize()
        m = _RATING_LABEL_RE.search(line)
        if m and m.group(1).lower() in _RATING_SET:
            anywhere = m.group(1).capitalize()
    # Without a label there is no call to read: a rating word in the prose may be
    # one the text argues against ("not a Sell"), and reading it reports a
    # direction nobody decided.
    return on_own_line or anywhere


def parse_rating(text: str, default: str = RATING_REVIEW) -> str:
    """Extract a 5-tier rating, or ``REVIEW`` when the decision has none.

    For callers that need a string for every decision, such as the memory log's
    entry tag. The default is the review sentinel, never a tradeable rating.
    """
    rating = extract_rating(text)
    return rating if rating is not None else default


def run_rating(final_state: dict) -> str:
    """A finished run's rating: the Portfolio Manager's own, else read from its decision.

    The fallback serves a state without ``final_rating``, such as a run an older
    version completed and a checkpoint hands back unchanged.
    """
    return final_state.get("final_rating") or parse_rating(final_state.get("final_trade_decision", ""))


def is_review(signal: str) -> bool:
    """Whether a signal is the non-tradeable REVIEW sentinel (#1170)."""
    return signal == RATING_REVIEW


def probability_rating(value):
    """用户确认的20日主口径跑赢概率边界；缺值不归档。"""
    import math
    try:
        probability=float(value)
    except (ValueError,TypeError):
        return None
    if not math.isfinite(probability) or not 0<=probability<=1:
        return None
    return 'Buy' if probability>=.65 else 'Overweight' if probability>=.55 else 'Hold' if probability>=.45 else 'Underweight' if probability>=.35 else 'Sell'


PROBABILITY_INSTRUCTION = ('以P=P(20个交易日主口径收益>0)先估计概率再选评级：Buy P≥0.65；Overweight 0.55≤P<0.65；Hold 0.45≤P<0.55；Underweight 0.35≤P<0.45；Sell P<0.35。'
    '个股/行业ETF主口径为对SPY超额，宽基/指数为绝对收益；概率反映不确定性，证据薄弱向0.5收缩。'
    '缺证据不伪造精度。')


DEFAULT_ALLOCATION_BANDS = {
    'Sell': {'lower':0.,'upper':20.,'lower_inclusive':True,'upper_inclusive':False},
    'Underweight': {'lower':20.,'upper':80.,'lower_inclusive':True,'upper_inclusive':False},
    'Hold': {'lower':80.,'upper':120.,'lower_inclusive':True,'upper_inclusive':True},
    'Overweight': {'lower':120.,'upper':135.,'lower_inclusive':False,'upper_inclusive':True},
    'Buy': {'lower':135.,'upper':150.,'lower_inclusive':False,'upper_inclusive':True},
}


def allocation_rating(value, bands=DEFAULT_ALLOCATION_BANDS):
    """只归档，不修正用户输出；None明确关闭。"""
    import math
    if bands is None:
        return None
    try:
        value=float(value)
    except (TypeError,ValueError):
        return None
    if not math.isfinite(value):
        return None
    for rating,band in bands.items():
        low=value>=band['lower'] if band.get('lower_inclusive',True) else value>band['lower']
        high=value<=band['upper'] if band.get('upper_inclusive',True) else value<band['upper']
        if low and high:
            return rating
    return '越界'


def allocation_instruction(config=None):
    """D3 null仅恢复配置提示，不改变其他开关。"""
    from tradingagents.agents.schemas import ALLOCATION_INSTRUCTION
    if config is None:
        from tradingagents.dataflows.config import get_config
        config=get_config()
    bands=config.get('allocation_bands',DEFAULT_ALLOCATION_BANDS)
    if bands is None:
        return ALLOCATION_INSTRUCTION
    rows=[]
    for rating in RATINGS_5_TIER:
        band=bands[rating]
        rows.append(f"{rating}：{band['lower']:g} {'≤' if band.get('lower_inclusive',True) else '<'} x {'≤' if band.get('upper_inclusive',True) else '<'} {band['upper']:g}%")
    return ALLOCATION_INSTRUCTION.replace('目标依证据和风险决定，不把评级固定映射为比例；没有依据可留空。',
        '配置区间（单标的标准仓位%）：'+'；'.join(rows)+'。证据越强取远离Hold端，财报等风险取接近Hold端；若风险需跨档，应调整评级而非越界。目标为计划完成后量，与等待入场时机分开；没有依据可留空。')


def output_flags(payload, config, *, layer):
    """标记矛盾而不改评级/概率/配置；缺字段不补默认。"""
    rating=payload.get('recommendation') or payload.get('action') or payload.get('rating')
    rating=getattr(rating,'value',rating)
    flags={}
    if layer=='rm':
        evidence=payload.get('evidence_check')
        cruxes=payload.get('cruxes')
        if isinstance(evidence,str) and len(evidence)>200:
            flags['evidence_check_overlength']=True
        if isinstance(cruxes,list) and not 3<=len(cruxes)<=5:
            flags['cruxes_count']=len(cruxes)
    if config.get('rating_probability_fields',True) and layer in ('rm','pm'):
        predicted=probability_rating(payload.get('prob_outperform_20d'))
        flags['rating_prob_mismatch']=bool(predicted is not None and rating is not None and predicted!=rating)
    bands=config.get('allocation_bands',DEFAULT_ALLOCATION_BANDS)
    value=payload.get('target_allocation_pct')
    if bands is not None and value is not None and rating is not None:
        allocated=allocation_rating(value,bands)
        if allocated!=rating:
            band=bands.get(str(rating))
            interval=f"{'[' if band.get('lower_inclusive',True) else '('}{band['lower']:g}, {band['upper']:g}{']' if band.get('upper_inclusive',True) else ')'}" if band else '未知评级'
            flags['allocation_flag']=f'{rating} 配 {value:g}% 不在 {interval} 区间（配置归档：{allocated or "不可解析"}）'
    return flags


def flags_for_text(text,config,*,layer):
    """自由文本只解析明确标签，仍不伪称结构化模型输出。"""
    label='Recommendation' if layer=='rm' else 'Action' if layer=='trader' else 'Rating'
    match=re.search(r'(?mi)^\s*\*\*'+label+r'\*\*\s*[:：]\s*(Buy|Overweight|Hold|Underweight|Sell)\b',str(text or ''))
    payload={'rating':match[1] if match else None}
    allocation=re.search(r'\*\*目标配置（标准仓位=100%）\*\*\s*[:：]\s*([-+]?[0-9]+(?:\.[0-9]+)?)\s*%',str(text or ''))
    if allocation:payload['target_allocation_pct']=float(allocation[1])
    probability=re.search(r'(?mi)^\s*\*\*Prob Outperform 20d\*\*\s*[:：]\s*([01](?:\.\d+)?)\s*$',str(text or ''))
    if probability:payload['prob_outperform_20d']=float(probability[1])
    return output_flags(payload,config,layer=layer)


def direction_flags(state, payload, *, layer, text=None):
    """只比明确评级及方向声明；不改值，缺评级不作确定结论。"""
    def rating(value, label, body):
        candidate=value.get('recommendation') or value.get('action') or value.get('rating') if isinstance(value,dict) else None
        candidate=getattr(candidate,'value',candidate)
        if candidate in RATINGS_5_TIER:return candidate
        match=re.search(r'(?mi)^\s*\*\*'+label+r'\*\*\s*[:：]\s*(Buy|Overweight|Hold|Underweight|Sell)\b',str(body or ''))
        return match[1] if match else None
    current=rating(payload,'Recommendation' if layer=='rm' else 'Action' if layer=='trader' else 'Rating',text)
    reference=current if layer=='rm' else rating(state.get('structured_research_plan'),'Recommendation',state.get('investment_plan'))
    if current is None or reference is None:return {}
    declaration=payload.get('direction_change') if isinstance(payload,dict) else None
    if declaration is None:
        match=re.search(r'(?m)^\s*\*\*方向变更\*\*\s*[:：]\s*([^\n]*(?:\n(?!\s*\*\*)[^\n]*)*)',str(text or ''))
        declaration=match[1].strip() if match else ''
    yes=bool(re.match(r'^是\s*[:：,，]\s*\S',str(declaration or ''),re.DOTALL))
    return {'direction_change_mismatch':current!=reference and not yes}
