"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, create_model

# LLMs sometimes write a placeholder string ("None", "N/A", ...) into an optional
# numeric field instead of omitting it. Coerce those to None so the structured
# call validates instead of erroring (#1058). Pydantic still parses real numeric
# strings ("189.5") to float.
_NULLISH_FLOAT = {"", "none", "n/a", "na", "null", "nil", "-", "tbd", "unknown"}


def _coerce_optional_float(value):
    """Normalise an LLM-written optional numeric field before validation.

    Three shapes show up in practice: a placeholder string ("None", "N/A") in
    place of an omitted value (#1058); a percentage where a price was asked for
    ("15%", #1288); and a human-formatted price ("$1,234.50"). A percentage
    cannot be salvaged into an absolute level -- reading "15%" as 15 would put a
    stop at $15 on a $600 stock -- so it is dropped like a placeholder, leaving
    one bad field to null out instead of failing the whole proposal. A formatted
    price is reduced to its number.

    Anything that is not a single number is dropped the same way. A range
    ("150-160") or a hedge ("around 150") would otherwise reach pydantic, fail
    validation, and discard the whole decision, losing every field the model got
    right along with the price.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() in _NULLISH_FLOAT or text.endswith("%"):
        return None
    cleaned = text.replace(",", "").lstrip("$€£¥").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(StrEnum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


# 兼容旧导入名称；新代码统一使用五档PortfolioRating。
TraderAction = PortfolioRating


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Conflicting arguments alone are not a "
            "reason to Hold: commit to the stronger side, sized by how "
            "decisively it wins. Choose Hold only when the evidence is still "
            "balanced after weighing, or too thin to support a call."
        ),
    )
    rationale: str = Field(
        description=(
            "≤600字，按决定性论据前3条（注明来源）、被驳回论据及理由、"
            "关键不确定性、复评触发条件四部分输出。"
        ),
    )
    strategic_actions: str = Field(
        description=(
            "交易员可执行的具体步骤，配置以单标的标准仓位100%为参考单位；"
            "研究团队未知真实持仓，不把相对目标当作账户资产比例或现有持仓买卖比例。"
        ),
    )
    target_allocation_pct: float | None = Field(
        default=None, ge=0, description="相对单标的标准仓位100%的目标配置，60表示标准量的六成；依据不足可不提供。",
    )


# 保留旧类本身和schema标题，关闭开关时不把新增字段描述注入旧提示。
LegacyResearchPlan = ResearchPlan


class EvidenceResearchPlan(LegacyResearchPlan):
    """研究经理直接读报告后的引用核对；旧记录允许缺字段。"""
    evidence_check: str | None = Field(default=None, max_length=200,
        description='引用核对≤200字：列出辩手引用与报告不符处、双方遗漏的关键事实；没有时写“无”并列已核对的2–3个数据点。')


class ResearchCrux(BaseModel):
    """裁判列出的决定性分歧点。"""
    bull_claim: str = Field(description='多方观点及来源条目。')
    bear_claim: str = Field(description='空方观点及来源条目。')
    evidence: str = Field(description='对应报告中的决定性证据及来源。')
    winner: str = Field(description='多方、空方或未决。')
    reason: str = Field(description='胜负或未决的具体理由。')


class CruxResearchPlan(LegacyResearchPlan):
    """结构化辩论的分歧裁决；旧记录允许缺字段。"""
    cruxes: list[ResearchCrux] | None = Field(default=None, min_length=3, max_length=5,
        description='先列3–5个分歧点，逐项比较双方主张、报告证据、胜负及理由，再给评级。')


class ResearchPlan(EvidenceResearchPlan):
    """默认研究计划增加引用核对和分歧裁决，兼容旧记录。"""
    cruxes: list[ResearchCrux] | None = Field(default=None, min_length=3, max_length=5,
        description='先列3–5个分歧点，逐项比较双方主张、报告证据、胜负及理由，再给评级。')


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    text = "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
        "",
        f"**目标配置（标准仓位=100%）**: {_allocation_text(plan.target_allocation_pct)}",
    ])
    if hasattr(plan, 'cruxes'):
        rows = ['**分歧点裁决**']
        if plan.cruxes:
            for index, crux in enumerate(plan.cruxes, 1):
                rows.append(f'{index}. 多方：{crux.bull_claim}；空方：{crux.bear_claim}；证据：{crux.evidence}；裁决：{crux.winner}；理由：{crux.reason}')
        else:
            rows.append('未提供；不能视为已裁决')
        text = '\n\n'.join(rows) + '\n\n' + text
    if hasattr(plan, 'evidence_check'):
        text += '\n\n**引用核对**: ' + (plan.evidence_check or '未提供；不能视为已核对')
    return text + _probability_lines(plan)


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


ALLOCATION_INSTRUCTION = (
    "统一配置单位：单标的标准仓位=100%，表示用户为该标的设定的计划持仓量。"
    "target_allocation_pct=60 表示标准量的六成，不是账户总资产60%，也不是卖出现有持仓60%。"
    "目标依证据和风险决定，不把评级固定映射为比例；没有依据可留空。"
    "无真实持仓、标准金额或股数时，不计算买卖数量；不要默认持仓已等于标准量。"
)


def _allocation_text(value: float | None) -> str:
    """保留零配置，缺少目标时明确说明。"""
    return f"{value:g}%" if value is not None else "未提供"


_PRICE_PLAN_BASE = (
    "entry_price/stop_loss 仍填单一数值，entry_plan/add_plan/reduce_plan 填带条件的区间方案。"
    "请同时给出参考价格（报价币种、数据时点及质量）、建仓、加仓和减仓点位。"
    "建仓面向尚未持仓，加仓面向已有仓位；每项包含价格区间、触发条件、失效条件和具体依据。"
    "减仓面向已有仓位，说明减配或止损的触发条件；每项首句固定为「区间 X–Y 美元（依据：具体价位名称）」或「不适用：原因」，供首页展示。"
    "区间必须来自所提供行情或技术报告，不得编造报价或用目标价代替入场位；"
    "上一交易日收盘价必须注明日期，不得称为当时最新价；未核验时段的报价不得称为有效盘前价。"
    "不建议买入或证据不足时仍须明确等待条件/不适用原因，方案不能与最终评级相矛盾。"
)


def price_plan_instruction(config=None) -> str:
    """从当前配置渲染方案规则；不做事后数值校验。"""
    if config is None:
        from tradingagents.dataflows.config import get_config
        config = get_config()
    minimum = config.get("price_plan_stop_atr_min", 1.0)
    normal = config.get("price_plan_stop_atr_normal", (1.5, 2.0))
    maximum = config.get("price_plan_stop_atr_max", 2.5)
    reward = config.get("price_plan_min_reward_risk", 1.5)
    base = _PRICE_PLAN_BASE
    if config.get("rating_timing_decoupled", True):
        base = base.replace("方案不能与最终评级相矛盾。", "点位仅表达执行时机，无合格点位不改变方向评级。")
    text = base + (
        "每个区间必须引用价位锚点、快照或关键价位表中的具体项及日期；方案≤200字。"
        "先把止损放在真实支撑或均线之外，买入按区间上沿（最不利端）计算距离。"
        f"允许止损距离{minimum:g}–{maximum:g}倍ATR，推荐{normal[0]:g}–{normal[1]:g}倍；"
        "低于下限不提供方案，超过上限改为等待回踩或下调配置，不直接给出该区间；"
        "允许范围内但超出推荐区间须说明原因，ATR不可用须说明无法校验。"
        "第一目标取最近的上方阻力，盈亏比=(第一目标−区间上沿)/(区间上沿−止损价)，"
        f"建仓/加仓要求≥{reward:g}；不满足时首句写不适用及原因，再说明等待的入场价。"
        "减仓不受盈亏比门槛约束，可在失效或第一目标附近分批执行。"
        "stop_loss必须与建仓方案失效价一致，不适用时不编造数值。"
    )

    if config.get('price_plan_alt_target', True):
        text += (
            '默认第一目标仍取最近上方阻力。仅当价格距最近上方阻力不足1ATR，或处于60日新高附近且上方无阻力时，'
            '可使用替代第一目标=区间上沿+2×ATR，必须写明“目标方法：ATR替代”。不得把任意无数据阻力当作符合条件。'
            '可选择突破跟随：区间为已知阻力上方0–0.25ATR，须收盘站上阻力才触发，止损在阻力下方1–1.5ATR。'
            f'仍按区间上沿计算总止损距离{minimum:g}–{maximum:g}ATR及盈亏比≥{reward:g}；不满足仍写不适用并等待，不降低门槛。'
        )
    if config.get('rating_timing_decoupled', True):
        text += 'Buy/Overweight没有合格入场点时保持评级；当前无合格入场点或盈亏比不足只影响entry_plan/add_plan。建仓首句写“不适用：等待回踩至 X 或突破 Y 确认”，X和Y均须为输入可核对的具体价位，不能作为改评级理由。缺任一锚点必须说明缺失并等待可靠行情，不编造X/Y。Underweight或Sell不新建仓。'
    return text


# 旧导入仍可使用缺省渲染结果；角色运行时调用函数读取配置。
PRICE_PLAN_INSTRUCTION = price_plan_instruction({})

class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: PortfolioRating = Field(
        description="动作取Buy / Overweight / Hold / Underweight / Sell之一；与研究经理不同须在reasoning解释。",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description=(
            "Optional entry price target as an absolute number in the instrument's "
            "quote currency (e.g. 189.5), never a percentage or a range. Omit it "
            "if you cannot state a specific level."
        ),
    )
    stop_loss: float | None = Field(
        default=None,
        description=(
            "Optional stop-loss as an absolute price in the instrument's quote "
            "currency (e.g. 172.0), never a percentage. Convert a percentage "
            "distance to the price level it implies, or omit it."
        ),
    )
    position_sizing: str | None = Field(
        default=None,
        description="相对单标的标准仓位100%的配置说明；真实持仓未知时不推算买卖比例或数量。",
    )
    target_allocation_pct: float | None = Field(
        default=None, ge=0, description="相对单标的标准仓位100%的目标配置，60表示标准量的六成；依据不足可不提供。",
    )

    reference_price: str | None = Field(
        default=None, description="有依据的参考价格、报价币种、数据日期/时间和质量；没有可靠报价时说明缺失。",
    )
    entry_plan: str | None = Field(
        default=None, description="建仓方案：首句固定为「区间 X–Y 美元（依据：具体价位名称）」或「不适用：原因」；后续写触发、失效条件与依据，不能编造行情；无证据时说明等待条件。",
    )
    add_plan: str | None = Field(
        default=None, description="加仓方案：首句固定为「区间 X–Y 美元（依据：具体价位名称）」或「不适用：原因」；后续写触发、失效条件与依据，不能编造行情；面向已有仓位，不假设持仓数量。",
    )
    reduce_plan: str | None = Field(
        default=None, description="减仓方案：首句固定为「区间 X–Y 美元（依据：具体价位名称）」或「不适用：原因」；后续写触发、失效条件与依据，不能编造行情；面向已有仓位，不假设持仓数量。",
    )

    @field_validator("entry_price", "stop_loss", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


# 旧输出类保留原schema标题及字段描述，用于关闭开关时的生成。
LegacyTraderProposal = TraderProposal


def _validate_direction_change(value):
    """只验证声明格式；不替模型生成证据或事后改动作。"""
    value = value.strip()
    if value == '否' or (value.startswith('是：') and value[2:].strip()):
        return value
    raise ValueError('direction_change必须为“否”或“是：具体新证据”')


class TraderProposal(LegacyTraderProposal):
    """新生成方案必须声明是否依据新证据变更研究方向。"""
    action: PortfolioRating = Field(description='默认沿用研究经理recommendation。只因研究经理未考虑的可核对新证据才可改变，盈亏比或买点不足不构成依据。')
    direction_change: str = Field(description='必填“否”或“是：具体新证据”，变更须注明研究经理未考虑的来源、事实及方向影响；不能仅写盈亏比、买点不足或笼统风险。')
    _direction_format = field_validator('direction_change')(_validate_direction_change)


class CompatibleTraderProposal(LegacyTraderProposal):
    """仅加载旧记录，缺声明显示未提供，不用于新生成schema。"""
    direction_change: str | None = None


def load_trader_proposal(data):
    return CompatibleTraderProposal.model_validate(data) if 'direction_change' not in data else TraderProposal.model_validate(data)


def render_trader_proposal(proposal: TraderProposal) -> str:
    """渲染五档动作，保留FINAL TRANSACTION PROPOSAL行及旧三档兼容。"""
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if hasattr(proposal, 'direction_change'):
        parts.extend(['', '**方向变更**: ' + (proposal.direction_change or '未提供；旧记录未声明')])
    # Named even when absent, so a reader can tell a level the trader chose not
    # to give from one the schema never asked for.
    for label, value in (("Entry Price", proposal.entry_price),
                         ("Stop Loss", proposal.stop_loss),
                         ("Position Sizing", proposal.position_sizing)):
        parts.extend(["", f"**{label}**: {value if value is not None and value != '' else 'not provided'}"])
    for label, value in (("参考价格与时点", proposal.reference_price),
                         ("建仓点位", proposal.entry_plan), ("加仓点位", proposal.add_plan),
                         ("减仓点位", proposal.reduce_plan)):
        parts.extend(["", f"**{label}**: {value or '未提供；等待可靠行情及条件确认'}"])
    parts.extend(["", f"**目标配置（标准仓位=100%）**: {_allocation_text(proposal.target_allocation_pct)}"])
    if hasattr(proposal, "first_target"):
        parts.extend(["", f"**First Target**: {proposal.first_target if proposal.first_target is not None else 'not provided'}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate. "
            "Conflicting arguments alone are not a reason to Hold: commit to the "
            "stronger side, sized by how decisively it wins. Choose Hold only "
            "when the evidence is still balanced after weighing, or too thin to "
            "support a call."
        ),
    )
    executive_summary: str = Field(
        description=(
            "执行摘要≤4句，说明方向、目标配置、关键风险和决策周期。"
        ),
    )
    investment_thesis: str = Field(
        description=(
            "投资论点≤800字，引用具体证据及历史教训，说明复评触发；默认沿用交易员点位，修改须给理由。"
        ),
    )
    price_target: float | None = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: str | None = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )
    target_allocation_pct: float | None = Field(
        default=None, ge=0, description="相对单标的标准仓位100%的最终目标配置，60表示标准量的六成；依据不足可不提供。",
    )

    reference_price: str | None = Field(
        default=None, description="有依据的参考价格、报价币种、数据日期/时间和质量；没有可靠报价时说明缺失。",
    )
    entry_plan: str | None = Field(
        default=None, description="建仓方案：首句固定为「区间 X–Y 美元（依据：具体价位名称）」或「不适用：原因」；后续写触发、失效条件与依据，不能编造行情；无证据时说明等待条件。",
    )
    add_plan: str | None = Field(
        default=None, description="加仓方案：首句固定为「区间 X–Y 美元（依据：具体价位名称）」或「不适用：原因」；后续写触发、失效条件与依据，不能编造行情；面向已有仓位，不假设持仓数量。",
    )
    reduce_plan: str | None = Field(
        default=None, description="减仓方案：首句固定为「区间 X–Y 美元（依据：具体价位名称）」或「不适用：原因」；后续写触发、失效条件与依据，不能编造行情；面向已有仓位，不假设持仓数量。",
    )

    @field_validator("price_target", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


LegacyPortfolioDecision = PortfolioDecision


class PortfolioDecision(LegacyPortfolioDecision):
    """新组合决定默认锚定研究方向，变更须披露新证据。"""
    time_horizon: str | None = Field(default=None, description='决策方向周期为未来5–20交易日；点位有效期另依输入，不写3–6个月。')
    rating: PortfolioRating = Field(description='默认沿用研究经理recommendation，而非审阅意见多数或交易员改后评级；只有研究经理未考虑的可核对新信息才允许变更。')
    direction_change: str = Field(description='必填“否”或“是：具体新证据”，列late news/新增数据或审阅人新事实的来源及方向影响；盈亏比、入场点或笼统风险不构成变更依据。')
    _direction_format = field_validator('direction_change')(_validate_direction_change)


class CompatiblePortfolioDecision(LegacyPortfolioDecision):
    """旧结果读取专用；没有字段不默认为否。"""
    direction_change: str | None = None


def load_portfolio_decision(data):
    return CompatiblePortfolioDecision.model_validate(data) if 'direction_change' not in data else PortfolioDecision.model_validate(data)


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if hasattr(decision, 'direction_change'):
        parts.extend(['', '**方向变更**: ' + (decision.direction_change or '未提供；旧记录未声明')])
    for label, value in (("参考价格与时点", decision.reference_price),
                         ("建仓点位", decision.entry_plan), ("加仓点位", decision.add_plan),
                         ("减仓点位", decision.reduce_plan)):
        parts.extend(["", f"**{label}**: {value or '未提供；等待可靠行情及条件确认'}"])
    parts.extend(["", f"**目标配置（标准仓位=100%）**: {_allocation_text(decision.target_allocation_pct)}"])
    # Named even when absent: a missing line reads as a field nobody asked for,
    # so a reader cannot tell "no target" from "target not reported".
    for label, field in (("Stop Loss", "stop_loss"), ("First Target", "first_target")):
        if hasattr(decision, field):
            parts.extend(["", f"**{label}**: {getattr(decision, field) if getattr(decision, field) is not None else 'not provided'}"])
    target = decision.price_target if decision.price_target is not None else "not provided"
    parts.extend(["", f"**Price Target**: {target}"])
    parts.extend(["", f"**Time Horizon**: {decision.time_horizon or 'not provided'}"])
    return "\n".join(parts) + _probability_lines(decision)


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(StrEnum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Replaces the previous free-form prose output so downstream consumers
    (dashboards, audit logs, PDF renderers, other agents) can read
    ``overall_band`` and ``overall_score`` without maintaining fragile regex
    fallbacks that drift with every model release. ``narrative`` preserves the
    rich source-by-source analysis; ``render_sentiment_report`` prepends a
    deterministic header so the saved report stays human-readable.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "Guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "按数据质量与样本量选low/medium/high；StockTwits和Reddit均无可用本标的观点时必须low。"
        ),
    )
    narrative: str = Field(
        description=(
            "narrative≤1000字，分列新闻语气与社交情绪，引用样本与来源，说明分歧、主题、催化与风险；"
            "无可用社交观点时首段写本评分仅反映新闻语气。"
        ),
    )


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to the markdown shape the rest of the system expects.

    The structured header (band + score + confidence) is prepended to the
    narrative so the saved report is both human-readable and machine-parseable
    without regex.
    """
    return "\n".join([
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])



_SCHEMA_BASES = {}


def _probability_value(value):
    """概率缺失不默认0.5；有效数值必须有限且位于0–1。"""
    import math
    if isinstance(value,str):
        value=value.strip()
        if value.lower() in _NULLISH_FLOAT:
            return None
    if value is None:
        return None
    try:
        parsed=float(value)
    except (ValueError,TypeError):
        return str(value)
    if not math.isfinite(parsed) or not 0<=parsed<=1:
        raise ValueError('概率必须为0–1有限数值或明确缺失文本')
    return parsed


from functools import lru_cache

@lru_cache(maxsize=64)
def _extended_decision_schema(base, prices, probabilities, layer):
    """缓存schema类以保持相同开关下工具签名稳定。"""
    fields={};validators={}
    if prices and layer!='rm':
        fields['first_target']=(float | None, Field(default=None, description='第一目标的真实数值；无依据不填，不把price_target当第一目标。'))
        numeric=['first_target']
        if layer=='pm':
            fields['stop_loss']=(float | None, Field(default=None, description='最终方案止损绝对价格，与点位失效价一致；无依据不填。'))
            numeric.append('stop_loss')
        validators['_optional_prices']=field_validator(*numeric,mode='before')(classmethod(lambda cls,value:_coerce_optional_float(value)))
    if probabilities and layer in ('rm','pm'):
        for field in ('prob_outperform_5d','prob_outperform_20d'):
            fields[field]=(float | str | None, Field(default=None, description='对应交易日窗口主口径收益>0的概率0–1；未知可用明确文字，不默认0.5。'))
        fields['expected_return_20d_range']=(str | None,Field(default=None,description='20交易日主口径预期收益区间，例如−2% ~ +5%；证据不足不编造。'))
        validators['_probabilities']=field_validator('prob_outperform_5d','prob_outperform_20d',mode='before')(classmethod(lambda cls,value:_probability_value(value)))
    if not fields:
        return base
    return create_model(base.__name__+'OptionalFields',__base__=base,__validators__=validators,**fields)


def decision_schema(base, config, layer):
    """每项关闭恢复旧字段；不顺带关闭其他功能。"""
    original=_SCHEMA_BASES.get(base,base)
    return _extended_decision_schema(original,config.get('price_plan_evaluation_enabled',True),config.get('rating_probability_fields',True),layer)


def _probability_lines(model):
    """仅扩展schema实例渲染新增可选字段。"""
    if not hasattr(model,'prob_outperform_20d'):
        return ''
    return ''.join('\n\n**'+label+'**: '+str(getattr(model,field) if getattr(model,field) is not None else '未提供')
        for label,field in [('Prob Outperform 5d','prob_outperform_5d'),('Prob Outperform 20d','prob_outperform_20d'),('Expected Return 20d Range','expected_return_20d_range')])


# 公共新schema包含可选字段，关闭配置仍选其原类；旧加载不要求新增字段。
for _name,_layer in [('ResearchPlan','rm'),('TraderProposal','trader'),('PortfolioDecision','pm')]:
    _base=globals()[_name]
    _new=decision_schema(_base,{},_layer)
    _SCHEMA_BASES[_new]=_base
    globals()[_name]=_new
