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
    '评级只表达未来5–20个交易日的方向及相对基准超额判断，点位方案只表达执行时机。'
    '当前无合格入场点或盈亏比不足仅写入entry_plan/add_plan首句“不适用：等待…”及具体价位，不能作为调整评级理由。'
)
RATING_DEFINITIONS = """五档评级定义（未来5–20交易日的方向及相对基准超额判断）：
- **Buy**：明确看多，预期显著正向超额。
- **Overweight**：偏多，预期正向超额。
- **Hold**：方向与超额判断中性、证据均衡或不足。
- **Underweight**：偏空，预期负向超额。
- **Sell**：明确看空，预期显著负向超额。
观点有分歧本身不是选择Hold的理由；权衡证据后仍均衡或证据不足才使用Hold。""" + '\n' + RATING_TIMING_INSTRUCTION


def rating_definitions(config=None):
    """按图私有配置选定义，不改变五档枚举和解析口径。"""
    if config is None:
        from tradingagents.dataflows.config import get_config
        config = get_config()
    return RATING_DEFINITIONS if config.get('rating_timing_decoupled', True) else LEGACY_RATING_DEFINITIONS


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
