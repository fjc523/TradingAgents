"""The memory log: an append-only markdown record of each decision and, once settled, its outcome."""

import re
import json
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from tradingagents.agents.rating import parse_rating
from tradingagents.dataflows.files import locked


class TradingMemoryLog:
    """Append-only markdown log of trading decisions and reflections."""

    # HTML comment: cannot appear in LLM prose output, safe as a hard delimiter
    _SEPARATOR = "\n\n<!-- ENTRY_END -->\n\n"
    # Precompiled patterns — avoids re-compilation on every load_entries() call
    _DECISION_RE = re.compile(r"DECISION:\n(.*?)(?=\nREFLECTION:|\Z)", re.DOTALL)
    _REFLECTION_RE = re.compile(r"REFLECTION:\n(.*?)$", re.DOTALL)

    def __init__(self, config: dict = None):
        cfg = config or {}
        self._log_path = None
        path = cfg.get("memory_log_path")
        if path:
            self._log_path = Path(path).expanduser()
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        # Optional cap on resolved entries. None disables rotation.
        self._max_entries = cfg.get("memory_log_max_entries")
        self._lesson_min = cfg.get("lesson_min_settled_same_ticker", 10)
        self._cross_lessons = cfg.get("cross_ticker_lessons", "stats")
        self._outcomes_path = cfg.get("evaluation_outcomes_path")

    # --- Write: a run records its decision ---

    def store_decision(
        self,
        ticker: str,
        trade_date: str,
        final_trade_decision: str,
        rating: str | None = None,
        *, replace_pending: bool = False,
    ) -> None:
        """Append pending entry at end of propagate(). No LLM call.

        ``rating`` is the decision's own rating when the caller has it; without
        one it is read from the decision text.
        """
        if not self._log_path:
            return
        with locked(self._log_path):
            # 来源默认不允许覆盖；成功live调用者显式授权，仅更新pending。
            rating = rating or parse_rating(final_trade_decision)
            if self._log_path.exists():
                raw = self._log_path.read_text(encoding="utf-8")
                blocks = raw.split(self._SEPARATOR)
                prefix = f"[{trade_date} | {ticker} |"
                matches = [index for index, block in enumerate(blocks)
                           if block.strip().splitlines() and block.strip().splitlines()[0].startswith(prefix)]
                if matches:
                    if not replace_pending or rating not in ('Buy','Overweight','Hold','Underweight','Sell') or not final_trade_decision.strip():
                        return
                    if any(not blocks[index].strip().splitlines()[0].endswith('| pending]') for index in matches):
                        return
                    # settled块原文及分隔符保持；原子替换正文和评级，不新增计数。
                    for index in matches:
                        blocks[index] = f"[{trade_date} | {ticker} | {rating} | pending]\n\nDECISION:\n{final_trade_decision}"
                    updated = self._SEPARATOR.join(blocks)
                    if updated != raw:
                        temporary = self._log_path.with_suffix('.tmp')
                        temporary.write_text(updated, encoding='utf-8')
                        temporary.replace(self._log_path)
                    return
            rating = rating or parse_rating(final_trade_decision)
            tag = f"[{trade_date} | {ticker} | {rating} | pending]"
            entry = f"{tag}\n\nDECISION:\n{final_trade_decision}{self._SEPARATOR}"
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(entry)

    # --- Read ---

    def load_entries(self) -> list[dict]:
        """Parse all entries from log. Returns list of dicts."""
        if not self._log_path or not self._log_path.exists():
            return []
        text = self._log_path.read_text(encoding="utf-8")
        raw_entries = [e.strip() for e in text.split(self._SEPARATOR) if e.strip()]
        entries = []
        for raw in raw_entries:
            parsed = self._parse_entry(raw)
            if parsed:
                entries.append(parsed)
        return entries

    def get_pending_entries(self) -> list[dict]:
        """Return entries with outcome:pending, for settlement."""
        return [e for e in self.load_entries() if e.get("pending")]

    def get_past_context(
        self, ticker: str, n_same: int = 5, n_cross: int = 3, as_of: str | None = None
    ) -> str:
        """按信息截止读取历史；带时区截止保守排除当日仅日期及未知时刻。"""
        if not (self._lesson_min == 0 and self._cross_lessons == "text"):
            return self._factual_context(ticker, n_same=n_same, n_cross=n_cross, as_of=as_of)
        entries = [e for e in self.load_entries() if not e.get("pending")]
        if as_of is not None:
            entries = [e for e in entries if self._known_by(e.get("resolved"), as_of)]
        if not entries:
            return ""

        same, cross = [], []
        for e in reversed(entries):
            if len(same) >= n_same and len(cross) >= n_cross:
                break
            if e["ticker"] == ticker and len(same) < n_same:
                same.append(e)
            elif e["ticker"] != ticker and len(cross) < n_cross:
                cross.append(e)

        if not same and not cross:
            return ""

        parts = []
        if same:
            parts.append(f"Past analyses of {ticker} (most recent first):")
            parts.extend(self._format_full(e) for e in same)
        if cross:
            parts.append("Recent cross-ticker lessons:")
            parts.extend(self._format_reflection_only(e) for e in cross)
        return "\n\n".join(parts)

    @staticmethod
    def _return_number(value):
        """旧标签百分比与C2有限小数均明确读取。"""
        try:
            parsed = float(value.rstrip('%')) / 100 if isinstance(value, str) and value.endswith('%') else float(value)
            return parsed if math.isfinite(parsed) else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _known_by(known, as_of):
        """完整时刻按时区比较；仅日期须等到美东下一日，未知时刻不推定。"""
        if as_of is None:
            return True
        try:
            cutoff_text = str(as_of)
            # 日期调用保留旧接口的日终语义；生产回放传入完整冻结时刻。
            cutoff = (datetime.combine(date.fromisoformat(cutoff_text) + timedelta(days=1), time.min, ZoneInfo("America/New_York"))
                      if len(cutoff_text) == 10 else datetime.fromisoformat(cutoff_text))
            if cutoff.tzinfo is None or not known:
                return False
            known_text = str(known)
            if len(known_text) == 10:
                resolved = datetime.combine(date.fromisoformat(known_text) + timedelta(days=1), time.min, ZoneInfo("America/New_York"))
            else:
                resolved = datetime.fromisoformat(known_text)
                if resolved.tzinfo is None:
                    return False
            if len(cutoff_text) == 10 and len(known_text) != 10:
                return resolved < cutoff
            return resolved <= cutoff
        except (TypeError, ValueError, OverflowError):
            return False

    def _visible_facts(self, as_of):
        """C2主收益优先，旧memory事实保留标识，不改任何落盘。"""
        entries = [dict(entry) for entry in self.load_entries() if not entry.get('pending') and (as_of is None or self._known_by(entry.get('resolved'), as_of))]
        facts = {}
        for entry in entries:
            entry['returns'] = {'5': self._return_number(entry.get('alpha'))}
            entry['basis'] = '旧memory 5日tag收益（可能为超额或绝对，未证明C2口径）'
            facts[(entry['date'], entry['ticker'])] = entry
        if self._outcomes_path and Path(self._outcomes_path).exists():
            try:
                outcomes = [json.loads(line) for line in Path(self._outcomes_path).read_text(encoding='utf-8').splitlines() if line.strip()]
            except (OSError, ValueError):
                # 损坏的独立评估不能变成伪造事实；仍保留旧memory的明确口径。
                outcomes = []
            for row in outcomes:
                if not row.get('is_current'):
                    continue
                visible = {}
                for days, window in row.get('windows', {}).items():
                    known = window.get('settled_at')
                    if window.get('status') == 'settled' and (as_of is None or self._known_by(known, as_of)):
                        value = self._return_number(window.get('primary_return'))
                        if value is not None:
                            visible[str(days)] = value
                if not visible:
                    continue
                key=(row['trade_date'], row['symbol'])
                old=facts.get(key, {})
                same_identity=bool(old.get('decision') and row.get('decision_fingerprint') == hashlib.sha256(old['decision'].encode('utf-8')).hexdigest())
                facts[key]={**old, 'date':row['trade_date'], 'ticker':row['symbol'], 'rating':row.get('ratings',{}).get('pm') or '未提供',
                            'returns':visible, 'basis':'C2 '+str(row.get('primary_metric') or '未提供'), 'reflection':old.get('reflection','') if same_identity else '', 'reflection_association':'精确决策正文一致' if same_identity else '反思关联不可得（C2 current与旧memory决策身份未核验）'}
        return sorted(facts.values(), key=lambda entry: (entry['date'], entry['ticker']))

    def _factual_context(self, ticker, *, n_same, n_cross, as_of):
        """门槛按全量可见样本统计，展示截断不影响n。"""
        entries=self._visible_facts(as_of)
        same=[entry for entry in entries if entry['ticker']==ticker]
        cross=[entry for entry in entries if entry['ticker']!=ticker]
        parts=[]
        if same:
            n=len(same)
            parts.append(f'{ticker}历史事实：样本 {n} 条，'+('不足以形成规律，仅供参考' if n<self._lesson_min else '历史记录仅供参考'))
            table=['|日期|PM评级|5日主收益|10日主收益|20日主收益|口径|','|---|---|---|---|---|---|']
            for entry in reversed(same[-n_same:] if n_same else []):
                values=[f"{entry['returns'][day]:+.2%}" if day in entry['returns'] else '未成熟/不可得' for day in ('5','10','20')]
                table.append('|'+ '|'.join([entry['date'], entry['rating'], *values, entry['basis']])+'|')
            parts.append('\n'.join(table))
            if any(entry.get('reflection_association','').startswith('反思关联不可得') for entry in same):
                parts.append('反思关联不可得：当前C2决策与旧memory无可核验一致身份，不注入旧反思。')
            if n>=self._lesson_min:
                directional=[]
                for entry in same:
                    value=entry['returns'].get('5')
                    sign=1 if entry['rating'] in ('Buy','Overweight') else -1 if entry['rating'] in ('Underweight','Sell') else 0
                    if value is not None and sign:
                        directional.append(value*sign>0)
                parts.append(f'方向命中率（可用5日主口径；Hold不计）：{sum(directional)/len(directional):.1%}，n={len(directional)}' if directional else '方向命中率：没有可用方向样本')
                reflections=[entry for entry in same if entry.get('reflection')][-3:]
                parts.extend(f"[{entry['date']} | {entry['ticker']}]\nREFLECTION:\n{entry['reflection']}" for entry in reversed(reflections))
        if self._cross_lessons=='text':
            legacy_cross=[entry for entry in self.load_entries() if not entry.get('pending') and entry['ticker'] != ticker and (as_of is None or self._known_by(entry.get('resolved'), as_of))]
            if legacy_cross:
                parts.append('Recent cross-ticker lessons:')
                parts.extend(self._format_reflection_only(entry) for entry in reversed(legacy_cross[-n_cross:]) if n_cross)
        elif self._cross_lessons=='stats' and entries:
            # 不混合旧alpha与C2绝对/超额，按评级及明确主口径分组。
            groups=defaultdict(list)
            for entry in entries:
                value=entry['returns'].get('5')
                if value is not None:
                    groups[(entry['rating'],entry['basis'])].append(value)
            table=['全体决策统计（按评级与口径；历史记录仅供参考）：','|PM评级|主口径|5日平均收益|n|','|---|---|---|---|']
            for (rating,basis),values in sorted(groups.items()):
                table.append(f'|{rating}|{basis}|{sum(values)/len(values):+.2%}|{len(values)}|')
            parts.append('\n'.join(table))
        return '\n\n'.join(parts)


    # --- Settle: record a decision's outcome and reflection ---

    def update_with_outcome(
        self,
        ticker: str,
        trade_date: str,
        raw_return: float,
        alpha_return: float,
        holding_days: int,
        reflection: str,
        resolution_date: str | None = None,
    ) -> None:
        """Replace pending tag and append REFLECTION section using atomic write.

        Finds the first pending entry matching (trade_date, ticker), updates
        its tag with return figures (and the ``resolution_date`` the outcome
        became known), and appends a REFLECTION section.  Uses a temp-file +
        os.replace() so a crash mid-write never corrupts the log.
        """
        if not self._log_path or not self._log_path.exists():
            return
        with locked(self._log_path):
            text = self._log_path.read_text(encoding="utf-8")
            blocks = text.split(self._SEPARATOR)

            pending_prefix = f"[{trade_date} | {ticker} |"
            raw_pct = f"{raw_return:+.1%}"
            alpha_pct = f"{alpha_return:+.1%}"

            updated = False
            new_blocks = []
            for block in blocks:
                stripped = block.strip()
                if not stripped:
                    new_blocks.append(block)
                    continue

                lines = stripped.splitlines()
                tag_line = lines[0].strip()

                if (
                    not updated
                    and tag_line.startswith(pending_prefix)
                    and tag_line.endswith("| pending]")
                ):
                    fields = [f.strip() for f in tag_line[1:-1].split("|")]
                    rating = fields[2]
                    new_tag = self._resolved_tag(
                        trade_date, ticker, rating, raw_pct, alpha_pct, holding_days, resolution_date
                    )
                    rest = "\n".join(lines[1:])
                    new_blocks.append(
                        f"{new_tag}\n\n{rest.lstrip()}\n\nREFLECTION:\n{reflection}"
                    )
                    updated = True
                else:
                    new_blocks.append(block)

            if not updated:
                return

            new_blocks = self._apply_rotation(new_blocks)
            new_text = self._SEPARATOR.join(new_blocks)
            tmp_path = self._log_path.with_suffix(".tmp")
            tmp_path.write_text(new_text, encoding="utf-8")
            tmp_path.replace(self._log_path)

    def batch_update_with_outcomes(self, updates: list[dict]) -> None:
        """Apply multiple outcome updates in a single read + atomic write.

        Each element of updates must have keys: ticker, trade_date,
        raw_return, alpha_return, holding_days, reflection.
        """
        if not self._log_path or not self._log_path.exists() or not updates:
            return
        with locked(self._log_path):
            text = self._log_path.read_text(encoding="utf-8")
            blocks = text.split(self._SEPARATOR)

            update_map = {(u["trade_date"], u["ticker"]): u for u in updates}

            new_blocks = []
            for block in blocks:
                stripped = block.strip()
                if not stripped:
                    new_blocks.append(block)
                    continue

                lines = stripped.splitlines()
                tag_line = lines[0].strip()

                matched = False
                for (trade_date, ticker), upd in list(update_map.items()):
                    pending_prefix = f"[{trade_date} | {ticker} |"
                    if tag_line.startswith(pending_prefix) and tag_line.endswith("| pending]"):
                        fields = [f.strip() for f in tag_line[1:-1].split("|")]
                        rating = fields[2]
                        raw_pct = f"{upd['raw_return']:+.1%}"
                        alpha_pct = f"{upd['alpha_return']:+.1%}"
                        new_tag = self._resolved_tag(
                            trade_date, ticker, rating, raw_pct, alpha_pct,
                            upd["holding_days"], upd.get("resolution_date"),
                        )
                        rest = "\n".join(lines[1:])
                        new_blocks.append(
                            f"{new_tag}\n\n{rest.lstrip()}\n\nREFLECTION:\n{upd['reflection']}"
                        )
                        del update_map[(trade_date, ticker)]
                        matched = True
                        break

                if not matched:
                    new_blocks.append(block)

            new_blocks = self._apply_rotation(new_blocks)
            new_text = self._SEPARATOR.join(new_blocks)
            tmp_path = self._log_path.with_suffix(".tmp")
            tmp_path.write_text(new_text, encoding="utf-8")
            tmp_path.replace(self._log_path)

    # --- Helpers ---

    @staticmethod
    def _resolved_tag(
        trade_date, ticker, rating, raw_pct, alpha_pct, holding_days, resolution_date
    ) -> str:
        """Build a resolved entry tag, recording the outcome's known-by date.

        ``resolution_date`` (the date of the last price bar used for the return)
        is the point-in-time cutoff a later run filters on (#1251). Omitted when
        unavailable, keeping the legacy 6-field tag.
        """
        tag = f"[{trade_date} | {ticker} | {rating} | {raw_pct} | {alpha_pct} | {holding_days}d"
        if resolution_date:
            tag += f" | resolved:{resolution_date}"
        return tag + "]"

    def _apply_rotation(self, blocks: list[str]) -> list[str]:
        """Drop oldest resolved blocks when their count exceeds max_entries.

        Pending blocks are always kept (they represent unprocessed work).
        Returns ``blocks`` unchanged when rotation is disabled or under cap.
        """
        if not self._max_entries or self._max_entries <= 0:
            return blocks

        # Tag each block with (kept, is_resolved) by parsing tag-line markers.
        decisions = []
        for block in blocks:
            stripped = block.strip()
            if not stripped:
                decisions.append((block, False))
                continue
            tag_line = stripped.splitlines()[0].strip()
            is_resolved = (
                tag_line.startswith("[")
                and tag_line.endswith("]")
                and not tag_line.endswith("| pending]")
            )
            decisions.append((block, is_resolved))

        resolved_count = sum(1 for _, r in decisions if r)
        if resolved_count <= self._max_entries:
            return blocks

        to_drop = resolved_count - self._max_entries
        kept: list[str] = []
        for block, is_resolved in decisions:
            if is_resolved and to_drop > 0:
                to_drop -= 1
                continue
            kept.append(block)
        return kept

    def _parse_entry(self, raw: str) -> dict | None:
        lines = raw.strip().splitlines()
        if not lines:
            return None
        tag_line = lines[0].strip()
        if not (tag_line.startswith("[") and tag_line.endswith("]")):
            return None
        fields = [f.strip() for f in tag_line[1:-1].split("|")]
        if len(fields) < 4:
            return None
        # Optional trailing "resolved:YYYY-MM-DD" field records when the outcome
        # became known, for point-in-time filtering (#1251).
        resolved = None
        for f in fields[6:]:
            if f.startswith("resolved:"):
                resolved = f[len("resolved:"):].strip()
        entry = {
            "date": fields[0],
            "ticker": fields[1],
            "rating": fields[2],
            "pending": fields[3] == "pending",
            "raw": fields[3] if fields[3] != "pending" else None,
            "alpha": fields[4] if len(fields) > 4 else None,
            "holding": fields[5] if len(fields) > 5 else None,
            "resolved": resolved,
        }
        body = "\n".join(lines[1:]).strip()
        decision_match = self._DECISION_RE.search(body)
        reflection_match = self._REFLECTION_RE.search(body)
        entry["decision"] = decision_match.group(1).strip() if decision_match else ""
        entry["reflection"] = reflection_match.group(1).strip() if reflection_match else ""
        return entry

    def _format_full(self, e: dict) -> str:
        raw = e["raw"] or "n/a"
        alpha = e["alpha"] or "n/a"
        holding = e["holding"] or "n/a"
        tag = f"[{e['date']} | {e['ticker']} | {e['rating']} | {raw} | {alpha} | {holding}]"
        parts = [tag, f"DECISION:\n{e['decision']}"]
        if e["reflection"]:
            parts.append(f"REFLECTION:\n{e['reflection']}")
        return "\n\n".join(parts)

    def _format_reflection_only(self, e: dict) -> str:
        tag = f"[{e['date']} | {e['ticker']} | {e['rating']} | {e['raw'] or 'n/a'}]"
        if e["reflection"]:
            return f"{tag}\n{e['reflection']}"
        text = e["decision"][:300]
        suffix = "..." if len(e["decision"]) > 300 else ""
        return f"{tag}\n{text}{suffix}"
