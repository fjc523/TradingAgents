"""财报表体期末与来源元数据：不从自然语言推断申报时点。"""
from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal, InvalidOperation
from io import StringIO


class StatementResult(str):
    """字符串表体不变；元数据仅交给来源观察者，持久结果仍兼容文本。"""
    def __new__(cls, text, statement_metadata=None):
        value = super().__new__(cls, text)
        value.statement_metadata = statement_metadata or {}
        return value


def latest_statement_period(text):
    """只解析既有CSV表头日期列，拒绝withheld和空表。"""
    if not isinstance(text, str) or 'data is withheld for this date' in text:
        return None
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith(('#', '⚠'))]
    table = list(csv.reader(StringIO('\n'.join(lines))))
    if len(table) < 2:
        return None
    periods = []
    for index, column in enumerate(table[0][1:], 1):
        try:
            period = date.fromisoformat(column.strip()[:10]).isoformat()
        except ValueError:
            continue
        # 表头有新日期但整列空/NaN/占位不能证明该期已有数据；零值有效。
        for row in table[1:]:
            if index >= len(row):
                continue
            try:
                value = Decimal(row[index].strip().replace(',', ''))
                if value.is_finite():
                    periods.append(period)
                    break
            except InvalidOperation:
                continue
    return max(periods) if periods else None
