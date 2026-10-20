"""未来美国事实视图：先冻结可见池，再逐窗复用主仓W4。"""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path


def visible_time(value, cutoff):
    try:
        timestamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return timestamp.tzinfo is not None and timestamp <= cutoff
    except (TypeError, ValueError):
        return False


def build_facts(log, as_of):
    from daily_analyzer.evaluation.report import deduplicate
    cutoff = datetime.fromisoformat(str(as_of).replace('Z', '+00:00'))
    if cutoff.tzinfo is None:
        raise ValueError('新事实视图必须有带时区as_of')
    raw = Path(log._outcomes_path).read_bytes() if log._outcomes_path and Path(log._outcomes_path).exists() else b''
    memory = log._log_path.read_bytes() if log._log_path and log._log_path.exists() else b''
    entries = [entry for part in memory.decode().split(log._SEPARATOR)
        if (entry := log._parse_entry(part.strip())) and not entry.get('pending') and log._known_by(entry.get('resolved'), as_of)]
    outcomes = [json.loads(line) for line in raw.splitlines() if line.strip()]
    candidates = [r for r in outcomes if r.get('is_current') is True and r.get('market', 'US') == 'US'
        and r.get('run_id') and r.get('symbol') and r.get('trade_date') and r.get('run_id') != log._current_run
        and visible_time(r.get('finished_at'), cutoff)]
    aliases = {r['symbol']: r.get('analysis_symbol') or r['symbol'] for r in candidates}
    aliases.update(log._aliases)
    log.fact_aliases = aliases
    legacy = {(e['date'], e['ticker']): e for e in entries}
    all_rows, authorities, audit = {}, {}, {}
    def opportunity(row):
        entry = row.get('entry') or {}
        basis = entry.get('planned_basis') or entry.get('basis')
        agent = row.get('analysis_symbol') or row['symbol']
        if row.get('price_data_end_date') and entry.get('date') and basis:
            return agent, row['price_data_end_date'], entry['date'], 'close' if basis == 'same_day_close' else 'open'
        return 'source', row['run_id'], row['symbol']
    for window in ('5', '10', '20'):
        pool = [r for r in candidates if r.get('primary_metric') in ('raw_return', 'excess_vs_spy')
            and (w := r.get('windows', {}).get(window, {})).get('status') == 'settled'
            and visible_time(w.get('settled_at'), cutoff) and log._return_number(w.get('primary_return')) is not None]
        authorities[window] = {(r['trade_date'], r.get('analysis_symbol') or r['symbol']) for r in pool}
        selected, diagnostics = deduplicate(pool)
        audit[window] = {'visible_ids': [f"{r['run_id']}|{r['symbol']}" for r in pool],
            'selected_ids': [f"{r['run_id']}|{r['symbol']}" for r in selected], 'n': len(selected), 'w4': diagnostics, 'legacy_suppressed': []}
        for row in selected:
            key = opportunity(row)
            old = legacy.get((row['trade_date'], row['symbol']), {})
            fingerprint_ok = bool(old.get('decision') and row.get('decision_fingerprint') == hashlib.sha256(old['decision'].encode()).hexdigest())
            fact = all_rows.setdefault(key, {'returns': {}, 'sources': {}, 'source_type': 'C2'})
            fact['returns'][window] = log._return_number(row['windows'][window]['primary_return'])
            fact['sources'][window] = {'date': row['trade_date'], 'ticker': row['symbol'],
                'rating': row.get('ratings', {}).get('pm') or '未提供', 'run_id': row['run_id'],
                'metric': row['primary_metric'], 'analysis_symbol': row.get('analysis_symbol') or row['symbol'], 'reflection': old.get('reflection', '') if fingerprint_ok else '',
                'reflection_association': '精确决策正文一致' if fingerprint_ok else '反思关联不可得'}
    facts = []
    for fact in all_rows.values():
        base = next(fact['sources'][w] for w in ('5', '10', '20') if w in fact['sources'])
        fact.update({**base, 'basis': 'C2 ' + base['metric'], 'c2_5': '5' in fact['sources'],
                     'reflection': fact['sources'].get('5', {}).get('reflection', '')})
        fact['different_winners'] = len({(s['run_id'], s['ticker']) for s in fact['sources'].values()}) > 1
        facts.append(fact)
    for entry in entries:
        agent = aliases.get(entry['ticker'], entry['ticker'])
        if (entry['date'], agent) in authorities['5']:
            audit['5']['legacy_suppressed'].append(f"{entry['date']}|{entry['ticker']}")
            continue
        value = log._return_number(entry.get('alpha'))
        if value is not None:
            facts.append({**entry, 'analysis_symbol': agent, 'returns': {'5': value}, 'sources': {}, 'source_type': 'legacy', 'c2_5': False,
                'basis': '旧memory 5日tag收益（口径未核验）', 'reflection': '', 'reflection_association': '反思关联不可得'})
    log.fact_audit = {'policy': 'w4-visible-v1', 'as_of': as_of, 'module_path': __file__, 'pid': os.getpid(),
        'outcomes_sha256': hashlib.sha256(raw).hexdigest(), 'memory_sha256': hashlib.sha256(memory).hexdigest(), 'windows': audit}
    return sorted(facts, key=lambda x: (x['date'], x['ticker']))
