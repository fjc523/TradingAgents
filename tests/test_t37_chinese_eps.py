"""计字及EPS近零/负基数契约，全部离线。"""
import pandas as pd
from tradingagents.agents.rating import chinese_length,output_flags
from tradingagents.dataflows.vendors.yahoo.expectations import render_expectations


def test_latin_numeric_tokens_do_not_exhaust_chinese_limit():
    evidence='字'*198+' 754.54 Alpha'
    flags=output_flags({'evidence_check':evidence},{},layer='rm')
    assert chinese_length(evidence)==200 and flags['evidence_check_count']==200
    assert flags['evidence_check_raw_length']==len(evidence) and not flags.get('evidence_check_overlength')
    assert output_flags({'evidence_check':evidence+'。'},{},layer='rm')['evidence_check_overlength']


def test_eps_source_percentage_cannot_override_near_zero_or_negative_base():
    frames={'eps_trend':pd.DataFrame({'current':[.26],'7daysAgo':[-.2]},index=['0q']),
        'earnings_history':pd.DataFrame({'epsActual':[.23,.29,.26],
            'epsEstimate':[.00705,.01313,-.2],'surprisePercent':[31.6241,21.0868,-2.3]},index=['a','b','c'])}
    text=render_expectations(frames,queried_at='2026-10-03T17:00:00+00:00')
    assert '变化+230.00%' in text and '|c|0.26|-0.2|+230.00%|' in text
    assert text.count('基数过小，百分比不可比')==2 and '3162' not in text and '2108' not in text
