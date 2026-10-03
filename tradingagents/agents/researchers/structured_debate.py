"""两阶段研究辩论：每个并行节点只写自己的字段，汇合点统一写旧状态。"""
from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction, report_or_absent
from tradingagents.agents.structured import NO_EXTERNAL_TOOLS


def create_research_turn(llm, side, phase):
    """首轮不接触历史或对方文本；反驳只读取屏障后已完成的对方首轮。"""
    role = '多头' if side == 'bull' else '空头'
    opponent = 'bear' if side == 'bull' else 'bull'
    key = f'{side}_{phase}'

    def node(state):
        reports = '\n\n'.join(f'**{label}：**\n{report_or_absent(state.get(field, ""), source)}'
            for field, label, source in [('market_report', '市场报告', 'market'),
                                        ('fundamentals_report', '基本面报告', 'fundamentals'),
                                        ('sentiment_report', '情绪报告', 'sentiment'),
                                        ('news_report', '新闻报告', 'news')])
        context = get_instrument_context_from_state(state, profile=f'{side}_researcher')
        prompt = f'你是{role}研究员，本次只完成' + ('首轮陈述。\n' if phase == 'opening' else '针对对方首轮的反驳。\n')
        prompt += context + '\n\n' + reports
        if phase == 'opening':
            prompt += '\n\n输出编号1–3的最强论据，每条注明报告来源和输入中的具体数字，并列出可证伪条件。全文≤700字。当前双方同时首轮，不预判、编造或回应对方发言，不给交易结论。'
        else:
            prompt += '\n\n**对方首轮：**\n' + state[f'{opponent}_opening']
            prompt += '\n\n只针对对方最强的2条证据逐条回应，明确引用所回应条目，承认有效部分；核对其可证伪条件是否已触发及依据。全文≤500字。不新增反驳轮次，不给交易结论。'
        prompt += '\n数字必须来自输入，推算须明确标注及列出输入；缺失则说明，不能捏造。\n' + NO_EXTERNAL_TOOLS + get_language_instruction()
        return {key: llm.invoke(prompt).content}

    return node


def join_research_debate(state):
    """固定顺序写一次兼容历史，避免并行更新同一个旧字典。"""
    bull_open = 'Bull Analyst 首轮: ' + state['bull_opening']
    bear_open = 'Bear Analyst 首轮: ' + state['bear_opening']
    bull_rebut = 'Bull Analyst 反驳: ' + state['bull_rebuttal']
    bear_rebut = 'Bear Analyst 反驳: ' + state['bear_rebuttal']
    return {'investment_debate_state': {
        'history': '\n\n'.join([bull_open, bear_open, bull_rebut, bear_rebut]),
        'bull_history': '\n\n'.join([bull_open, bull_rebut]),
        'bear_history': '\n\n'.join([bear_open, bear_rebut]),
        'current_response': bear_rebut, 'count': 4,
    }}
