"""T52 目标方法生成枚举与两条真实结构化链路的离线回归。"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from tradingagents.agents import schemas as S
from tradingagents.agents.price_plan_legs import BuyLeg, render_legs
from tradingagents.agents.structured import invoke_decision
from tradingagents.llm_clients.codex_exec.chat_model import CodexExecChatModel, _validate_arguments
from tradingagents.llm_clients.codex_exec.errors import CodexOutputFormatError
from tradingagents.llm_clients.codex_exec.runner import CodexExecRunner
from tradingagents.llm_clients.codex_exec.schemas import schema_for
from tradingagents.llm_clients.claude_exec.client import ClaudeExecChatModel
from tradingagents.llm_clients.claude_exec.runner import ClaudeExecRunner


@pytest.mark.parametrize('provider', ['codex_exec', 'claude_exec'])
def test_target_method_enum_and_actual_structured_path(provider, monkeypatch, tmp_path):
    """SPCX 两层原值均拒于生成契约，运行期只软验收字段而不整段回退。"""
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    field = BuyLeg.model_json_schema()['properties']['target_method']
    assert field['description'] == '阻力位或ATR替代'
    assert next(item['enum'] for item in field['anyOf'] if item.get('type') == 'string') == ['阻力位', 'ATR替代']
    assert {'type': 'null'} in field['anyOf']
    for legal in ('阻力位', 'ATR替代', None):
        assert BuyLeg(target_method=legal).target_method == legal
    for layer, raw in (
        ('trader', '阻力位（目标距离6.76个ATR，超过3个ATR，单列统计）'),
        ('pm', '阻力位（距区间上沿6.76ATR，超过3ATR，单列统计）'),
    ):
        with pytest.raises(ValidationError):
            BuyLeg(target_method=raw)
        with pytest.raises(CodexOutputFormatError):
            _validate_arguments(raw, field, 'target_method')
        base = S.TraderProposal if layer == 'trader' else S.PortfolioDecision
        model = S.decision_schema(base, {'price_plan_legs': True}, layer)
        strict = schema_for(model)
        payload = {'action': 'Buy', 'reasoning': '证据', 'direction_change': '否'} if layer == 'trader' else {
            'rating': 'Buy', 'executive_summary': '摘要', 'investment_thesis': '证据', 'direction_change': '否'}
        instance = model(**payload)
        output = {key: instance.model_dump(mode='json').get(key) for key in strict['properties']}
        index = 1 if layer == 'trader' else 0
        output['buy_legs'] = [{}] * index + [{'target_method': raw, 'first_target': 225.64, 'target_anchor': '252d_High'}]
        sent = []

        class Process:
            returncode = 0

            def communicate(self, *args, **kwargs):
                if provider == 'codex_exec':
                    return '', ''
                return '\n'.join(json.dumps(event, ensure_ascii=False) for event in [
                    {'type': 'system', 'subtype': 'init', 'model': 'claude-opus-5-5', 'apiKeySource': 'none',
                     'tools': ['StructuredOutput'], 'mcp_servers': [], 'skills': []},
                    {'type': 'result', 'subtype': 'success', 'is_error': False, 'structured_output': output,
                     'modelUsage': {'claude-opus-5-5': {'provider': 'firstParty'}}},
                ]), ''

        def popen(argv, **kwargs):
            if provider == 'codex_exec':
                sent.append(json.loads(Path(argv[argv.index('--output-schema') + 1]).read_text()))
                Path(argv[argv.index('-o') + 1]).write_text(json.dumps(output, ensure_ascii=False))
            else:
                sent.append(json.loads(argv[argv.index('--json-schema') + 1]))
            return Process()

        if provider == 'codex_exec':
            llm = CodexExecChatModel(model_name='gpt-6.1-sol', runner=CodexExecRunner(model='gpt-6.1-sol', popen=popen))
        else:
            runner = ClaudeExecRunner(popen=popen, auth_run=lambda *a, **kw: SimpleNamespace(returncode=0,
                stdout=json.dumps({'loggedIn': True, 'authMethod': 'claude.ai', 'apiProvider': 'firstParty', 'subscriptionType': 'pro'})))
            llm = ClaudeExecChatModel(model_name='claude-opus-5-5', runner=runner)
        plain = Mock()
        plain.invoke.side_effect = AssertionError('禁止整段自由文本回退')
        text, parsed = invoke_decision(llm.with_structured_output(model), plain, '离线提示', render_legs, layer)
        assert sent == [dict(strict)]
        assert '"enum": ["阻力位", "ATR替代"]' in json.dumps(sent[0]['properties']['buy_legs'], ensure_ascii=False)
        assert parsed['buy_legs'][index]['target_method'] is None
        key = f'buy_legs.{index}.target_method'
        assert parsed['leg_validation_raw'][key] == raw
        assert key + ':invalid_value' in parsed['leg_validation_flags']
        assert parsed['buy_legs'][index]['first_target'] == 225.64
        assert parsed['buy_legs'][index]['target_anchor'] == '252d_High'
        assert '225.64' in text
        plain.invoke.assert_not_called()
