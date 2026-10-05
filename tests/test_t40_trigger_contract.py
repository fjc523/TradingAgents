"""T40严格生成、可信腿软验收及B实际传递；全部离线桩。"""
import copy
import json
from types import SimpleNamespace

import pytest

from tradingagents.agents import schemas as S
from tradingagents.agents.price_plan_legs import normalize_legs, render_legs, soft_direction
from tradingagents.agents.rating import output_flags
from tradingagents.llm_clients.codex_exec.schemas import schema_for, _DecisionLegSchema
from tradingagents.llm_clients.codex_exec.chat_model import _validate_arguments
from tradingagents.llm_clients.codex_exec.errors import CodexOutputFormatError
from tradingagents.llm_clients.claude_exec.client import ClaudeExecChatModel
from tradingagents.llm_clients.claude_exec.runner import ClaudeExecRunner, parse_output, ClaudeConfigError


def decision(layer='trader', **config):
    base = S.TraderProposal if layer == 'trader' else S.PortfolioDecision
    model = S.decision_schema(base, {'price_plan_legs': True, **config}, layer)
    payload = {'action': 'Buy', 'reasoning': '证据', 'direction_change': '否'} if layer == 'trader' else {
        'rating': 'Buy', 'executive_summary': '摘要', 'investment_thesis': '证据', 'direction_change': '否'}
    instance = model(**payload)
    strict = schema_for(model)
    output = {key: instance.model_dump(mode='json').get(key) for key in strict['properties']}
    output['buy_legs'] = [{'status': '待触发', 'trigger_rule': '收盘站上201.99', 'trigger_price': 201.99,
                           'confirm_days': 2, 'zone_low': 201.99, 'zone_high': 204.9}]
    return model, strict, output


def events(output, **init_extra):
    return '\n'.join(json.dumps(event, ensure_ascii=False) for event in [
        {'type': 'system', 'subtype': 'init', 'model': 'claude-opus-5-5', 'apiKeySource': 'none',
         'tools': ['StructuredOutput'], 'mcp_servers': [], 'skills': [], **init_extra},
        {'type': 'result', 'subtype': 'success', 'is_error': False, 'structured_output': output,
         'modelUsage': {'claude-opus-5-5': {'provider': 'firstParty'}}}])


@pytest.mark.parametrize('layer', ['trader', 'pm'])
@pytest.mark.parametrize('probabilities', [False, True])
@pytest.mark.parametrize('timing', [False, True])
def test_actual_b_runner_identity_raw_and_strict_json(layer, probabilities, timing):
    model, strict, output = decision(layer, rating_probability_fields=probabilities, rating_timing_decoupled=timing)
    sent = []
    class Process:
        returncode = 0
        def communicate(self, **kwargs):
            return events(output), ''
    def popen(argv, **kwargs):
        sent.append(json.loads(argv[argv.index('--json-schema') + 1]))
        return Process()
    runner = ClaudeExecRunner(popen=popen, auth_run=lambda *a, **kw: SimpleNamespace(returncode=0,
        stdout=json.dumps({'loggedIn': True, 'authMethod': 'claude.ai', 'apiProvider': 'firstParty', 'subscriptionType': 'pro'})))
    received = []
    original = runner.run
    def run(prompt, schema):
        received.append(schema)
        return original(prompt, schema)
    runner.run = run
    parsed = ClaudeExecChatModel(model_name='claude-opus-5-5', runner=runner).with_structured_output(model).invoke('离线提示')
    assert isinstance(received[0], _DecisionLegSchema)
    assert sent == [dict(strict)]
    assert parsed.buy_legs[0].trigger_rule is None
    assert parsed.leg_validation_raw['buy_legs.0.trigger_rule'] == '收盘站上201.99'
    assert 'buy_legs.0.trigger_rule:invalid_value' in parsed.leg_validation_flags
    dumped = parsed.model_dump(mode='json')
    loaded = (S.load_trader_proposal if layer == 'trader' else S.load_portfolio_decision)(dumped)
    assert loaded.leg_validation_flags == parsed.leg_validation_flags
    assert loaded.leg_validation_raw == parsed.leg_validation_raw
    flags = output_flags(dumped, {}, layer=layer)
    assert flags['leg_validation_raw'] == parsed.leg_validation_raw


def test_generation_enums_metadata_and_optional_leg_missing_fields():
    for layer in ('trader', 'pm'):
        model, strict, output = decision(layer)
        assert 'leg_validation_raw' not in strict['properties'] and 'leg_validation_flags' not in strict['properties']
        for leg, choices in (('BuyLeg', ['触及区间', '收盘站上']), ('ReduceLeg', ['进入区间受阻', '收盘跌破', '立即', '其他条件'])):
            options = model.model_json_schema()['$defs'][leg]['properties']['trigger_rule']['anyOf']
            assert next(option['enum'] for option in options if option.get('type') == 'string') == choices
            assert {'type': 'null'} in options
        _validate_arguments(output, strict, 'Claude')
        assert output['buy_legs'][0]['trigger_rule'] == '收盘站上201.99'
        assert 'preconditions' not in output['buy_legs'][0]


@pytest.mark.parametrize('mutation', ['top_required', 'leg_key', 'top_unknown', 'leg_unknown', 'nonleg_type', 'nonleg_enum'])
def test_soft_projection_does_not_relax_other_fields(mutation):
    _, strict, output = decision()
    if mutation == 'top_required':
        del output['reasoning']
    elif mutation == 'leg_key':
        del output['buy_legs']
    elif mutation == 'top_unknown':
        output['unexpected'] = 1
    elif mutation == 'leg_unknown':
        output['buy_legs'][0]['unexpected'] = 1
    elif mutation == 'nonleg_type':
        output['reasoning'] = 3
    else:
        output['action'] = '未知评级'
    with pytest.raises(CodexOutputFormatError):
        _validate_arguments(output, strict, 'Claude')


def test_untrusted_schema_and_copy_contract_loss_fail_closed():
    _, strict, output = decision()
    for untrusted in (dict(strict), strict.copy(), schema_for(dict(strict))):
        assert not isinstance(untrusted, _DecisionLegSchema)
        with pytest.raises(CodexOutputFormatError):
            _validate_arguments(output, untrusted, 'Claude')
    altered = copy.deepcopy(strict)
    altered['properties']['reasoning'] = {'type': 'number'}
    with pytest.raises(CodexOutputFormatError, match='schema已改变'):
        _validate_arguments(output, altered, 'Claude')
    simple = {'anyOf': [{'type': 'string', 'enum': ['合法']}, {'type': 'null'}]}
    for invalid in ('非法', 3, False):
        with pytest.raises(CodexOutputFormatError):
            _validate_arguments(invalid, simple, '普通字段')
    _validate_arguments(None, simple, '普通字段')
    assert not isinstance(schema_for(S.decision_schema(S.TraderProposal, {'price_plan_legs': False}, 'trader')), _DecisionLegSchema)
    assert not isinstance(schema_for(S.decision_schema(S.ResearchPlan, {}, 'rm')), _DecisionLegSchema)


@pytest.mark.parametrize('init_extra', [{'model': '错误模型'}, {'apiKeySource': 'user'}, {'tools': ['Read']},
    {'mcp_servers': ['外部']}, {'skills': ['外部']}])
def test_b_identity_isolation_still_strict(init_extra):
    _, strict, output = decision()
    with pytest.raises(ClaudeConfigError):
        parse_output(events(output, **init_extra), strict, 'claude-opus-5-5')


def test_raw80_synonyms_mismatch_and_preconditions_long():
    payload = normalize_legs({'buy_legs': [{'status': 'pending', 'trigger_rule': '触及', 'kind': 'breakout',
        'preconditions': '条件' * 50}, {'trigger_rule': '错' * 100}], 'reduce_legs': [{'confirm_days': '2'}]})
    assert payload['buy_legs'][0]['status'] == '待触发' and payload['buy_legs'][0]['trigger_rule'] == '触及区间'
    assert payload['buy_legs'][0]['preconditions'] == '条件' * 50
    assert 'buy_legs.0:trigger_status_mismatch' in payload['leg_validation_flags']
    assert 'buy_legs.0:preconditions_long' in payload['leg_validation_flags']
    assert payload['leg_validation_raw']['buy_legs.1.trigger_rule'] == '错' * 80
    assert payload['reduce_legs'][0]['confirm_days'] == 2
    assert normalize_legs(payload)['leg_validation_flags'] == payload['leg_validation_flags']
    assert normalize_legs(payload)['leg_validation_raw'] == payload['leg_validation_raw']


@pytest.mark.parametrize('status,rule,expected', [('待触发', None, '收盘站上201.99（连续2日）后'),
    ('可执行', None, '触及区间'), ('仅观察', None, '仅观察'), (None, None, '状态未提供'),
    ('待触发', '触及区间', '触发规则与状态不一致')])
def test_status_render_contract(status, rule, expected):
    model = S.LegTraderProposal(action='Buy', reasoning='依据', direction_change='否', buy_legs=[{
        'status': status, 'trigger_rule': rule, 'trigger_price': 201.99, 'confirm_days': 2}], reduce_legs=[{}])
    text = render_legs(model)
    assert expected in text and text.count('触发未提供') == 1
    if status != '可执行':
        assert '触及区间' not in text
    model.buy_legs[0].confirm_days = None
    if status == '待触发':
        assert '连续未提供日' in render_legs(model)
    assert render_legs(S.LegTraderProposal(action='Hold', reasoning='依据', direction_change='否')) == ''


def test_outer_parentheses_multiple_groups_preserved():
    assert soft_direction('是（来源A）和（来源B）') == '是：（来源A）和（来源B）'
    assert soft_direction('是(来源A)和(来源B)') == '是：(来源A)和(来源B)'
