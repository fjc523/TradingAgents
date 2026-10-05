"""角色回退和批次并发离线回归，禁止真实模型。"""
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
from tradingagents.graph.role_fallback import RoleBatchBreaker, RoleFallbackLLM
from tradingagents.llm_clients.claude_exec.runner import ClaudeQuotaError, ClaudeConfigError, ClaudeTransientError
from tradingagents.llm_clients.codex_exec import runner as common

class Model:
    def __init__(self,error=None):self.error=error;self.calls=0
    def invoke(self,value,*args,**kwargs):
        self.calls+=1
        if self.error:raise self.error
        return value
    def with_structured_output(self,schema,**kwargs):return self

@pytest.fixture(autouse=True)
def isolated_abort():
    common.reset_abort()
    yield
    common.reset_abort()

def wrapper(primary,fallback,breaker,path=None,role='research_manager'):
    return RoleFallbackLLM(primary,fallback,role=role,breaker=breaker,
        target={'provider':'codex_exec','model':'gpt-6.1-sol','effort':'xhigh'},usage_log_path=path)

@pytest.mark.parametrize('error',[ClaudeQuotaError('额度'),ClaudeConfigError('配置')])
def test_immediate_breaker_roles_and_safe_log(error,tmp_path):
    breaker=RoleBatchBreaker();primary=Model(error);sol=Model();path=tmp_path/'calls.jsonl'
    with common.codex_usage_context(ticker='MOCK'):
        assert wrapper(primary,sol,breaker,path).with_structured_output({}).invoke('结果')=='结果'
        assert wrapper(primary,sol,breaker,path,'trader').invoke('结果')=='结果'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert primary.calls==1 and sol.calls==2 and breaker.snapshot()['open']
    assert rows[1]['category'].startswith('breaker:') and rows[1]['ticker']=='MOCK'
    assert rows[0]['to']=='codex_exec/gpt-6.1-sol/xhigh' and rows[0]['model_requests']==0

def test_second_exhaustion_and_new_batch_reset():
    breaker=RoleBatchBreaker();error=ClaudeTransientError('耗尽');error.category='timeout'
    primary=Model(error);sol=Model();model=wrapper(primary,sol,breaker)
    model.invoke('一');assert not breaker.snapshot()['open']
    model.invoke('二');assert breaker.snapshot()['open']
    model.invoke('三');assert primary.calls==2 and sol.calls==3
    fresh=RoleBatchBreaker();assert not fresh.snapshot()['open'] and fresh.snapshot()['exhaustions']==0

@pytest.mark.parametrize('error',[ValueError('校验'),RuntimeError('未知')])
def test_parse_and_unknown_error_do_not_fallback(error):
    breaker=RoleBatchBreaker();sol=Model()
    with pytest.raises(type(error)):wrapper(Model(error),sol,breaker).invoke('失败')
    assert sol.calls==0 and not breaker.snapshot()['open']

def test_abort_never_fallback():
    from tradingagents.llm_clients.codex_exec.errors import CodexAbortedError
    sol=Model();common._ABORT.set()
    with pytest.raises(CodexAbortedError):wrapper(Model(),sol,RoleBatchBreaker()).invoke('中止')
    assert sol.calls==0 and common._ABORT.is_set()

def test_concurrent_breaker_and_append_rows_complete(tmp_path):
    breaker=RoleBatchBreaker();breaker.fail(ClaudeQuotaError('额度'));path=tmp_path/'usage.jsonl'
    def invoke(index):return wrapper(Model(),Model(),breaker,path).invoke(index)
    with ThreadPoolExecutor(max_workers=8) as pool:assert list(pool.map(invoke,range(32)))==list(range(32))
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows)==32 and all(row['category']=='breaker:quota' for row in rows)

def test_abort_during_primary_does_not_fallback():
    class Abort(Model):
        def invoke(self,value,*args,**kwargs):
            common._ABORT.set()
            raise ClaudeQuotaError('人工中止后的返回')
    sol=Model()
    with pytest.raises(ClaudeQuotaError):wrapper(Abort(),sol,RoleBatchBreaker()).invoke('中止')
    assert sol.calls==0


def test_fallback_runner_role_clone_preserves_shared_default(monkeypatch):
    from tradingagents.graph import role_llms
    from tradingagents.llm_clients import create_llm_client
    quick=create_llm_client('codex_exec','gpt-6.1-sol',reasoning_effort='medium',role='quick').get_llm()
    deep=create_llm_client('codex_exec','gpt-6.1-sol',reasoning_effort='xhigh',role='deep').get_llm()
    monkeypatch.setattr(role_llms,'create_llm_client',lambda *a,**kw:SimpleNamespace(get_llm=lambda:Model(ClaudeQuotaError('额度'))))
    config={'llm_provider':'codex_exec','quick_think_llm':'gpt-6.1-sol','deep_think_llm':'gpt-6.1-sol','codex_quick_reasoning_effort':'medium','codex_deep_reasoning_effort':'xhigh','role_llm_scheme':'B'}
    models,_=role_llms.build_role_llms(config,quick,deep)
    observed=[]
    def run(prompt,schema):
        observed.append(common._USAGE_CONTEXT.get()['role'])
        return common.CodexResult({'content':'离线'}, {}, [])
    for role in ('research_manager','trader','portfolio_manager'):
        models[role].fallback.runner.run=run
        assert models[role].invoke('内容').content=='离线'
    assert observed==['research_manager','trader','portfolio_manager']
    assert deep.runner.role=='deep' and quick.runner.role=='quick'

def test_config_context_deepcopy_preserves_batch_shared_state(monkeypatch):
    from copy import deepcopy
    from tradingagents.dataflows import config as data_config
    from tradingagents.dataflows.config import run_config, get_config, set_config, run_config_context
    monkeypatch.setattr(data_config,'_config',deepcopy(data_config._config))
    breaker=RoleBatchBreaker();config={'_role_llm_breaker':breaker}
    assert deepcopy(config)['_role_llm_breaker'] is breaker
    set_config(config)
    assert get_config()['_role_llm_breaker'] is breaker
    assert run_config_context(config).run(get_config)['_role_llm_breaker'] is breaker
    with run_config(config):
        scoped=get_config()
        scoped['_role_llm_breaker'].fail(ClaudeQuotaError('额度'))
        assert scoped['_role_llm_breaker'] is breaker
    assert breaker.snapshot()['open']
