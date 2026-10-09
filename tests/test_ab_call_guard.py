"""实验provider尝试预算拦截；全部使用本地假CLI/进程。"""
from pathlib import Path
from types import SimpleNamespace
import json
import pytest

from tradingagents.llm_clients.codex_exec.runner import CodexExecRunner, model_call_guard, reserve_model_call, reset_abort
from tradingagents.llm_clients.claude_exec.runner import ClaudeExecRunner
from tradingagents.llm_clients.errors import LLMNonRecoverableError


def test_codex_guard_counts_transient_retry_and_default_unchanged(tmp_path,monkeypatch):
    reset_abort();calls=[]
    fake=Path(__file__).parent/'fixtures/fake_codex'
    monkeypatch.setenv('FAKE_CODEX_ARGV_LOG',str(tmp_path/'argv.jsonl'))
    monkeypatch.setenv('FAKE_CODEX_OUTPUT','{"content":"离线"}')
    monkeypatch.setenv('FAKE_CODEX_EXIT_SEQUENCE','1,0')
    monkeypatch.setenv('FAKE_CODEX_STDERR_SEQUENCE','connection reset,')
    runner=CodexExecRunner(binary=str(fake),model='gpt-offline',reasoning_effort='medium',retries=1,sleeper=lambda _:None)
    with model_call_guard(lambda p,c:calls.append(p)):
        runner.run('零模型夹具',{'type':'object'})
    assert calls==['codex_exec','codex_exec']
    reserve_model_call('codex_exec')
    assert len(calls)==2


def test_claude_guard_counts_each_failed_attempt_after_auth(tmp_path):
    from tradingagents.llm_clients.claude_exec.runner import ClaudeTransientError
    calls=[]
    class Process:
        returncode=1
        def communicate(self,**kwargs):return '', 'connection reset'
        def poll(self):return self.returncode
    auth=lambda *a,**k:SimpleNamespace(returncode=0,stdout=json.dumps({'loggedIn':True,'authMethod':'claude.ai','apiProvider':'firstParty','subscriptionType':'pro'}))
    runner=ClaudeExecRunner(popen=lambda *a,**k:Process(),auth_run=auth,retries=1,sleeper=lambda _:None)
    with model_call_guard(lambda p,c:calls.append(p)),pytest.raises(ClaudeTransientError):
        runner.run('零模型夹具',{'type':'object'})
    assert calls==['claude_exec','claude_exec']


def test_guard_stops_before_provider_process():
    class Stop(LLMNonRecoverableError):pass
    def stop(*args):raise Stop('已到硬上限')
    runner=CodexExecRunner(binary='不执行',model='gpt-offline',reasoning_effort='medium',popen=lambda *a,**k:pytest.fail('不得发起进程'))
    with model_call_guard(stop),pytest.raises(Stop):runner.run('零模型夹具',{'type':'object'})
