from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from tradingagents.llm_clients.codex_exec.chat_model import CodexExecChatModel
from tradingagents.llm_clients.codex_exec.errors import (
    CodexAbortedError,
    CodexFatalConfigError,
    CodexOutputFormatError,
    CodexQuotaError,
    CodexTransientExhaustedError,
    CodexToolCallError,
)
from tradingagents.llm_clients.codex_exec.runner import (
    CodexExecRunner,
    abort_all_calls,
    codex_usage_context,
    reset_abort,
)
from tradingagents.llm_clients.errors import LLMNonRecoverableError
from tradingagents.llm_clients.factory import build_llm_kwargs, create_llm_client


FAKE = Path(__file__).parent / "fixtures" / "fake_codex"


@pytest.fixture(autouse=True)
def clear_abort():
    reset_abort()
    yield
    reset_abort()


def _runner(tmp_path, monkeypatch, *, retries=0, timeout=2, monotonic=time.monotonic):
    monkeypatch.setenv("FAKE_CODEX_ARGV_LOG", str(tmp_path / "argv.jsonl"))
    monkeypatch.setenv("FAKE_CODEX_OUTPUT", '{"content":"offline response"}')
    return CodexExecRunner(
        binary=str(FAKE), model="gpt-offline", reasoning_effort="medium",
        retries=retries, timeout=timeout, sleeper=lambda _: None,
        monotonic=monotonic,
        usage_log_path=str(tmp_path / "usage.jsonl"),
    )


@pytest.mark.unit
def test_runner_uses_restricted_cli_empty_workdir_stdin_and_usage_context(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_CODEX_PROMPT_LOG", str(tmp_path / "prompt.txt"))
    monkeypatch.setenv("FAKE_CODEX_CWD_LOG", str(tmp_path / "cwd.json"))
    stdout = '\n'.join([
        json.dumps({"type": "item.completed", "item": {"type": "command_execution"}}),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 12, "output_tokens": 3}}),
    ])
    monkeypatch.setenv("FAKE_CODEX_STDOUT", stdout)
    with codex_usage_context(ticker="AAPL", role="deep", call_type="text"):
        result = runner.run("local test prompt", {"type": "object"})
    argv = json.loads((tmp_path / "argv.jsonl").read_text().splitlines()[0])
    assert "--ignore-user-config" in argv and "--ignore-rules" in argv
    assert "--sandbox" in argv and argv[argv.index("--sandbox") + 1] == "read-only"
    assert argv[-1] == "-"
    for feature in ("memories", "shell_tool", "unified_exec", "apps", "plugins", "multi_agent"):
        assert "--disable" in argv and feature in argv
    assert json.loads((tmp_path / "cwd.json").read_text()) == []
    assert (tmp_path / "prompt.txt").read_text() == "local test prompt"
    assert result.events["usage"] == {"input_tokens": 12, "output_tokens": 3}
    assert result.events["agent_actions"] == [{"type": "command_execution"}]
    usage = json.loads((tmp_path / "usage.jsonl").read_text())
    assert usage["ticker"] == "AAPL" and usage["role"] == "deep"
    assert usage["tokens"]["input_tokens"] == 12 and usage["result"] == "success"
    assert "local test prompt" not in (tmp_path / "usage.jsonl").read_text()


@pytest.mark.unit
def test_runner_retries_transient_and_records_each_attempt(tmp_path, monkeypatch):
    clock_values = iter([10.0, 11.25, 11.25, 15.0])
    runner = _runner(tmp_path, monkeypatch, retries=2, monotonic=lambda: next(clock_values))
    monkeypatch.setenv("FAKE_CODEX_EXIT_SEQUENCE", "1,0")
    monkeypatch.setenv("FAKE_CODEX_STDERR_SEQUENCE", "connection reset,")
    delays = []
    runner._sleep = delays.append
    runner.run("prompt", {"type": "object"})
    records = [json.loads(row) for row in (tmp_path / "usage.jsonl").read_text().splitlines()]
    assert delays == [30]
    assert [row["result"] for row in records] == ["transient", "success"]
    assert [row["retry"] for row in records] == [0, 1]
    assert [row["duration_seconds"] for row in records] == [1.25, 3.75]


@pytest.mark.unit
def test_runner_abort_between_popen_and_registration_terminates_child(tmp_path, monkeypatch):
    class FakeProcess:
        pid = 987654321
        returncode = None

        def __init__(self):
            self.terminated = False
            self.communicated = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

        def communicate(self, *args, **kwargs):
            self.communicated = True
            return "", ""

    process = FakeProcess()

    def popen_after_abort(*args, **kwargs):
        abort_all_calls()
        return process

    kill_attempts = []

    def no_process_group(pid, sig):
        kill_attempts.append((pid, sig))
        raise ProcessLookupError

    monkeypatch.setattr("tradingagents.llm_clients.codex_exec.runner.os.killpg", no_process_group)
    runner = CodexExecRunner(
        binary="fake-codex", model="offline", retries=0, popen=popen_after_abort,
    )
    with pytest.raises(CodexAbortedError, match="全局中止"):
        runner.run("prompt", {"type": "object"})
    assert kill_attempts and kill_attempts[0][0] == process.pid
    assert process.terminated
    assert not process.communicated


@pytest.mark.unit
def test_non_tool_text_format_error_is_propagated_unchanged():
    class FakeRunner:
        role = "quick"

        def __init__(self):
            self.calls = 0

        def run(self, prompt, schema):
            from tradingagents.llm_clients.codex_exec.runner import CodexResult
            self.calls += 1
            return CodexResult({}, {}, [])

    runner = FakeRunner()
    with pytest.raises(CodexOutputFormatError, match="文本输出缺少字符串字段 content"):
        CodexExecChatModel(model_name="offline", runner=runner).invoke("plain text")
    assert runner.calls == 1


@pytest.mark.unit
def test_runner_classifies_fatal_quota_and_exhausted_errors(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, retries=1)
    monkeypatch.setenv("FAKE_CODEX_EXIT", "1")
    monkeypatch.setenv("FAKE_CODEX_STDERR", "unknown configuration key")
    with pytest.raises(CodexFatalConfigError):
        runner.run("prompt", {"type": "object"})
    assert issubclass(CodexFatalConfigError, LLMNonRecoverableError)
    reset_abort()
    monkeypatch.setenv("FAKE_CODEX_STDERR", "quota exceeded")
    with pytest.raises(CodexQuotaError):
        runner.run("prompt", {"type": "object"})
    reset_abort()
    monkeypatch.setenv("FAKE_CODEX_STDERR", "connection reset")
    with pytest.raises(CodexTransientExhaustedError):
        runner.run("prompt", {"type": "object"})


@pytest.mark.unit
def test_global_abort_terminates_running_and_rejects_queued_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_STARTED", str(tmp_path / "started"))
    monkeypatch.setenv("FAKE_CODEX_SLEEP", "30")
    runner = _runner(tmp_path, monkeypatch, retries=0, timeout=40)
    outcomes = []

    def call():
        try:
            runner.run("prompt", {"type": "object"})
            outcomes.append("returned")
        except CodexAbortedError:
            outcomes.append("aborted")

    first = threading.Thread(target=call)
    first.start()
    deadline = time.monotonic() + 5
    while not (tmp_path / "started").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    second = threading.Thread(target=call)
    second.start()
    time.sleep(0.1)
    abort_all_calls()
    first.join(8)
    second.join(8)
    assert not first.is_alive() and not second.is_alive()
    assert outcomes.count("aborted") == 2


@pytest.mark.unit
def test_tools_retry_validation_once_and_structured_output_parses(tmp_path):
    class FakeRunner:
        role = "quick"

        def __init__(self, outputs):
            self.outputs = outputs
            self.prompts = []

        def run(self, prompt, schema):
            from tradingagents.llm_clients.codex_exec.runner import CodexResult
            self.prompts.append(prompt)
            return CodexResult(self.outputs.pop(0), {}, [])

    @tool
    def lookup(symbol: str) -> str:
        """Look up one symbol."""
        return symbol

    bad = {"kind": "tool_calls", "content": "", "tool_calls": [
        {"name": "unknown", "arguments_json": "{}"}
    ]}
    good = {"kind": "tool_calls", "content": "", "tool_calls": [
        {"name": "lookup", "arguments_json": '{"symbol":"AAPL"}'}
    ]}
    runner = FakeRunner([bad, good])
    model = CodexExecChatModel(model_name="offline", runner=runner).bind_tools([lookup])
    response = model.invoke("get AAPL")
    assert response.tool_calls[0]["args"] == {"symbol": "AAPL"}
    assert response.tool_calls[0]["id"] != "codex_call_0"
    assert "校验失败" in runner.prompts[1]

    bad_again = FakeRunner([bad, bad])
    with pytest.raises(CodexToolCallError):
        CodexExecChatModel(model_name="offline", runner=bad_again).bind_tools([lookup]).invoke("x")

    history_prompt = CodexExecChatModel(model_name="offline", runner=runner)._prompt([
        AIMessage(content="", tool_calls=[{
            "name": "lookup", "args": {"symbol": "AAPL"}, "id": "call-history", "type": "tool_call",
        }]),
        ToolMessage(content="looked up", name="lookup", tool_call_id="call-history"),
    ])
    history = json.loads(history_prompt)["messages"]
    assert history[0]["tool_calls"][0]["id"] == "call-history"
    assert history[1]["role"] == "tool" and history[1]["tool_call_id"] == "call-history"
    assert history[1]["name"] == "lookup"

    final = FakeRunner([{"kind": "final", "content": "done", "tool_calls": []}])
    final_response = CodexExecChatModel(model_name="offline", runner=final).bind_tools([lookup]).invoke("x")
    assert final_response.content == "done" and final_response.tool_calls == []

    class Result(BaseModel):
        value: int

    structured = CodexExecChatModel(
        model_name="offline", runner=FakeRunner([{"value": 4}])
    ).with_structured_output(Result)
    assert structured.invoke("give a value") == Result(value=4)


@pytest.mark.unit
def test_role_config_and_factory_use_keyless_codex_provider():
    config = {
        "llm_provider": "codex_exec", "codex_reasoning_effort": "medium",
        "codex_deep_reasoning_effort": "xhigh", "codex_quick_reasoning_effort": "low",
        "codex_binary": "/usr/local/bin/codex",
    }
    assert build_llm_kwargs(config, "deep")["reasoning_effort"] == "xhigh"
    quick = build_llm_kwargs(config, "quick")
    assert quick["reasoning_effort"] == "low" and quick["codex_binary"] == "/usr/local/bin/codex"
    assert create_llm_client("codex_exec", "custom-model").get_provider_name() == "codex_exec"


@pytest.mark.unit
def test_graph_constructs_deep_and_quick_with_separate_effort(tmp_path, monkeypatch):
    import copy
    from types import SimpleNamespace

    import tradingagents.graph.trading_graph as graph_module
    from tradingagents.default_config import DEFAULT_CONFIG

    captured = []

    class FakeClient:
        def get_llm(self):
            return object()

    class FakeSetup:
        def __init__(self, *args, **kwargs):
            pass

        def setup_graph(self, analysts):
            return SimpleNamespace(compile=lambda **kwargs: object())

    class FakeLogic:
        def __init__(self, **kwargs):
            pass

    class FakePropagator:
        def __init__(self, **kwargs):
            pass

    class FakeMemory:
        def __init__(self, config):
            pass

    monkeypatch.setattr(graph_module, "set_config", lambda _: None)
    monkeypatch.setattr(graph_module.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(graph_module, "GraphSetup", FakeSetup)
    monkeypatch.setattr(graph_module, "ConditionalLogic", FakeLogic)
    monkeypatch.setattr(graph_module, "Propagator", FakePropagator)
    monkeypatch.setattr(graph_module, "TradingMemoryLog", FakeMemory)
    monkeypatch.setattr(graph_module, "Reflector", lambda _: object())

    def create_client(*, provider, model, base_url, **kwargs):
        captured.append((provider, model, kwargs))
        return FakeClient()

    monkeypatch.setattr(graph_module, "create_llm_client", create_client)
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update({
        "llm_provider": "codex_exec", "deep_think_llm": "same-model",
        "quick_think_llm": "same-model", "codex_deep_reasoning_effort": "xhigh",
        "codex_quick_reasoning_effort": "low", "data_cache_dir": str(tmp_path),
        "results_dir": str(tmp_path),
    })
    graph_module.TradingAgentsGraph(config=config)
    assert [entry[2]["reasoning_effort"] for entry in captured] == ["xhigh", "low"]


@pytest.mark.unit
def test_codex_path_settings_do_not_change_run_signature():
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    graph = object.__new__(TradingAgentsGraph)
    graph.selected_analysts = ("market",)
    graph.config = {
        "codex_binary": "/path/one", "codex_usage_log_path": "/tmp/one",
        "codex_prompt_log_dir": "/tmp/prompts-one", "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
    }
    first = graph._run_signature("stock")
    graph.config.update({
        "codex_binary": "/path/two", "codex_usage_log_path": "/tmp/two",
        "codex_prompt_log_dir": "/tmp/prompts-two",
    })
    assert graph._run_signature("stock") == first


@pytest.mark.unit
def test_model_unsupported_account_and_stdout_error_are_fatal(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, retries=2)
    monkeypatch.setenv("FAKE_CODEX_EXIT", "1")
    monkeypatch.setenv("FAKE_CODEX_STDOUT", json.dumps({
        "type": "error",
        "message": "The 'gpt-6.1-sol' model is not supported when using Codex with a ChatGPT account.",
    }))
    with pytest.raises(CodexFatalConfigError, match="请检查 codex_binary"):
        runner.run("prompt", {"type": "object"})


@pytest.mark.unit
def test_schema_strict_inlines_pydantic_definitions_and_falls_back_to_json_string():
    from tradingagents.llm_clients.codex_exec.schemas import schema_for

    class Child(BaseModel):
        count: int

    class Parent(BaseModel):
        child: Child
        optional: str | None = None

    strict = schema_for(Parent)
    assert "$ref" not in json.dumps(strict)
    assert strict["required"] == ["child", "optional"]
    assert strict["properties"]["child"]["properties"]["count"]["type"] == "integer"
    assert strict["properties"]["optional"]["anyOf"][1]["type"] == "null"
    fallback = schema_for({"$ref": "https://example.invalid/schema"})
    assert fallback["properties"] == {"json": {"type": "string"}}


@pytest.mark.unit
@pytest.mark.parametrize("stderr", ["HTTP 401 unauthorized", "requires a newer version of Codex"])
def test_codex_auth_or_cli_version_errors_are_fatal(tmp_path, monkeypatch, stderr):
    runner = _runner(tmp_path, monkeypatch, retries=2)
    monkeypatch.setenv("FAKE_CODEX_EXIT", "1")
    monkeypatch.setenv("FAKE_CODEX_STDERR", stderr)
    with pytest.raises(CodexFatalConfigError):
        runner.run("prompt", {"type": "object"})
