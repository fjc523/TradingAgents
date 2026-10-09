"""受限 Codex CLI 执行器：临时空工作目录、并发上限和全局中止。"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable

from .errors import (
    CodexAbortedError,
    CodexFatalConfigError,
    CodexOutputFormatError,
    CodexQuotaError,
    CodexTransientExhaustedError,
)

_DISABLE = (
    "memories", "shell_tool", "unified_exec", "apps", "plugins", "multi_agent",
    "multi_agent_v2", "image_generation", "view_image", "browser_use",
    "browser_use_external", "computer_use", "goals", "skill_search", "tool_suggest",
    "sleep_tool", "in_app_browser", "hooks", "remote_plugin",
    "skill_mcp_dependency_install", "shell_snapshot",
)
_ABORT = threading.Event()
_ACTIVE: set[subprocess.Popen] = set()
_ACTIVE_LOCK = threading.Lock()
_SLOTS = threading.Condition()
_ACTIVE_CALLS = 0
_USAGE_CONTEXT: ContextVar[dict[str, str | None]] = ContextVar(
    "codex_exec_usage_context", default={}
)


_CALL_GUARD: ContextVar[Any] = ContextVar("实验调用前预算守卫", default=None)


@contextmanager
def model_call_guard(guard):
    """仅显式实验作用域启用；正常生产默认不改变调用行为。"""
    token = _CALL_GUARD.set(guard)
    try:
        yield
    finally:
        _CALL_GUARD.reset(token)


def reserve_model_call(provider):
    """每次真实进程尝试前计量，包含失败与重试。"""
    guard = _CALL_GUARD.get()
    if guard is not None:
        guard(provider, dict(_USAGE_CONTEXT.get()))


def set_usage_context(*, ticker: str | None = None, role: str | None = None,
                      call_type: str | None = None):
    """设置当前上下文调用标签并返回可传给 reset_usage_context 的 token。"""
    merged = dict(_USAGE_CONTEXT.get())
    for key, value in (("ticker", ticker), ("role", role), ("call_type", call_type)):
        if value is not None:
            merged[key] = value
    return _USAGE_CONTEXT.set(merged)


def reset_usage_context(token) -> None:
    _USAGE_CONTEXT.reset(token)


@contextmanager
def codex_usage_context(*, ticker: str | None = None, role: str | None = None,
                        call_type: str | None = None):
    """临时标记当前模型调用，适用于主项目按标的注入的上下文。"""
    token = set_usage_context(ticker=ticker, role=role, call_type=call_type)
    try:
        yield
    finally:
        reset_usage_context(token)


def abort_all_calls() -> None:
    """拒绝排队调用并终止所有正在运行的 Codex 进程组。"""
    _ABORT.set()
    with _ACTIVE_LOCK:
        processes = tuple(_ACTIVE)
    for process in processes:
        _terminate(process)
    with _SLOTS:
        _SLOTS.notify_all()


def reset_abort() -> None:
    """为新批次清除中止标记；调用方需确保上一批次已停止。"""
    _ABORT.clear()


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.terminate()
        except OSError:
            return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            process.kill()


def _acquire_slot(limit: int) -> None:
    global _ACTIVE_CALLS
    limit = max(1, min(int(limit), 4))
    with _SLOTS:
        while _ACTIVE_CALLS >= limit:
            if _ABORT.is_set():
                raise CodexAbortedError("Codex 调用已被全局中止")
            _SLOTS.wait(timeout=0.1)
        if _ABORT.is_set():
            raise CodexAbortedError("Codex 调用已被全局中止")
        _ACTIVE_CALLS += 1


def _release_slot() -> None:
    global _ACTIVE_CALLS
    with _SLOTS:
        _ACTIVE_CALLS -= 1
        _SLOTS.notify_all()


def build_argv(
    binary: str, model: str, effort: str, schema_path: Path, output_path: Path,
) -> list[str]:
    """构造唯一的受限 Codex 命令参数。"""
    argv = [
        binary, "exec", "--json", "--ephemeral", "--skip-git-repo-check",
        "--ignore-user-config", "--ignore-rules", "--sandbox", "read-only",
        "--color", "never", "-m", model, "-c", f'model_reasoning_effort="{effort}"',
        "-c", f'model_instructions_file="{files("tradingagents.llm_clients.codex_exec").joinpath("minimal_instructions.md").resolve()}"',
        "-c", 'developer_instructions=""', "-c", "project_doc_max_bytes=0",
        "-c", 'web_search="disabled"', "-c", "include_permissions_instructions=false",
        "-c", "include_environment_context=false", "-c", "include_apps_instructions=false",
        "-c", "include_collaboration_mode_instructions=false",
    ]
    for feature in _DISABLE:
        argv.extend(("--disable", feature))
    argv.extend(("--output-schema", str(schema_path), "-o", str(output_path), "-"))
    return argv


def parse_events(stdout: str) -> dict[str, Any]:
    """解析 Codex JSONL 事件的用量、动作和配置漂移。"""
    usage: dict[str, Any] = {}
    actions: list[dict[str, Any]] = []
    drift: list[str] = []
    errors: list[str] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        event_type = str(event.get("type", ""))
        if event_type in {"error", "turn.failed", "turn_error", "turn.failed.error"}:
            errors.append("Codex JSONL error event")
        payload = event.get("payload") or event.get("item") or {}
        if event_type in {"turn.completed", "turn_complete"}:
            usage = event.get("usage", payload.get("usage", payload.get("token_usage", usage)))
        if event_type in {"item.completed", "item.started"}:
            item_type = str(payload.get("type", ""))
            if any(word in item_type for word in ("command", "file", "web", "search", "network")):
                actions.append({"type": item_type})
        message = " ".join(_strings(event))
        if re.search(r"ignoring\s+\d+\s+unrecognized configuration settings", message, re.I):
            drift.append(message[:300])
    return {"usage": usage, "agent_actions": actions, "config_drift": drift, "errors": errors}


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _classify(stderr: str, returncode: int) -> str:
    text = stderr.lower()
    if any(phrase in text for phrase in ("quota", "rate limit", "usage limit", "insufficient credits", "429")):
        return "quota"
    if any(phrase in text for phrase in (
        "unknown configuration", "unrecognized configuration", "invalid configuration",
        "unknown option", "not logged in", "authentication required", "invalid model",
        "unauthorized", "authentication failed", "http 401", "status code 401",
        "unsupported model", "not supported by your chatgpt account",
        "not supported by chatgpt account", "model is not supported by your account",
        "model is not supported when using codex with a chatgpt account",
        "requires a newer version of codex", "upgrade codex",
    )):
        return "fatal_config"
    return "transient"


def _failure_text(stdout: str) -> str:
    messages = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            messages.append(line)
            continue
        if isinstance(event, dict) and event.get("type") in {
            "error", "turn.failed", "turn_error", "turn.failed.error",
        }:
            messages.extend(_strings(event))
    return "\n".join(messages)


@dataclass
class CodexResult:
    output: Any
    events: dict[str, Any]
    argv: list[str]


class CodexExecRunner:
    def __init__(
        self,
        *,
        binary: str = "codex",
        model: str,
        reasoning_effort: str = "high",
        timeout: float = 600,
        retries: int = 3,
        max_concurrency: int = 4,
        usage_log_path: str | None = None,
        prompt_log_dir: str | None = None,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.binary = binary
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.max_concurrency = int(max_concurrency)
        self.usage_log_path = usage_log_path
        self.prompt_log_dir = prompt_log_dir
        self._popen = popen
        self._sleep = sleeper
        self._monotonic = monotonic

    def run(self, prompt: str, schema: dict[str, Any]) -> CodexResult:
        _acquire_slot(self.max_concurrency)
        try:
            return self._run(prompt, schema)
        finally:
            _release_slot()

    def _run(self, prompt: str, schema: dict[str, Any]) -> CodexResult:
        delays = (30, 120, 300)
        last_error: str = "Codex 调用失败"
        for attempt in range(self.retries + 1):
            if _ABORT.is_set():
                raise CodexAbortedError("Codex 调用已被全局中止")
            attempt_started = self._monotonic()
            with tempfile.TemporaryDirectory(prefix="ta-codex-work-") as work_dir, \
                    tempfile.TemporaryDirectory(prefix="ta-codex-meta-") as meta_dir:
                schema_path = Path(meta_dir) / "output-schema.json"
                output_path = Path(meta_dir) / "output.json"
                schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
                argv = build_argv(self.binary, self.model, self.reasoning_effort, schema_path, output_path)
                try:
                    reserve_model_call('codex_exec')
                    process = self._popen(
                        argv, cwd=work_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True, start_new_session=True,
                    )
                except FileNotFoundError as exc:
                    self._record(prompt, {"usage": {}, "config_drift": [], "agent_actions": []},
                                 result="fatal_config", retry=attempt,
                                 duration_seconds=max(0.0, self._monotonic() - attempt_started))
                    abort_all_calls()
                    raise CodexFatalConfigError(f"找不到 Codex CLI：{self.binary}") from exc
                with _ACTIVE_LOCK:
                    aborted_before_registration = _ABORT.is_set()
                    if not aborted_before_registration:
                        _ACTIVE.add(process)
                if aborted_before_registration:
                    _terminate(process)
                    raise CodexAbortedError("Codex 子进程已被全局中止")
                try:
                    stdout, stderr = process.communicate(prompt, timeout=self.timeout)
                except subprocess.TimeoutExpired as exc:
                    _terminate(process)
                    stdout, stderr = process.communicate()
                    last_error = f"Codex 调用超时（{self.timeout:g} 秒）"
                finally:
                    with _ACTIVE_LOCK:
                        _ACTIVE.discard(process)
                if _ABORT.is_set():
                    raise CodexAbortedError("Codex 子进程已被全局中止")
                events = parse_events(stdout or "")
                warnings = list(events.get("config_drift", []))
                for line in (stderr or "").splitlines():
                    if re.search(r"ignoring\s+\d+\s+unrecognized configuration settings", line, re.I):
                        warnings.append(line[:300])
                events["config_drift"] = warnings
                if process.returncode == 0 and output_path.exists():
                    try:
                        output = json.loads(output_path.read_text(encoding="utf-8"))
                    except (ValueError, OSError) as exc:
                        self._record(
                            prompt, events, result="output_format", retry=attempt,
                            duration_seconds=max(0.0, self._monotonic() - attempt_started),
                        )
                        raise CodexOutputFormatError("Codex 输出文件不是有效 JSON") from exc
                    if not events.get("errors"):
                        self._record(
                            prompt, events, result="success", retry=attempt,
                            duration_seconds=max(0.0, self._monotonic() - attempt_started),
                        )
                        return CodexResult(output=output, events=events, argv=argv)
                category = _classify(
                    (stderr or "") + "\n" + _failure_text(stdout or ""),
                    process.returncode or 0,
                )
                # 原样错误输出和提示可能含敏感内容，因此日志只保存分类结果。
                last_error = f"Codex {category} failure (exit {process.returncode})"
                self._record(
                    prompt, events, result=category, retry=attempt,
                    duration_seconds=max(0.0, self._monotonic() - attempt_started),
                )
                self._raise_classified(category, process.returncode)
            if attempt < self.retries:
                delay = delays[min(attempt, len(delays) - 1)]
                self._sleep(delay)
        raise CodexTransientExhaustedError(last_error)

    def _raise_classified(self, category: str, returncode: int | None) -> None:
        if category == "fatal_config":
            abort_all_calls()
            raise CodexFatalConfigError(
                f"Codex 配置或模型不可用（exit {returncode}）。请检查 codex_binary、所选模型是否受当前 ChatGPT 账户支持，并确认 Codex CLI 版本满足要求。"
            )
        if category == "quota":
            raise CodexQuotaError(f"Codex 额度或请求频率已达到限制（exit {returncode}）")

    def _record(
        self, prompt: str, events: dict[str, Any], *, result: str, retry: int,
        duration_seconds: float,
    ) -> None:
        if self.usage_log_path:
            path = Path(self.usage_log_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            context = _USAGE_CONTEXT.get()
            row = {
                "timestamp": datetime.now(UTC).isoformat(),
                "ticker": context.get("ticker"),
                "role": context.get("role"),
                "call_type": context.get("call_type"),
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "tokens": events.get("usage", {}),
                "result": result,
                "retry": retry,
                "duration_seconds": duration_seconds,
                "warnings": events.get("config_drift", []) + events.get("errors", []),
                "agent_actions": events.get("agent_actions", []),
            }
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.prompt_log_dir:
            directory = Path(self.prompt_log_dir).expanduser()
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "prompts.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
