"""角色订阅失败回退；批次共享熔断，仅允许已有Codex默认实例。"""
from datetime import UTC, datetime
import json
from pathlib import Path
import threading
from langchain_core.runnables import RunnableLambda
from tradingagents.llm_clients.claude_exec.runner import (
    ClaudeExecError, ClaudeQuotaError, ClaudeConfigError, ClaudeTransientError, _LOG_LOCK,
)
from tradingagents.llm_clients.codex_exec import runner as common


class RoleBatchBreaker:
    """同批次并发安全；额度/配置立即熔断，第二次暂态耗尽熔断。"""
    def __init__(self):
        self._lock = threading.Lock()
        self._open = False
        self._exhaustions = 0
        self._category = None

    def __deepcopy__(self, memo):
        """配置隔离复制不复制批次运行状态或线程锁。"""
        memo[id(self)] = self
        return self

    def snapshot(self):
        with self._lock:
            return {'open': self._open, 'exhaustions': self._exhaustions, 'category': self._category}

    def fail(self, error):
        category = 'quota' if isinstance(error, ClaudeQuotaError) else 'configuration' if isinstance(error, ClaudeConfigError) else getattr(error, 'category', 'transport')
        with self._lock:
            if isinstance(error, (ClaudeQuotaError, ClaudeConfigError)):
                self._open = True
            elif isinstance(error, ClaudeTransientError):
                self._exhaustions += 1
                self._open = self._open or self._exhaustions >= 2
            self._category = category
            return {'open': self._open, 'exhaustions': self._exhaustions, 'category': category}


class RoleFallbackLLM:
    """结构化与文本共用同一回退策略；解析错误及人工中止不回退。"""
    def __init__(self, primary, fallback, *, role, breaker, target, usage_log_path=None):
        self.primary = primary
        self.fallback = fallback
        self.role = role
        self.breaker = breaker
        self.target = target
        self.usage_log_path = usage_log_path

    def _invoke(self, primary, fallback, value, *args, **kwargs):
        if common._ABORT.is_set():
            from tradingagents.llm_clients.codex_exec.errors import CodexAbortedError
            raise CodexAbortedError('调用已被人工中止，不回退')
        state = self.breaker.snapshot()
        skipped = state['open']
        if not state['open']:
            try:
                return primary.invoke(value, *args, **kwargs)
            except (ClaudeQuotaError, ClaudeConfigError, ClaudeTransientError) as error:
                if common._ABORT.is_set():
                    raise
                state = self.breaker.fail(error)
        # 未分类错误不进入此处；记录控制事件不冒称一次模型消费。
        row = {'timestamp': datetime.now(UTC).isoformat(), **common._USAGE_CONTEXT.get(),
               'role': self.role, 'provider': 'claude_exec', 'result': 'fallback',
               'category': ('breaker:' if skipped else '') + str(state['category']), 'breaker': state,
               'from': 'claude_exec/claude-opus-5-5',
               'to': f"{self.target['provider']}/{self.target['model']}/{self.target['effort']}", 'model_requests': 0}
        if self.usage_log_path:
            path = Path(self.usage_log_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            with _LOG_LOCK, path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        with common.codex_usage_context(role=self.role):
            return fallback.invoke(value, *args, **kwargs)

    def invoke(self, value, *args, **kwargs):
        return self._invoke(self.primary, self.fallback, value, *args, **kwargs)

    def with_structured_output(self, schema, **kwargs):
        primary = self.primary.with_structured_output(schema, **kwargs)
        fallback = self.fallback.with_structured_output(schema, **kwargs)
        return RunnableLambda(lambda value: self._invoke(primary, fallback, value))
