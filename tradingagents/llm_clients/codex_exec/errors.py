"""Codex CLI 后端的错误类型。"""

from tradingagents.llm_clients.errors import LLMNonRecoverableError


class CodexError(LLMNonRecoverableError):
    """Codex 执行错误基类。"""


class CodexFatalConfigError(CodexError):
    """二进制或运行配置错误；不能通过重试恢复。"""


class CodexQuotaError(CodexError):
    """Codex 额度不足。"""


class CodexAbortedError(CodexError):
    """全局中止或调用中止。"""


class CodexTransientExhaustedError(CodexError):
    """瞬时故障重试耗尽。"""


class CodexOutputFormatError(ValueError):
    """Codex 返回值不符合约定格式，可由上游格式兜底处理。"""


class CodexToolCallError(CodexOutputFormatError):
    """工具调用经过一次纠错后仍不符合绑定的工具定义。"""
