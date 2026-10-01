"""错误传播分类，供编排层区分不可恢复的 LLM 故障。"""


class LLMNonRecoverableError(RuntimeError):
    """不可由结构化输出自由文本兜底掩盖的 LLM 错误。"""
