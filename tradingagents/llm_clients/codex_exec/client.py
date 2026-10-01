"""CodexExecClient 与上游 BaseLLMClient 对接。"""

from tradingagents.llm_clients.base_client import BaseLLMClient
from tradingagents.llm_clients.validators import validate_model

from .chat_model import CodexExecChatModel
from .runner import CodexExecRunner


class CodexExecClient(BaseLLMClient):
    provider = "codex_exec"

    def get_llm(self):
        runner = CodexExecRunner(
            binary=self.kwargs.get("codex_binary", "codex"),
            model=self.model,
            reasoning_effort=self.kwargs.get("reasoning_effort", "high"),
            timeout=self.kwargs.get("codex_timeout", 600),
            retries=self.kwargs.get("codex_retries", 3),
            max_concurrency=self.kwargs.get("codex_max_concurrency", 4),
            usage_log_path=self.kwargs.get("codex_usage_log_path"),
            prompt_log_dir=self.kwargs.get("codex_prompt_log_dir"),
        )
        runner.role = self.kwargs.get("role")
        return CodexExecChatModel(
            model_name=self.model,
            reasoning_effort=self.kwargs.get("reasoning_effort", "high"),
            runner=runner,
        )

    def validate_model(self) -> bool:
        return validate_model("codex_exec", self.model)
