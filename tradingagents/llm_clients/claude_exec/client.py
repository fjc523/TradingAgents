"""订阅CLI后端复用成熟LangChain JSON/schema校验，不提供分析师工具接口。"""
from tradingagents.llm_clients.base_client import BaseLLMClient
from tradingagents.llm_clients.codex_exec.chat_model import CodexExecChatModel
from .runner import ClaudeExecRunner

class ClaudeExecChatModel(CodexExecChatModel):
    @property
    def _llm_type(self):return 'claude_exec'

    def bind_tools(self,*args,**kwargs):
        raise ValueError('Claude仅决策角色，禁止分析师工具绑定')

class ClaudeExecClient(BaseLLMClient):
    provider='claude_exec'
    def get_llm(self):
        runner=ClaudeExecRunner(model=self.model,reasoning_effort=self.kwargs.get('reasoning_effort','high'),
            binary=self.kwargs.get('claude_binary','~/.local/bin/claude'),timeout=self.kwargs.get('claude_timeout',300),
            retries=self.kwargs.get('claude_retries',1),max_concurrency=self.kwargs.get('claude_max_concurrency',4),
            usage_log_path=self.kwargs.get('codex_usage_log_path'))
        runner.role=self.kwargs.get('role')
        return ClaudeExecChatModel(model_name=self.model,reasoning_effort=runner.reasoning_effort,runner=runner)
    def validate_model(self):return self.model=='claude-opus-5-5'
