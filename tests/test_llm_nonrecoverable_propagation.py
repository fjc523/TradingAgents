from __future__ import annotations

import pytest

from tradingagents.agents.structured import invoke_structured, invoke_structured_or_freetext
from tradingagents.llm_clients.codex_exec.errors import CodexOutputFormatError, CodexQuotaError
from tradingagents.llm_clients.errors import LLMNonRecoverableError
from tradingagents.memory import settlement


@pytest.mark.unit
def test_structured_llm_quota_is_not_swallowed_but_format_error_can_fallback():
    class Failing:
        def __init__(self, error):
            self.error = error

        def invoke(self, prompt):
            raise self.error

    with pytest.raises(CodexQuotaError):
        invoke_structured(Failing(CodexQuotaError("offline quota")), "p", "test")
    assert invoke_structured(Failing(ValueError("bad structure")), "p", "test") is None
    assert not issubclass(CodexOutputFormatError, LLMNonRecoverableError)

    class Plain:
        calls = 0

        def invoke(self, prompt):
            self.calls += 1
            return type("Response", (), {"content": "fallback"})()

    plain = Plain()
    result = invoke_structured_or_freetext(
        Failing(CodexOutputFormatError("bad schema")), plain, "p", str, "test"
    )
    assert result == "fallback" and plain.calls == 1


@pytest.mark.unit
def test_settlement_propagates_nonrecoverable_reflection_failure(monkeypatch):
    class Memory:
        def get_pending_entries(self):
            return [{"ticker": "AAPL", "date": "2020-01-01", "decision": "hold"}]

    class Reflector:
        def reflect_on_final_decision(self, **kwargs):
            raise CodexQuotaError("offline quota")

    monkeypatch.setattr(settlement, "fetch_returns", lambda *args, **kwargs: (0.1, 0.05, 5, "2020-01-08"))
    with pytest.raises(CodexQuotaError):
        settlement.settle_pending("AAPL", Memory(), Reflector(), {})
