"""LangChain 聊天模型适配器。"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field

from .errors import CodexOutputFormatError, CodexToolCallError
from .runner import CodexExecRunner
from .schemas import CONTENT_SCHEMA, TOOL_CALLS_SCHEMA, schema_for


class CodexExecChatModel(BaseChatModel):
    """以 Codex CLI JSON 输出实现 LangChain 的 `_generate`。"""

    model_name: str
    reasoning_effort: str = "high"
    runner: Any = Field(exclude=True)
    bound_tools: list[dict[str, Any]] = Field(default_factory=list, exclude=True)
    response_schema: dict[str, Any] | None = Field(default=None, exclude=True)
    output_mode: str = "text"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "codex_exec"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "reasoning_effort": self.reasoning_effort}

    def bind_tools(
        self, tools: Sequence[Any], *, tool_choice: str | None = None, **kwargs: Any,
    ) -> CodexExecChatModel:
        if tool_choice not in (None, "auto"):
            raise ValueError("codex_exec 目前只支持自动选择工具")
        converted = [convert_to_openai_tool(tool) for tool in tools]
        return self.model_copy(update={
            "bound_tools": converted,
            "output_mode": "tools",
            "response_schema": TOOL_CALLS_SCHEMA,
        })

    def with_structured_output(
        self, schema: type | dict, *, include_raw: bool = False, **kwargs: Any,
    ):
        json_schema = schema_for(schema)
        model = self.model_copy(update={"response_schema": json_schema, "output_mode": "structured"})

        def invoke(value):
            raw = model.invoke(value)
            try:
                parsed = _parse_structured(raw.content, schema)
            except Exception as exc:
                if include_raw:
                    return {"raw": raw, "parsed": None, "parsing_error": exc}
                raise CodexOutputFormatError(f"Codex 结构化结果无法解析：{exc}") from exc
            if include_raw:
                return {"raw": raw, "parsed": parsed, "parsing_error": None}
            return parsed

        return RunnableLambda(invoke)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._prompt(messages)
        usage_context = getattr(self.runner, "role", None)
        # 主项目可在 runner 外标记标的，避免修改批次共享配置。
        from .runner import set_usage_context, reset_usage_context
        token = set_usage_context(role=usage_context, call_type=self.output_mode)
        schema = self.response_schema or CONTENT_SCHEMA
        last_error = None
        result = None
        try:
            for correction in range(2 if self.output_mode == "tools" else 1):
                try:
                    result = self.runner.run(prompt, schema)
                    message = self._message(result.output, result.events)
                    if self.output_mode == "tools":
                        self._validate_tool_calls(message.tool_calls)
                    break
                except CodexOutputFormatError as exc:
                    if self.output_mode != "tools":
                        raise
                    last_error = exc
                    if correction:
                        raise CodexToolCallError(str(exc)) from exc
                    prompt += (
                        "\n\n上一次工具输出校验失败，请只修正后重新输出。错误："
                        + str(exc)
                    )
        finally:
            reset_usage_context(token)
        if result is None:
            raise CodexOutputFormatError(str(last_error or "Codex 未返回结果"))
        generation = ChatGeneration(message=message)
        return ChatResult(
            generations=[generation],
            llm_output={
                "model_name": self.model_name,
                "reasoning_effort": self.reasoning_effort,
                **result.events,
            },
        )

    def _prompt(self, messages: list[BaseMessage]) -> str:
        rows = []
        for message in messages:
            role = {
                "ai": "assistant", "human": "user", "system": "system",
                "tool": "tool", "function": "tool",
            }.get(message.type, message.type)
            row = {"role": role, "content": message.content}
            if getattr(message, "name", None):
                row["name"] = message.name
            if getattr(message, "tool_call_id", None):
                row["tool_call_id"] = message.tool_call_id
            if message.type == "ai" and getattr(message, "tool_calls", None):
                row["tool_calls"] = [
                    {
                        "id": call.get("id"), "name": call.get("name"),
                        "arguments": call.get("args", {}),
                    }
                    for call in message.tool_calls
                ]
            rows.append(row)
        payload: dict[str, Any] = {"messages": rows}
        if self.bound_tools:
            payload["tools"] = self.bound_tools
            payload["tool_choice"] = "auto"
        if self.response_schema:
            payload["output_schema"] = self.response_schema
        return json.dumps(payload, ensure_ascii=False, default=str)

    def _message(self, output: Any, events: dict[str, Any]) -> AIMessage:
        if not isinstance(output, dict):
            raise CodexOutputFormatError("Codex 输出必须是 JSON 对象")
        if self.output_mode == "tools":
            content = output.get("content", "")
            calls = output.get("tool_calls")
            kind = output.get("kind")
            if kind not in {"final", "tool_calls"} or not isinstance(calls, list):
                raise CodexOutputFormatError("工具响应必须包含 kind=final/tool_calls 和 tool_calls 数组")
            if kind == "final":
                if calls:
                    raise CodexOutputFormatError("kind=final 时 tool_calls 必须为空")
                if not isinstance(content, str):
                    raise CodexOutputFormatError("最终回答 content 必须为字符串")
                return AIMessage(content=content, response_metadata=events)
            tool_calls = []
            for index, call in enumerate(calls):
                if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                    raise CodexOutputFormatError("工具调用缺少有效 name")
                raw_args = call.get("arguments_json")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else None
                except ValueError as exc:
                    raise CodexOutputFormatError(f"工具参数不是有效 JSON：{call['name']}") from exc
                if not isinstance(args, dict):
                    raise CodexOutputFormatError(f"工具参数必须为 JSON 对象：{call['name']}")
                tool_calls.append({
                    "name": call["name"], "args": args,
                    "id": f"codex_call_{uuid.uuid4().hex}", "type": "tool_call",
                })
            return AIMessage(content=content, tool_calls=tool_calls, response_metadata=events)
        if self.output_mode == "structured":
            content = json.dumps(output, ensure_ascii=False)
        else:
            content = output.get("content")
            if not isinstance(content, str):
                raise CodexOutputFormatError("文本输出缺少字符串字段 content")
        return AIMessage(content=content, response_metadata=events)

    def _validate_tool_calls(self, calls: list[dict[str, Any]]) -> None:
        tools = {item["function"]["name"]: item["function"] for item in self.bound_tools}
        for call in calls:
            name, args = call["name"], call["args"]
            if name not in tools:
                raise CodexOutputFormatError(f"模型调用了未绑定的工具：{name}")
            schema = tools[name].get("parameters", {})
            definitions = schema.get("$defs", schema.get("definitions", {}))
            _validate_arguments(args, schema, name, definitions)


def _validate_arguments(value: Any, schema: dict[str, Any], path: str, definitions=None) -> None:
    definitions = definitions or {}
    if "$ref" in schema:
        ref = schema["$ref"]
        prefixes = ("#/$defs/", "#/definitions/")
        prefix = next((item for item in prefixes if ref.startswith(item)), None)
        if prefix is None:
            raise CodexOutputFormatError(f"{path} 使用不支持的 schema 引用 {ref}")
        key = ref[len(prefix):].replace("~1", "/").replace("~0", "~")
        if key not in definitions:
            raise CodexOutputFormatError(f"{path} 找不到 schema 引用 {ref}")
        return _validate_arguments(value, definitions[key], path, definitions)
    if "anyOf" in schema or "oneOf" in schema:
        options = schema.get("anyOf", schema.get("oneOf", []))
        for option in options:
            try:
                _validate_arguments(value, option, path, definitions)
                break
            except CodexOutputFormatError:
                continue
        else:
            raise CodexOutputFormatError(f"{path} 不符合允许的任一参数结构")
    if "const" in schema and value != schema["const"]:
        raise CodexOutputFormatError(f"{path} 必须等于 {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise CodexOutputFormatError(f"{path} 不在允许值列表中")
    expected = schema.get("type")
    matches = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
    }
    if expected in matches and not matches[expected](value):
        raise CodexOutputFormatError(f"{path} 类型应为 {expected}")
    if expected == "object":
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise CodexOutputFormatError(f"{path} 缺少必填参数 {key}")
        if schema.get("additionalProperties") is False:
            unknown = set(value) - set(props)
            if unknown:
                raise CodexOutputFormatError(f"{path} 包含未知参数：{sorted(unknown)}")
        for key, item in value.items():
            if key in props:
                _validate_arguments(item, props[key], f"{path}.{key}", definitions)
    if expected == "array" and isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            _validate_arguments(item, schema["items"], f"{path}[{index}]", definitions)
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", float("inf")):
            raise CodexOutputFormatError(f"{path} 字符长度不符合 schema")
        if schema.get("pattern") and not re.search(schema["pattern"], value):
            raise CodexOutputFormatError(f"{path} 不符合 schema pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < schema.get("minimum", float("-inf")) or value > schema.get("maximum", float("inf")):
            raise CodexOutputFormatError(f"{path} 数值范围不符合 schema")


def _parse_structured(content: Any, schema: type | dict):
    if isinstance(content, (dict, list)):
        value = content
    else:
        value = json.loads(str(content))
    if isinstance(value, dict) and set(value) == {"json"} and isinstance(value["json"], str):
        value = json.loads(value["json"])
    if hasattr(schema, "model_validate"):
        return schema.model_validate(value)
    if hasattr(schema, "parse_obj"):
        return schema.parse_obj(value)
    return value
