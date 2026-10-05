"""Codex CLI strict 输出 Schema 构造。"""

from __future__ import annotations

from typing import Any
import json


class _DecisionLegSchema(dict):
    """JSON仍是严格生成契约；私有属性仅绑定真实决策模型的既有腿软验收。"""

    def __init__(self, strict, model):
        super().__init__(strict)
        self._leg_models = {name: model.model_fields[name].annotation for name in ('buy_legs', 'reduce_legs')}
        self._strict_snapshot = json.dumps(strict, sort_keys=True, ensure_ascii=False)

    def validation_projection(self, value):
        from .errors import CodexOutputFormatError
        from tradingagents.agents.price_plan_legs import normalize_legs, BuyLeg, ReduceLeg
        if json.dumps(self, sort_keys=True, ensure_ascii=False) != self._strict_snapshot:
            raise CodexOutputFormatError('决策schema已改变，腿软合同不可复用')
        if not isinstance(value, dict):
            return value
        props = self.get('properties', {})
        # 顶层required与全部未知属性必须在归一补字段/删除字段之前验证。
        missing = set(self.get('required', [])) - set(value)
        unknown = set(value) - set(props)
        if missing or unknown:
            raise CodexOutputFormatError(f'决策参数缺失或未知：missing={sorted(missing)}, unknown={sorted(unknown)}')
        for name, leg_model in (('buy_legs', BuyLeg), ('reduce_legs', ReduceLeg)):
            legs = value.get(name)
            if isinstance(legs, list):
                for leg in legs:
                    if isinstance(leg, dict) and (extra := set(leg) - set(leg_model.model_fields)):
                        raise CodexOutputFormatError(f'{name} 包含未知参数：{sorted(extra)}')
        normalized = normalize_legs({name: value.get(name) for name in self._leg_models})
        # 非腿字段完整保留原值；原输出不改，随后模型before校验才能留raw及flags。
        return {**value, **{name: normalized[name] for name in self._leg_models}}


CONTENT_SCHEMA = {
    "type": "object",
    "properties": {"content": {"type": "string"}},
    "required": ["content"],
    "additionalProperties": False,
}

TOOL_CALLS_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["final", "tool_calls"]},
        "content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments_json": {"type": "string"},
                },
                "required": ["name", "arguments_json"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["kind", "content", "tool_calls"],
    "additionalProperties": False,
}


def _inline_refs(schema: dict[str, Any], definitions: dict[str, Any], seen=()) -> dict[str, Any]:
    ref = schema.get("$ref")
    if ref:
        prefix = "#/$defs/"
        if not ref.startswith(prefix):
            raise ValueError(f"不支持的 JSON Schema 引用：{ref}")
        key = ref[len(prefix):]
        if key in seen or key not in definitions:
            raise ValueError(f"无法安全展开 JSON Schema 引用：{ref}")
        return _inline_refs(definitions[key], definitions, (*seen, key))
    result = {}
    for key, value in schema.items():
        if key in {"$defs", "definitions", "$schema", "title"}:
            continue
        if isinstance(value, dict):
            result[key] = _inline_refs(value, definitions, seen)
        elif isinstance(value, list):
            result[key] = [
                _inline_refs(item, definitions, seen) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """将常见 JSON Schema 转换为 Codex strict 模式接受的形式。"""
    unsupported = {"allOf", "not", "patternProperties", "unevaluatedProperties"}
    if unsupported.intersection(schema):
        raise ValueError("JSON Schema 使用了 strict 模式不支持的构造")
    definitions = schema.get("$defs", schema.get("definitions", {}))
    result = _inline_refs(schema, definitions)
    props = result.get("properties")
    if isinstance(props, dict):
        old_required = set(result.get("required", []))
        strict_props = {}
        for key, value in props.items():
            prop = strict_schema(value) if isinstance(value, dict) else value
            if key not in old_required:
                prop = {"anyOf": [prop, {"type": "null"}]}
            strict_props[key] = prop
        result["required"] = list(props)
        result["properties"] = strict_props
    if result.get("type") == "object":
        result["additionalProperties"] = False
    if isinstance(result.get("items"), dict):
        result["items"] = strict_schema(result["items"])
    if isinstance(result.get("anyOf"), list):
        result["anyOf"] = [strict_schema(item) for item in result["anyOf"]]
    return result


def schema_for(schema: Any) -> dict[str, Any]:
    try:
        if hasattr(schema, "model_json_schema"):
            raw = schema.model_json_schema()
        elif hasattr(schema, "schema"):
            raw = schema.schema()
        elif isinstance(schema, dict):
            raw = schema
        else:
            raise TypeError(f"不支持的结构化输出 schema：{schema!r}")
        strict = strict_schema(raw)
        # dict输入、普通工具、RM及关闭腿模型不拥有受信类型身份。
        if isinstance(schema, type):
            from tradingagents.agents.schemas import _LEG_DECISION_MODELS
            if schema in _LEG_DECISION_MODELS:
                return _DecisionLegSchema(strict, schema)
        return strict
    except (TypeError, ValueError):
        return {
            "type": "object",
            "properties": {"json": {"type": "string"}},
            "required": ["json"],
            "additionalProperties": False,
        }
