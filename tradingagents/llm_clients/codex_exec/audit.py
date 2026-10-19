"""实际模型输入审计：发送前脱敏、独立调用文件与受控权限。"""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import uuid


def redact(text):
    for key, value in os.environ.items():
        if len(value) >= 8 and any(word in key.upper() for word in ("API_KEY", "TOKEN", "PASSWORD", "SECRET")):
            text = text.replace(value, "[已脱敏]")
    def clean(node):
        if isinstance(node, dict):
            return {key: "[已脱敏]" if re.fullmatch(r"(?i)(?:authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|client[_-]?secret)", str(key)) else clean(value) for key, value in node.items()}
        if isinstance(node, list):
            return [clean(value) for value in node]
        if isinstance(node, str):
            try:
                nested = json.loads(node)
            except (ValueError, TypeError):
                return re.sub(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s\"']+", r"\1[已脱敏]", node)
            if isinstance(nested, (dict, list)):
                return json.dumps(clean(nested), ensure_ascii=False)
        return node
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return clean(text)
    return json.dumps(clean(value), ensure_ascii=False)


def write_record(directory, record, *, call_id=None):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    identity = call_id or uuid.uuid4().hex
    if not re.fullmatch(r"[a-f0-9]{32}(?:-output)?", identity):
        raise ValueError("审计call_id必须是本地生成的安全标识")
    data = redact(json.dumps(record, ensure_ascii=False, default=str))
    envelope = {"call_id": identity, "sha256": sha256(data.encode()).hexdigest(), "record": json.loads(data)}
    path = root / (identity + ".json")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(envelope, ensure_ascii=False) + "\n")
    return identity


def market_payload(prompt):
    """兼容旧调用；市场契约已在角色模板/schema构造处绑定，发送层不改事实。"""
    return prompt
