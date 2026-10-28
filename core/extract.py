"""What a recording is allowed to become.

A claim is one thing a family needs to know: who the giỗ is for, on which lunar date, what goes on the
tray, and in what order. Every claim carries the words that produced it and the moment in the recording
they were said. A claim that cannot carry both does not survive `core.guard`.

That is the whole architecture of the product: the model is allowed to *suggest* claims, and only code
decides which of them exist.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict

from core import textnorm, lunar


@dataclass
class Claim:
    kind: str                       # person | gio_date | offering | instruction
    quote: str                      # verbatim words from the recording
    t_start: float | None = None    # citation: seconds into the recording
    t_end: float | None = None
    lunar_day: int | None = None
    lunar_month: int | None = None
    leap: bool = False
    solar_next: str | None = None   # ISO date of the next occurrence, computed not generated
    status: str = "suggested"       # suggested | kept | refused
    reason: str = ""
    model: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def citation(self) -> str:
        if self.t_start is None:
            return ""
        return f"{_mmss(self.t_start)}–{_mmss(self.t_end or self.t_start)}"


def _mmss(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"


EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["person", "gio_date", "offering", "instruction"]},
                    "quote": {"type": "string"},
                    "lunar_day": {"type": ["integer", "null"]},
                    "lunar_month": {"type": ["integer", "null"]},
                    "leap": {"type": "boolean"},
                },
                "required": ["kind", "quote"],
            },
        }
    },
    "required": ["claims"],
}

PROMPT = """Bạn đang giúp một gia đình Việt Nam ghi lại ngày giỗ của người đã mất.

Dưới đây là bản ghi lời nói, chia theo mốc thời gian. Nhiệm vụ: rút ra danh sách các điều gia đình cần biết.

Quy tắc bắt buộc:
- Trường "quote" phải là một đoạn NGUYÊN VĂN, chép đúng từng chữ từ bản ghi bên dưới. Không được diễn giải, không được sửa, không được thêm.
- Nếu bản ghi không nói rõ một điều gì đó, ĐỪNG đoán và ĐỪNG bịa. Chỉ bỏ qua.
- "kind" chỉ được là một trong: person, gio_date, offering, instruction.
- Với gio_date: điền "lunar_day" và "lunar_month" nếu bản ghi nói rõ (ví dụ "mười hai tháng bảy" là ngày 12 tháng 7 âm lịch), nếu không rõ thì để null.

Trả về JSON đúng dạng: {{"claims": [{{"kind": "...", "quote": "...", "lunar_day": null, "lunar_month": null, "leap": false}}]}}

BẢN GHI:
{transcript}
"""


def build_prompt(pieces: list[tuple[float, str]]) -> str:
    """`pieces` is [(start_seconds, text)] — only the spans that survived the two-model agreement."""
    lines = [f"[{_mmss(t)}] {text}" for t, text in pieces]
    return PROMPT.format(transcript="\n".join(lines))


_FENCE = re.compile(r"^```(?:json)?|```$", re.MULTILINE)


def parse_model_output(raw: str) -> list[Claim]:
    """Be forgiving about the wrapper, strict about the content."""
    text = _FENCE.sub("", raw).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    out = []
    for item in data.get("claims", []) or []:
        if not isinstance(item, dict):
            continue
        quote = (item.get("quote") or "").strip()
        if not quote:
            continue
        out.append(Claim(
            kind=str(item.get("kind", "")).strip() or "unknown",
            quote=quote,
            lunar_day=_as_int(item.get("lunar_day")),
            lunar_month=_as_int(item.get("lunar_month")),
            leap=bool(item.get("leap", False)),
        ))
    return out


def _as_int(value) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def ollama_extract(prompt: str, model: str = "gemma3:1b-it-qat", host: str = "http://127.0.0.1:11434",
                   timeout: int = 240) -> tuple[list[Claim], dict]:
    """Ask a local open-weight model for claims. Returns (claims, usage)."""
    import time
    import urllib.request

    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 900, "seed": 7},
    }).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    dt = time.time() - t0
    claims = parse_model_output(payload.get("response", ""))
    for c in claims:
        c.model = model
    usage = {
        "model": model,
        "latency_s": round(dt, 2),
        "prompt_tokens": payload.get("prompt_eval_count"),
        "completion_tokens": payload.get("eval_count"),
        "raw": payload.get("response", "")[:4000],
    }
    return claims, usage
