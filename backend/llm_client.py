"""
llm_client.py — Gemini-backed shim that mimics the Groq Python client surface
this codebase already uses.

Drop-in replacement:
    from llm_client import GeminiClient as Groq
    client = Groq()
    resp   = client.chat.completions.create(
        model="gemini-3.5-flash",
        messages=[{"role": "system", ...}, {"role": "user", ...}],
        temperature=0.8,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,           # or True
        stop=None,
    )
    text  = resp.choices[0].message.content               # non-stream
    for chunk in resp:                                    # stream
        delta = chunk.choices[0].delta.content

Auth:
    GEMINI_API_KEY (preferred) or GOOGLE_API_KEY

Required dep:
    pip install google-generativeai
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Iterator, List, Dict, Optional

try:
    import google.generativeai as genai
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "google-generativeai not installed. Run: pip install google-generativeai"
    ) from e


# ── value objects (duck-type Groq response shapes) ────────────────────────────
class _Message:
    __slots__ = ("content", "role")

    def __init__(self, content: str, role: str = "assistant"):
        self.content = content
        self.role = role


class _Delta:
    __slots__ = ("content",)

    def __init__(self, content: str):
        self.content = content


class _Choice:
    __slots__ = ("message", "delta", "index", "finish_reason")

    def __init__(self, *, message: Optional[_Message] = None,
                 delta: Optional[_Delta] = None,
                 index: int = 0, finish_reason: Optional[str] = None):
        self.message = message
        self.delta = delta
        self.index = index
        self.finish_reason = finish_reason


class _Completion:
    """Non-streaming response: `.choices[0].message.content`."""
    __slots__ = ("choices", "model")

    def __init__(self, text: str, model: str):
        self.choices = [_Choice(message=_Message(content=text or ""))]
        self.model = model


class _StreamChunk:
    """Streaming chunk: `.choices[0].delta.content`."""
    __slots__ = ("choices", "model")

    def __init__(self, text: str, model: str, finish_reason: Optional[str] = None):
        self.choices = [_Choice(delta=_Delta(content=text or ""),
                                finish_reason=finish_reason)]
        self.model = model


# ── message converter (OpenAI/Groq → Gemini) ──────────────────────────────────
def _split_system(messages: List[Dict[str, str]]) -> tuple[str, List[Dict[str, Any]]]:
    """
    Collapse all system messages into a single systemInstruction string and
    convert remaining messages into Gemini `contents` list format.

    Gemini roles: "user" | "model". OpenAI roles: "system" | "user" | "assistant".
    """
    system_parts: List[str] = []
    contents: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content)

        if role == "system":
            if content:
                system_parts.append(content)
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})
        else:  # user (and any other role) → "user"
            contents.append({"role": "user", "parts": [{"text": content}]})

    # Gemini rejects empty `contents`. Inject sentinel if caller sent system-only.
    if not contents:
        contents.append({"role": "user", "parts": [{"text": ""}]})

    return ("\n\n".join(system_parts), contents)


# ── completions namespace ─────────────────────────────────────────────────────
class _Completions:
    def __init__(self, client: "GeminiClient"):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,             # alt name some callers use
        top_p: float = 1.0,
        stream: bool = False,
        stop: Any = None,
        **_unused: Any,
    ) -> Any:
        system_instruction, contents = _split_system(messages)

        generation_config: Dict[str, Any] = {
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        max_tok = max_completion_tokens or max_tokens
        # if max_tok:
        #     generation_config["max_output_tokens"] = int(max_tok)
        if stop:
            generation_config["stop_sequences"] = (
                list(stop) if isinstance(stop, (list, tuple)) else [str(stop)]
            )

        gm = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_instruction or None,
        )

        if stream:
            iterator = gm.generate_content(
                contents,
                generation_config=generation_config,
                stream=True,
            )
            return _stream_iter(iterator, model)

        resp = gm.generate_content(
            contents,
            generation_config=generation_config,
            stream=False,
        )
        text = _extract_text(resp)
        return _Completion(text, model)


def _extract_text(resp: Any) -> str:
    """Pull text out of Gemini response, tolerant of empty/blocked outputs."""
    try:
        t = getattr(resp, "text", None)
        if t:
            return t
    except Exception:
        pass
    out: List[str] = []
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            txt = getattr(part, "text", None)
            if txt:
                out.append(txt)
    return "".join(out)


def _stream_iter(iterator: Iterable[Any], model: str) -> Iterator[_StreamChunk]:
    for chunk in iterator:
        text = _extract_text(chunk)
        if text:
            yield _StreamChunk(text, model)
    # final chunk with finish_reason to mirror Groq behaviour
    yield _StreamChunk("", model, finish_reason="stop")


class _Chat:
    def __init__(self, client: "GeminiClient"):
        self.completions = _Completions(client)


# ── public client ─────────────────────────────────────────────────────────────
class GeminiClient:
    """
    Groq-compatible client backed by Gemini.

    Auth resolution order:
        1. explicit `api_key=` kwarg
        2. env GEMINI_API_KEY
        3. env GOOGLE_API_KEY
        4. env GROQ_API_KEY (legacy slot, only if it looks like a Gemini key)
    """

    _configured = False

    def __init__(self, api_key: Optional[str] = None, **_unused: Any):
        key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not key:
            legacy = os.getenv("GROQ_API_KEY", "")
            if legacy and not legacy.startswith("gsk_"):
                key = legacy
        if not key:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY)."
            )

        if not GeminiClient._configured:
            genai.configure(api_key=key)
            GeminiClient._configured = True

        self.chat = _Chat(self)
