"""Request schemas for the OpenAI-compatible API (the subset the engine supports)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StreamOptions(BaseModel):
    include_usage: bool = False


class _SamplingFields(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    top_k: int = Field(-1, ge=-1)
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    ignore_eos: bool = False
    stream: bool = False
    stream_options: StreamOptions | None = None
    n: int = 1
    # Accepted so clients that always send them get a clear error, not silent misbehaviour.
    logprobs: Any = None
    seed: int | None = None

    def stop_strings(self) -> list[str]:
        if self.stop is None:
            return []
        return [self.stop] if isinstance(self.stop, str) else list(self.stop)

    def unsupported(self) -> str | None:
        if self.n != 1:
            return "n > 1 is not supported"
        if self.logprobs not in (None, False, 0):
            return "logprobs are not supported"
        if self.seed is not None:
            return "per-request seeds are not supported"
        return None


class CompletionRequest(_SamplingFields):
    prompt: str | list[int]
    max_tokens: int = Field(16, ge=1)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[dict[str, Any]] | None = None

    def text(self) -> str:
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "".join(part.get("text", "") for part in self.content if part.get("type") == "text")


class ChatCompletionRequest(_SamplingFields):
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(None, ge=1)
