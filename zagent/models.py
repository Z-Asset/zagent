"""LLM clients for the three roles.

All roles talk to any OpenAI-compatible chat endpoint (DeepSeek, Qwen, GLM,
GPT, a local vLLM/Ollama server, ...) through the same thin client. The three
roles default to the same model and endpoint; a config can override any of
them independently (e.g. a cheap model for the operator, a stronger one for
the refiner).

Two offline stubs are provided so the loop can be exercised without any API:

  - LambdaModel: response computed by a Python callable (prompt-aware demo).
  - MockModel:   a single canned response.

The client uses only the stdlib (urllib), so the package has no third-party
runtime dependency.
"""
from __future__ import annotations

import json
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .harness.types import HarnessState


class ChatModel:
    def __init__(self, name: str = "", *, base_url: str = "", api_key: str = "", temperature: float = 0.0):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature

    def complete(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        """Send a chat request and return the assistant text.

        Synchronous and blocking. Raises RuntimeError with the URL/status so a
        reachability failure is diagnosable rather than silently swallowed.
        """
        import urllib.error
        import urllib.request

        payload: dict[str, Any] = {
            "model": self.name,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"model {self.name!r} HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"cannot reach {self.base_url}: {exc.reason}") from exc

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected response shape from {self.base_url}: {json.dumps(body)[:500]}") from exc


class MockModel(ChatModel):
    """A model that always returns one canned response. For offline smoke tests."""

    def __init__(self, response: str):
        super().__init__("mock")
        self._response = response

    def complete(self, messages, *, json_mode: bool = False) -> str:
        return self._response


class LambdaModel(ChatModel):
    """A model whose response is computed by a Python callable (offline demo)."""

    def __init__(self, fn: Callable[[list[dict[str, str]]], str]):
        super().__init__("lambda")
        self._fn = fn

    def complete(self, messages, *, json_mode: bool = False) -> str:
        return self._fn(messages)


def build_harness_messages(harness: HarnessState, user_prompt: str) -> list[dict[str, str]]:
    """Assemble the message list an operator sees: system + memory + skills + user."""
    system = harness.system_prompt
    if harness.memory:
        system += "\n\n## Memory\n" + "\n".join(f"- {m}" for m in harness.memory)
    if harness.skills:
        system += "\n\n## Skills\n" + "\n\n".join(
            f"### {name}\n{body}" for name, body in harness.skills.items()
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]
