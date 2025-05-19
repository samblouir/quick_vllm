"""vllm_client.py
================
A lightweight, object‑oriented wrapper around the helper functions you already
have.  Each instance manages its own connection (host + port) so you can spin
up **multiple** clients against different vLLM back‑ends in the *same* Python
process.

Example
-------
>>> from vllm_client import VLLMClient
>>> east = VLLMClient(port=8000, host="10.0.0.11")
>>> west = VLLMClient(port=8000, host="10.0.0.12")
>>>
>>> east.send_message("Hello, east!")
>>> west.send_message("Hello, west!", just_return_text=True)

The public surface mirrors the original helper functions (`send_message`,
`send_message_no_cache`, `send`) but without global state so the same script
can talk to many servers concurrently.
"""
from __future__ import annotations

import base64
import copy
import datetime as _dt
import multiprocessing as _mp
import os
import time
import traceback
from typing import Any, Iterable, List

from openai import OpenAI

# Your existing utility imports ------------------------------------------------
from quick_vllm import cache  # type: ignore
from quick_vllm.utils import arg_dict  # type: ignore

__all__ = ["VLLMClient"]


class VLLMClient:
    """Stateful wrapper around a single vLLM HTTP endpoint."""

    _DEFAULT_SAMPLING = {
        "min_p": 0.05,
        "top_p": 0.95,
        "temperature": 0.7,
        "top_k": 40,
        # *Always* filled in `__init__` using `arg_dict` so it's never ``None``.
        "max_tokens": None,  # type: ignore [assignment]
    }

    # ---------------------------------------------------------------------
    # Construction & representation
    # ---------------------------------------------------------------------
    def __init__(
        self,
        port: int | str = 8000,
        host: str = "localhost",
        *,
        mdl: str | None = None,
        default_sampling_parameters: dict[str, Any] | None = None,
    ) -> None:
        """Create a new client bound to *host:port*.

        Parameters
        ----------
        port, host : network location of the vLLM HTTP server.
        mdl         : explicit model ID.  If ``None`` the first model returned
                      by the server is cached lazily on first use.
        default_sampling_parameters : per‑instance overrides applied *before*
                      parameters passed to :py:meth:`send_message`.
        """
        self.host = host
        self.port = int(port)
        self.base_url = f"http://{self.host}:{self.port}/v1/"
        self.client = OpenAI(base_url=self.base_url, api_key="vllm")

        # Model is resolved lazily because the server might still be loading.
        self._mdl: str | None = mdl

        # Merge caller overrides with the library defaults.
        self.default_sampling_parameters = copy.deepcopy(self._DEFAULT_SAMPLING)
        if default_sampling_parameters:
            self.default_sampling_parameters.update(default_sampling_parameters)

        # Inject *max_tokens* early so it's never None later on.
        self.default_sampling_parameters["max_tokens"] = int(
            arg_dict.get("max_tokens", 4096)
        )

    # ------------------------------------------------------------------
    # Static helpers (no access to self required)
    # ------------------------------------------------------------------
    @staticmethod
    def encode_image(path: str) -> str:
        """Return *path* as base‑64‑encoded **utf‑8** string suitable for JSON."""
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode()

    @staticmethod
    def _strip_none_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop chat messages whose *content* key is ``None``."""
        return [m for m in messages if m.get("content") is not None]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _model_id(self) -> str:
        """Resolve and cache the model ID from ``/models``."""
        while self._mdl is None:
            try:
                self._mdl = self.client.models.list()[0].id
            except Exception as exc:  # pragma: no cover – network retries
                print(
                    f"[{_dt.datetime.now():%Y-%m-%d %H:%M:%S}] "
                    f"{self.base_url}: failed to fetch model list -> {exc}"
                )
                time.sleep(1)
        return self._mdl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def send_message(
        self,
        msg: str | list[dict[str, Any]],
        system_prompt: str | None = None,
        *,
        stream: bool = True,
        force_cache_miss: bool = False,
        just_return_text: bool = False,
        silent: bool = True,
        **kw: Any,
    ) -> Any:
        """One‑shot chat request (close match to original helper)."""

        # Normalise *messages* into openai‑style array‑of‑dicts -----------------
        if isinstance(msg, str):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ]
        else:
            messages = msg

        messages = self._strip_none_messages(messages)

        # Build *sampling* dict: instance defaults < call overrides -------------
        sampling = copy.deepcopy(self.default_sampling_parameters)
        sampling.update({k: v for k, v in kw.items() if v is not None})

        settings = {
            "model": kw.get("model", self._model_id()),
            "messages": messages,
            "stream": stream,
            "extra_body": sampling,
        }

        return self._run(settings, force_cache_miss, silent, just_return_text)

    # ------------------------------------------------------------------
    def send(
        self,
        msgs: str | Iterable[str],
        *,
        max_pool_size: int | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Parallel batched variant mirroring the old *send* helper."""
        if isinstance(msgs, str):
            msgs = [msgs]

        max_pool_size = min(max_pool_size or _mp.cpu_count(), len(msgs))
        pool = _mp.Pool(processes=max_pool_size)

        try:
            return pool.map(
                lambda m: self.send_message(m, **kwargs),  # noqa: E731
                msgs,
            )
        finally:
            pool.close()
            pool.join()

    # ------------------------------------------------------------------
    # Core request/response logic (adapted from *_run_message*)
    # ------------------------------------------------------------------
    def _run(
        self,
        settings: dict[str, Any],
        force_cache_miss: bool,
        silent: bool,
        just_return_text: bool,
    ) -> Any:
        timeout_retry = 5

        # Cache key excludes *stream* -----------------------------------------
        cache_key = {k: v for k, v in settings.items() if k != "stream"}
        cache_path = cache.quick_hash(cache_key)

        if not (force_cache_miss or int(arg_dict.get("disable_cache", 0))):
            try:
                cached = cache.quick_load(cache_path)
                if isinstance(cached, list) and cached:
                    return [c["text"] for c in cached] if just_return_text else cached
            except Exception:
                # Any error => ignore cache and fall through.
                pass

        # ------------------------------------------------------------------
        # Make the real network request
        # ------------------------------------------------------------------
        while True:
            try:
                completion = self.client.chat.completions.create(
                    **settings, timeout=999_999
                )
                break
            except Exception as exc:
                print(f"{__file__}: retrying after error -> {exc}")
                time.sleep(timeout_retry)
                timeout_retry = min(timeout_retry + 5, 30)

        # Aggregate streamed or non‑streamed responses ------------------------
        if settings["stream"]:
            chunks: dict[int, list[str]] = {}
            for chunk in completion:
                choice = chunk.choices[0]
                if choice.delta.content is not None:
                    chunks.setdefault(choice.index, []).append(choice.delta.content)
            texts = ["".join(t) for _, t in sorted(chunks.items())]
        else:
            texts = [c.message.content or "" for c in completion.choices]

        packaged = [
            {
                "text": t,
                "settings": {k: v for k, v in settings.items() if k != "stream"},
            }
            for t in texts
        ]

        if not int(arg_dict.get("disable_cache", 0)):
            cache.quick_save(packaged, cache_path)

        return texts if just_return_text else packaged

    # ------------------------------------------------------------------
    # Dunder helpers ----------------------------------------------------
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<VLLMClient host={self.host!r} port={self.port} "
            f"model={self._mdl or '(lazy)'}>"
        )

if __name__ == "__main__":
	# Test the VLLMClient class
	client = VLLMClient()
	print(client)
	print(client.send_message("Hello, world!"))
	print(client.send(["Hello, world!", "How are you?"]))