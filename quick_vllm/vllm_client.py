"""vllm_client.py
================
Object‑oriented wrapper around the quick‑vllm functional helpers *without*
global state — now multiprocessing‑safe.

Key change in this revision
---------------------------
`VLLMClient.send()` no longer uses a `lambda` (which cannot be pickled).  We
introduce a top‑level helper `_worker_send_wrapper` so the function passed to
`multiprocessing.Pool.map` is fully pickleable.
"""
from __future__ import annotations

import base64
import copy
import datetime as _dt
import multiprocessing as mp
import time
from typing import Any, Iterable

from openai import OpenAI

from quick_vllm import cache  # type: ignore
from quick_vllm.utils import arg_dict  # type: ignore

__all__ = ["VLLMClient"]


class VLLMClient:
    """Stateful client bound to a single vLLM HTTP endpoint."""

    _DEFAULT_SAMPLING: dict[str, Any] = {
        "min_p": 0.05,
        "top_p": 0.95,
        "temperature": 0.7,
        "top_k": 40,
        "max_tokens": None,  # filled in __init__
    }

    # ------------------------------------------------------------------
    # Construction ------------------------------------------------------
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        port: int | str = 8000,
        host: str = "localhost",
        mdl: str | None = None,
        default_sampling_parameters: dict[str, Any] | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.base_url = f"http://{self.host}:{self.port}/v1/"
        self.client = OpenAI(base_url=self.base_url, api_key="vllm")

        # model id (lazy‑loaded)
        self._mdl: str | None = mdl

        # copy default sampling params & merge caller overrides
        self.default_sampling_parameters = copy.deepcopy(self._DEFAULT_SAMPLING)
        self.default_sampling_parameters["max_tokens"] = int(
            arg_dict.get("max_tokens", 4096)
        )
        if default_sampling_parameters:
            self.default_sampling_parameters.update(default_sampling_parameters)

    # ------------------------------------------------------------------
    # Static helpers ----------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def encode_image(path: str) -> str:
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode()

    @staticmethod
    def _strip_none_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [m for m in messages if m.get("content") is not None]

    # ------------------------------------------------------------------
    # Internal helpers --------------------------------------------------
    # ------------------------------------------------------------------
    def _model_id(self) -> str:
        """Return cached model id or fetch the first model from /models."""
        while self._mdl is None:
            try:
                self._mdl = self.client.models.list()[0].id
            except Exception as exc:
                print(
                    f"[{_dt.datetime.now():%Y-%m-%d %H:%M:%S}] {self.base_url}: \n"
                    f"  retrying /models after error -> {exc}"
                )
                time.sleep(1)
        return self._mdl

    # ------------------------------------------------------------------
    # Public API --------------------------------------------------------
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
        if isinstance(msg, str):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ]
        else:
            messages = msg
        messages = self._strip_none_messages(messages)

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
        """Batch variant that is *multiprocessing*‑safe."""
        if isinstance(msgs, str):
            msgs = [msgs]

        max_pool_size = min(max_pool_size or mp.cpu_count(), len(msgs))
        pool = mp.Pool(processes=max_pool_size)

        mdl_id = self._model_id()  # resolve once so every worker gets it
        common = {
            "host": self.host,
            "port": self.port,
            "mdl": mdl_id,
            "dsp": self.default_sampling_parameters,
            "kwargs": kwargs,
        }

        try:
            args = [{**common, "msg": m} for m in msgs]
            return pool.map(_worker_send_wrapper, args)
        finally:
            pool.close()
            pool.join()

    # ------------------------------------------------------------------
    def _run(
        self,
        settings: dict[str, Any],
        force_cache_miss: bool,
        silent: bool,
        just_return_text: bool,
    ) -> Any:
        cache_key = {k: v for k, v in settings.items() if k != "stream"}
        cache_path = cache.quick_hash(cache_key)

        if not (force_cache_miss or int(arg_dict.get("disable_cache", 0))):
            try:
                cached = cache.quick_load(cache_path)
                if isinstance(cached, list) and cached:
                    return [c["text"] for c in cached] if just_return_text else cached
            except Exception:
                pass  # ignore and regenerate

        retry = 5
        while True:
            try:
                completion = self.client.chat.completions.create(
                    **settings, timeout=999_999
                )
                break
            except Exception as exc:
                print(f"{__file__}: retrying after error -> {exc}")
                time.sleep(retry)
                retry = min(retry + 5, 30)

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
    def __repr__(self) -> str:  # pragma: no cover
        return f"<VLLMClient host={self.host!r} port={self.port} model={self._mdl or '(lazy)'}>"


# ----------------------------------------------------------------------
# Multiprocessing worker helper (top‑level for pickling!)
# ----------------------------------------------------------------------

def _worker_send_wrapper(d: dict[str, Any]):  # noqa: D401 – simple wrapper
    """Recreate a *throw‑away* client inside the worker process and run."""
    client = VLLMClient(
        host=d["host"],
        port=d["port"],
        mdl=d["mdl"],
        default_sampling_parameters=d["dsp"],
    )
    return client.send_message(d["msg"], **d["kwargs"])
