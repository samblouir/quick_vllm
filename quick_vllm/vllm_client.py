"""vllm_client.py – OO version (pickle‑safe)
================================================
Restores the original **`VLLMClient`** class but fixes the multiprocessing
pickle error by eliminating lambdas/closures inside `send()`.  A *top‑level*
helper `_worker_send_wrapper` is now used, which is fully pickleable by
`multiprocessing`.

Existing code that does `from quick_vllm import VLLMClient` will work without
changes.
"""
from __future__ import annotations

import base64
import copy
import datetime as _dt
import multiprocessing as mp
import time
from typing import Any, Iterable
import traceback
from openai import OpenAI

from quick_vllm import cache  # type: ignore
from quick_vllm.utils import arg_dict  # type: ignore
from quick_vllm.api import _AsyncSendResult

__all__ = ["VLLMClient"]

###############################################################################
# Helper functions (module‑level => pickleable)
###############################################################################

_DEFAULT_SAMPLING = {
    "min_p": 0.05,
    "top_p": 0.95,
    "temperature": 0.7,
    "top_k": 40,
    "max_tokens": int(arg_dict.get("max_tokens", 4096)),
}


def _encode_image(path: str) -> str:  # exported via class method
    with open(path, "rb") as fh:
        import base64

        return base64.b64encode(fh.read()).decode()


def _strip_none_messages(msgs: list[dict[str, Any]]):
    return [m for m in msgs if m.get("content") is not None]


###############################################################################
# Worker function (needs to be at top level for multiprocessing)
###############################################################################

def _worker_send_wrapper(d: dict[str, Any]):  # noqa: D401 – simple wrapper
    """Recreate a `VLLMClient` inside the worker and forward the call."""
    client = VLLMClient(
        host=d["host"],
        port=d["port"],
        mdl=d["mdl"],
        default_sampling_parameters=d["dsp"],
    )
    return client.send_message(d["msg"], **d["kwargs"])


###############################################################################
# Main class
###############################################################################

class VLLMClient:
    """Stateful wrapper around one vLLM HTTP endpoint (multiprocessing‑safe)."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int | str = 8000,
        mdl: str | None = None,
        default_sampling_parameters: dict[str, Any] | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.base_url = f"http://{self.host}:{self.port}/v1/"
        self.client = OpenAI(base_url=self.base_url, api_key="vllm")

        self._mdl: str | None = mdl  # resolved lazily

        self.default_sampling_parameters = copy.deepcopy(_DEFAULT_SAMPLING)
        if default_sampling_parameters:
            self.default_sampling_parameters.update(default_sampling_parameters)

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------
    encode_image = staticmethod(_encode_image)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _model_id(self) -> str:
        while self._mdl is None:
            counter = 0
            try:
                self._mdl = self.client.models.list()[0].id
            except Exception as exc:
                
                try:
                    self._mdl = self.client.models.list().data[0].id
                    break
                except Exception as exc2:
                    traceback.print_exc()
                    print(
                        f"[{_dt.datetime.now():%Y-%m-%d %H:%M:%S}] {self.base_url}: \n"
                        f"  retrying /models after error -> {exc}"
                    )
                    traceback.print_exc()
                    time.sleep(1)
                    
        return self._mdl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def send_message(
        self,
        msg: str | list[dict[str, Any]],
        *,
        system_prompt: str | None = None,
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
        messages = _strip_none_messages(messages)

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
        async_: bool = False,
        **kwargs: Any,
    ) -> list[Any] | Any:
        """Multiprocessing batch helper (now pickle‑safe)."""
        if isinstance(msgs, str):
            msgs = [msgs]

        max_pool_size = min(max_pool_size or mp.cpu_count(), len(msgs))
        pool = mp.Pool(processes=max_pool_size)

        # shared data passed to every worker (must be pickleable)
        common = {
            "host": self.host,
            "port": self.port,
            "mdl": self._model_id(),
            "dsp": self.default_sampling_parameters,
            "kwargs": kwargs,
        }

        args = [{**common, "msg": m} for m in msgs]

        if async_:
            async_results = [
                pool.apply_async(_worker_send_wrapper, (a,)) for a in args
            ]
            pool.close()
            return _AsyncSendResult(async_results, pool)

        try:
            return pool.map(_worker_send_wrapper, args)
        finally:
            pool.close()
            pool.join()

    def send_async(
        self,
        msgs: str | Iterable[str],
        *,
        max_pool_size: int | None = None,
        **kwargs: Any,
    ) -> _AsyncSendResult:
        """Convenience wrapper for :meth:`send` with ``async_=True``."""
        return self.send(
            msgs,
            max_pool_size=max_pool_size,
            async_=True,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Core request logic
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
                pass

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
            texts = ["".join(v) for _, v in sorted(chunks.items())]
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
    def __repr__(self):  # noqa: D401 – simple repr
        return (
            f"<VLLMClient host={self.host!r} port={self.port} "
            f"model={self._mdl or '(lazy)'}>"
        )
