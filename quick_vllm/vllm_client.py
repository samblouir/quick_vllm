"""vllm_client.py – **FUNCTIONAL** edition
================================================
A multiprocessing-friendly, stateless wrapper around vLLM that lets you hit any
*host:port* without relying on global state or class instances (which can be
tricky to pickle on some platforms).

Key design points
-----------------
*   **No lambdas / closures** – every function is module-level → 100 % picklable.
*   `send_message()` / `send()` now take `host` & `port` kwargs so you can talk
    to many back-ends in one programme.
*   Existing helpers from `quick_vllm.api` are reused where sensible but
    extended with explicit network parameters.

Example
-------
```python
from quick_vllm.vllm_client import send_message, send

text = send_message("Hello", host="10.0.0.11", port=8000, just_return_text=1)
print(text)

batch = ["A", "B", "C"]
print(send(batch, host="10.0.0.12", port=8001, n=4, just_return_text=1))
```
"""
from __future__ import annotations

import base64
import copy
import datetime as _dt
import multiprocessing as mp
import os
import time
from typing import Any, Iterable

from openai import OpenAI

from quick_vllm import cache
from quick_vllm.utils import arg_dict

__all__ = [
    "send_message",
    "send_message_no_cache",
    "send",
    "encode_image",
    "print_chat",
]

###############################################################################
# Defaults & small helpers
###############################################################################

_DEFAULT_SAMPLING: dict[str, Any] = {
    "min_p": 0.05,
    "top_p": 0.95,
    "temperature": 0.7,
    "top_k": 40,
    "max_tokens": int(arg_dict.get("max_tokens", 4096)),
}


def _get_client(host: str, port: int | str) -> OpenAI:  # noqa: D401 – small util
    return OpenAI(base_url=f"http://{host}:{int(port)}/v1/", api_key="vllm")


###############################################################################
# Optionally re-exported helpers (from original api.py)
###############################################################################

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def print_chat(
    responses: dict[int, Any] | dict[str, Any],
    *,
    clear_thinking: bool = False,
    only_thinking: bool = False,
    only_box: bool = False,
) -> None:
    to_print: list[str] = ["*" * 60]
    for key, value in responses.items():
        if isinstance(value, dict) and "text" in value:
            value = value["text"]
        elif isinstance(value, list):
            value = "".join(value)

        if not value:
            continue

        if only_box:
            if "<box>" in value and "</box>" in value:
                value = value.rsplit("<box>", 1)[-1].rsplit("</box>", 1)[0]
            else:
                value = ""
        else:
            if only_thinking and "</think>" in value:
                value = value.split("</think>", 1)[0] + "</think>"
            if clear_thinking and "</think>" in value:
                value = value.rsplit("</think>", 1)[-1]

        to_print.extend(["\n", "-" * 40, f"[[{key}]]", value, "-" * 40, "\n"])
    to_print.append("*" * 60)
    print("\n" * 4 + "\n".join(to_print))


def _remove_invalid_messages(messages: list[dict[str, Any]]):
    return [m for m in messages if m.get("content") is not None]


###############################################################################
# Core single-message runner (adapted from original api._run_message)
###############################################################################

def _run_message(
    messages: list[dict[str, Any]],
    *,
    host: str,
    port: int | str,
    force_cache_miss: bool = False,
    stream: bool = True,
    silent: bool = True,
    **kw: Any,
):
    timeout_retry = 5
    client = kw.pop("client", None) or _get_client(host, port)
    messages = _remove_invalid_messages(messages)

    # filter body-level kwargs -------------------------------------------------
    allowed_body = {
        k: v
        for k, v in kw.items()
        if k
        in {
            "min_p",
            "top_p",
            "temperature",
            "top_k",
            "max_tokens",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "n",
        }
        and v is not None
    }

    sampling = {**_DEFAULT_SAMPLING, **allowed_body}
    settings = {
        "model": kw.get("model"),  # resolved just-in-time below
        "messages": messages,
        "stream": stream,
        "extra_body": sampling,
    }

    # model resolution (cached per-process) -----------------------------------
    if settings["model"] is None:
        while True:
            try:
                settings["model"] = client.models.list()[0].id
                break
            except Exception as exc:
                print(f"retrying /models after error -> {exc}")
                time.sleep(1)

    cache_key = {k: v for k, v in settings.items() if k != "stream"}
    cache_path = cache.quick_hash(cache_key)

    if not (force_cache_miss or int(arg_dict.get("disable_cache", 0))):
        try:
            cached = cache.quick_load(cache_path)
            if isinstance(cached, list) and cached:
                return cached
        except Exception:
            pass  # ignore cache issues

    # actual network call ------------------------------------------------------
    while True:
        try:
            completion = client.chat.completions.create(**settings, timeout=999_999)
            break
        except Exception as exc:
            print(f"{__name__}: retrying after error -> {exc}")
            time.sleep(timeout_retry)
            timeout_retry = min(timeout_retry + 5, 30)

    if settings["stream"]:
        chunks: dict[int, list[str]] = {}
        for chunk in completion:
            choice = chunk.choices[0]
            if choice.delta.content is not None:
                chunks.setdefault(choice.index, []).append(choice.delta.content)
        texts = ["".join(v) for _, v in sorted(chunks.items())]
    else:
        texts = [c.message.content or "" for c in completion.choices]

    packaged = [{"text": t, "settings": settings} for t in texts]
    if not int(arg_dict.get("disable_cache", 0)):
        cache.quick_save(packaged, cache_path)

    return packaged


###############################################################################
# Public helpers
###############################################################################

def send_message(
    msg: str | list[dict[str, Any]],
    *,
    host: str = "localhost",
    port: int | str = 8000,
    system_prompt: str | None = None,
    just_return_text: bool = False,
    force_cache_miss: bool = False,
    **kw: Any,
):
    """Synchronous helper mirroring the old OO method but *stateless*."""
    if isinstance(msg, str):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": msg},
        ]
    else:
        messages = msg

    out = _run_message(
        messages,
        host=host,
        port=port,
        force_cache_miss=force_cache_miss,
        **kw,
    )
    return [e["text"] for e in out] if just_return_text else out


def send_message_no_cache(msg, **kw):
    return send_message(msg, force_cache_miss=True, **kw)


# --------------------------------------------------------------------
# Multiprocessing batch helper
# --------------------------------------------------------------------

def _worker(d: dict[str, Any]):
    """Top-level so it’s pickleable."""
    return send_message(**d)


def send(
    msgs: str | Iterable[str],
    *,
    host: str = "localhost",
    port: int | str = 8000,
    max_pool_size: int | None = None,
    just_return_text: bool = False,
    **kwargs: Any,
):
    """Batch interface mirroring previous helper but fully functional."""
    if isinstance(msgs, str):
        msgs = [msgs]

    max_pool_size = min(max_pool_size or mp.cpu_count(), len(msgs))
    pool = mp.Pool(processes=max_pool_size)

    try:
        args = [
            {
                "msg": m,
                "host": host,
                "port": port,
                "just_return_text": just_return_text,
                **kwargs,
            }
            for m in msgs
        ]
        return pool.map(_worker, args)
    finally:
        pool.close()
        pool.join()
