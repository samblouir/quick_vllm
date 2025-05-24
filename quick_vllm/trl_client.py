"""vllm_client.py – OO version (pickle-safe, HF tokenizer)

* fixes the multiprocessing pickle error by keeping helpers at module scope
* always decodes token IDs with a Hugging Face tokenizer (if you don’t pass
  one, the class instantiates one from `mdl`)
"""

from __future__ import annotations


import copy
import datetime as _dt
import multiprocessing as mp
import time
from typing import Any, Dict, Iterable, List, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    class OpenAI:  # type: ignore
        pass
from quick_vllm import cache  # type: ignore
from quick_vllm.utils import arg_dict  # type: ignore



# ---------------------------------------------------------------------
# Tokenizer (Hugging Face required)
# ---------------------------------------------------------------------
try:
    from transformers import AutoTokenizer
except ImportError as exc:  # noqa: N818 – sentinel
    raise ImportError(
        "Hugging Face `transformers` is required – "
        "pip install transformers>=4.40"
    ) from exc

__all__ = ["VLLMClient"]

###############################################################################
# Helper functions (module-level => pickleable)
###############################################################################

_DEFAULT_SAMPLING: Dict[str, Any] = {
    "min_p": 0.05,
    "top_p": 0.95,
    "temperature": 0.7,
    "top_k": 40,
    "max_tokens": int(arg_dict.get("max_tokens", 4096)),
}


def _encode_image(path: str) -> str:
    """Base64-encode a local image (used for vision models)."""
    import base64

    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def _strip_none_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove chat messages whose `content` is None (OpenAI SDK quirk)."""
    return [m for m in msgs if m.get("content") is not None]


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Collapse OpenAI chat history into a single prompt string."""
    system_ctx = " ".join(m["content"] for m in messages if m["role"] == "system")
    last_user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    return f"{system_ctx}\n{last_user}" if system_ctx else last_user


def _chat_to_generate_settings(chat_settings: Dict[str, Any]) -> Dict[str, Any]:
    """Translate kwargs meant for `chat.completions.create()` into
    the JSON body FastAPI’s `GenerateRequest` wants."""
    messages = chat_settings["messages"]
    extra = chat_settings.get("extra_body", {})
    return {
        "prompts": [_messages_to_prompt(messages)],
        "n": extra.get("n", 1),
        "temperature": extra.get("temperature", 0.7),
        "top_p": extra.get("top_p", 0.95),
        "top_k": extra.get("top_k", 40),
        "min_p": extra.get("min_p", 0.05),
        "max_tokens": extra.get("max_tokens", 4096),
        "repetition_penalty": extra.get("repetition_penalty", 1.0),
        "guided_decoding_regex": chat_settings.get("guided_decoding_regex"),
    }


###############################################################################
# Worker function (must be at top level for multiprocessing)
###############################################################################

def _worker_send_wrapper(d: Dict[str, Any]):  # noqa: D401 – simple wrapper
    """Recreate a `VLLMClient` inside the worker and forward the call."""
    tok = AutoTokenizer.from_pretrained(d["mdl"]) if d["mdl"] else None
    client = VLLMClient(
        host=d.get("host", "localhost"),
        port=d.get("port", 8000),
        mdl=d["mdl"],
        default_sampling_parameters=d.get("dsp", {}),
        tokenizer=tok,
    )
    return client.send_message(d["msg"], **d["kwargs"])


###############################################################################
# Main class
###############################################################################

class VLLMClient:
    """Stateful wrapper around one vLLM HTTP endpoint (multiprocessing-safe)."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int | str = 8000,
        mdl: str | None = None,
        default_sampling_parameters: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.base_url = f"http://{self.host}:{self.port}/"
        self.client = OpenAI(base_url=self.base_url, api_key="vllm")

        self._mdl: str | None = mdl  # resolved lazily

        # ---------------- sampling defaults ----------------
        self.default_sampling_parameters = copy.deepcopy(_DEFAULT_SAMPLING)
        if default_sampling_parameters:
            self.default_sampling_parameters.update(default_sampling_parameters)

        # ---------------- tokenizer ----------------
        if tokenizer is not None:
            self.tokenizer = tokenizer
        # elif mdl is not None:
        #     self.tokenizer = AutoTokenizer.from_pretrained(mdl)
        else:
            raise ValueError(
                "You must either pass `mdl` so the client can load an "
                "AutoTokenizer, or supply an explicit `tokenizer=` instance."
            )

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------
    encode_image = staticmethod(_encode_image)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _model_id(self) -> str | None:
        return None  # override if you need dynamic /models probing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def send_message(
        self,
        msg: str | List[Dict[str, Any]],
        *,
        system_prompt: str | None = None,
        stream: bool = True,
        force_cache_miss: bool = False,
        just_return_text: bool = False,
        silent: bool = True,
        **kw: Any,
    ) -> Any:
        """Send one prompt (or chat history) and get a completion."""
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
        **kwargs: Any,
    ) -> List[Any]:
        """Multiprocessing batch helper (now pickle-safe)."""
        if isinstance(msgs, str):
            msgs = [msgs]

        max_pool_size = min(max_pool_size or mp.cpu_count(), len(msgs))
        pool = mp.Pool(processes=max_pool_size)

        common = {
            "host": self.host,
            "port": self.port,
            "mdl": self._mdl,
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
    # Core request logic
    # ------------------------------------------------------------------
    def _run(
        self,
        settings: Dict[str, Any],
        force_cache_miss: bool,
        silent: bool,
        just_return_text: bool,
    ) -> Any:
        cache_key = {k: v for k, v in settings.items() if k != "stream"}
        cache_path = cache.quick_hash(cache_key)

        if not (force_cache_miss or int(arg_dict.get("disable_cache", 1))):
            try:
                cached = cache.quick_load(cache_path)
                if isinstance(cached, list) and cached:
                    return [c["text"] for c in cached] if just_return_text else cached
            except Exception:
                pass

        retry = 5
        while True:
            try:
                body = _chat_to_generate_settings(settings)
                completion = self.client.post("/generate/", body=body, cast_to=Dict[str, Any])
                break
            except Exception as exc:
                print(f"{__file__}: retrying after error -> {exc}")
                import traceback

                traceback.print_exc()
                print("*" * 60)
                time.sleep(retry)
                retry = min(retry + 5, 30)

        completion_ids = completion["completion_ids"]

        # if settings["stream"]:
        #     chunks: Dict[int, List[str]] = {}
        #     for chunk in completion:
        #         choice = chunk.choices[0]
        #         if choice.delta.content is not None:
        #             chunks.setdefault(choice.index, []).append(choice.delta.content)
        #     texts = ["".join(v) for _, v in sorted(chunks.items())]
        # else:

        texts = self.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        texts = [x.strip() for x in texts]

        def cleaner(x):
            trd = {
                "\\n": "\n",
                "\\t": "\t",
                "\\r": "\r",
                "\\'": "'",
                '\\"': '"',
                "\\\\": "\\",
            }
            for k, v in trd.items():
                while k in x:
                    x = x.replace(k, v)
            return x
        texts = [cleaner(x) for x in texts]



        packaged = [
            {
                "text": t,
                "settings": {k: v for k, v in settings.items() if k != "stream"},
            }
            for t in texts
        ]

        if not int(arg_dict.get("disable_cache", 1)):
            cache.quick_save(packaged, cache_path)

        return texts if just_return_text else packaged

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # noqa: D401 – simple repr
        return (
            f"<VLLMClient host={self.host!r} port={self.port} "
            f"model={self._mdl or '(lazy)'}>"
        )

###############################################################################
# Self-test section
###############################################################################

if __name__ == "__main__":
    import multiprocessing as mp

    from transformers import AutoTokenizer  # ensure HF is available
    real_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    # ---------- 1. pickle / multiprocessing sanity check ----------
    class DummyVLLMClient(VLLMClient):
        def send_message(self, msg, **kw):
            return f"Echo: {msg}"

    def _dummy_worker_send_wrapper(d):
        client = DummyVLLMClient(
            host=d["host"],
            port=d["port"],
            mdl=d["mdl"],
            default_sampling_parameters=d["dsp"],
            tokenizer=real_tok,
        )
        return client.send_message(d["msg"], **d["kwargs"])

    DummyVLLMClient._worker_send_wrapper = staticmethod(_dummy_worker_send_wrapper)

    def send_dummy(self, msgs, *, max_pool_size=None, **kwargs):
        if isinstance(msgs, str):
            msgs = [msgs]
        max_pool_size = min(max_pool_size or mp.cpu_count(), len(msgs))
        pool = mp.Pool(processes=max_pool_size)
        common = {
            "host": self.host,
            "port": self.port,
            "mdl": self._mdl,
            "dsp": self.default_sampling_parameters,
            "kwargs": kwargs,
        }
        try:
            args = [{**common, "msg": m} for m in msgs]
            return pool.map(_dummy_worker_send_wrapper, args)
        finally:
            pool.close()
            pool.join()

    DummyVLLMClient.send = send_dummy


    print("Testing multiprocessing send with DummyVLLMClient …")
    test_client = DummyVLLMClient(mdl="meta-llama/Llama-3.2-1B-Instruct", tokenizer=real_tok, port=8000)
    echoes = test_client.send(["hello", "world"])
    print("Echoes:", echoes)

    # ---------- 2. Round-trip test against a real vLLM server ----------
    try:
        
        client = VLLMClient(
            port=8000,
            mdl="meta-llama/Llama-3.2-1B-Instruct",
            tokenizer=real_tok,
            default_sampling_parameters={"max_tokens": 32},
        )

        test_messages = [
            "Hello, world!",
            # "Tell me a joke.",
            # "What's the capital of France?",
            # "Summarize the theory of relativity in one sentence.",
        ]

        print("\n== Single-threaded test (send_message) ==")
        for msg in test_messages:
            try:
                response = client.send_message(msg, just_return_text=True, stream=False)
                # response = real_tok.decode(response, skip_special_tokens=True)
                print(f"[SYNC] {msg!r} → {response}")
            except Exception as e:
                print(f"Error: {e}")

        print("\n== Multiprocessing test (send) ==")
        results = client.send(test_messages, just_return_text=True, stream=False)
        for prompt, resp in zip(test_messages, results):
            # resp = real_tok.decode(resp, skip_special_tokens=True)
            print(f"yeah [ASYNC] {prompt!r} → {resp}")

    except Exception as e:
        print("\nReal vLLM test skipped (no server listening on :8000).")
        print("Error was:", e)
