## File: quick_vllm/__init__.py
"""
Package initialisation for *quick_vllm*.

Re-exports:
-----------

Functional helpers
    send, send_message, send_message_no_cache, encode_image, print_chat,
    remove_invalid_messages   – from :pymod:`quick_vllm.api`

Object-oriented client
    VLLMClient                – from :pymod:`quick_vllm.vllm_client`
"""
import sys

# ---------------------------------------------------------------------------
# macOS uses the 'spawn' start method by default which breaks pooling in this
# package. Switch to 'fork' if running on a Mac.
if sys.platform == "darwin":
    import multiprocessing as mp
    try:  # set_start_method() can only be called once per session
        mp.set_start_method("fork")
    except RuntimeError:
        pass


from .api import (
    send,
    send_async,
    send_message,
    send_message_no_cache,
    encode_image,
    print_chat,
    remove_invalid_messages,
)

from .vllm_client import VLLMClient

__all__ = [
    # functional API
    "send",
    "send_async",
    "send_message",
    "send_message_no_cache",
    "encode_image",
    "print_chat",
    "remove_invalid_messages",
    # OO API
    "VLLMClient",
]
