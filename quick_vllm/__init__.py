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

from .api import (
    send,
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
    "send_message",
    "send_message_no_cache",
    "encode_image",
    "print_chat",
    "remove_invalid_messages",
    # OO API
    "VLLMClient",
]
