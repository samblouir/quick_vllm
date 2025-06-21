"""Fallback stub for the :mod:`openai` package used in tests.

When the real ``openai`` library is available it will be imported and all of
its symbols re-exported.  If the import fails (e.g. in the network isolated
test environment) a very small stub is provided instead so that the rest of the
package can be imported without errors.
"""

from __future__ import annotations

import importlib.util
import os
import sys

__all__ = ["OpenAI"]

_DEBUG = os.getenv("QUICK_VLLM_DEBUG") == "1"


def _load_real_openai():
    """Attempt to import the real ``openai`` package.

    This module itself is named ``openai`` so we temporarily remove our own
    directory from ``sys.path`` to avoid recursive imports.  If the real package
    is found, its public attributes are injected into this module's namespace
    and the function returns ``True``.  Otherwise ``False`` is returned and the
    lightweight stub defined below will be used instead.
    """

    spec = importlib.util.find_spec("openai")
    if spec is None or os.path.abspath(spec.origin) == os.path.abspath(__file__):
        return False

    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    globals().update(module.__dict__)
    return True


if _DEBUG or not _load_real_openai():
    class _DummyDelta:
        def __init__(self, content="Hello World"):
            self.content = content

    class _DummyMessage:
        def __init__(self, content="Hello World"):
            self.content = content

    class _DummyChoice:
        def __init__(self, content="Hello World"):
            self.index = 0
            self.delta = _DummyDelta(content)
            self.message = _DummyMessage(content)
            self.finish_reason = "stop"
            self.stop_reason = "stop"

    class _DummyChunk:
        def __init__(self, content="Hello World"):
            self.choices = [_DummyChoice(content)]

    class _DummyCompletion:
        def __init__(self, stream: bool = True, content: str = "Hello World"):
            self._stream = stream
            self._content = content
            self._yielded = False
            self.choices = [_DummyChoice(content)]

        def __iter__(self):
            self._yielded = False
            return self

        def __next__(self):
            if not self._stream or self._yielded:
                raise StopIteration
            self._yielded = True
            return _DummyChunk(self._content)

    class OpenAI:  # pragma: no cover - used only when real package unavailable
        def __init__(self, *args, **kwargs):
            self.chat = type(
                "chat",
                (),
                {
                    "completions": type(
                        "completions",
                        (),
                        {
                            "create": staticmethod(
                                lambda *a, **kw: _DummyCompletion(
                                    kw.get("stream", True)
                                )
                            )
                        },
                    )(),
                },
            )()

        class _Models:
            @staticmethod
            def list():
                return [type("model", (), {"id": "debug-model"})()]

        def models(self):
            return self._Models()
