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


if not _load_real_openai():
    class OpenAI:  # pragma: no cover - used only when real package unavailable
        def __init__(self, *args, **kwargs):
            pass

        class chat:
            class completions:
                @staticmethod
                def create(*args, **kwargs):
                    raise RuntimeError("OpenAI stub not functional")

        def models(self):
            return type("models", (), {"list": lambda self: []})()
