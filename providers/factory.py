"""
providers/factory.py
====================
Factory function that resolves a provider name string into a live
:class:`~providers.base.LLMProvider` instance.

Supported names (case-insensitive)
-----------------------------------
``"anthropic"``  → :class:`~providers.anthropic.ClaudeProvider`  (default)
``"claude"``     → alias for ``"anthropic"``
``"gemini"``     → :class:`~providers.gemini.GeminiProvider`

Extending
---------
To add a new provider, import its class and add an entry to ``_REGISTRY``.
No other changes are required.

Usage
-----
>>> from providers.factory import get_provider
>>> provider = get_provider("anthropic")
>>> response = await provider.complete("Analyse battery tech", max_tokens=512)
"""

from __future__ import annotations

from typing import Type

from .base import LLMProvider, ProviderError
from .gemini import GeminiProvider
from .anthropic import ClaudeProvider

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Type[LLMProvider]] = {
    "anthropic": ClaudeProvider,
    "claude": ClaudeProvider,   # alias
    "gemini": GeminiProvider,
}

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_provider(name: str, **kwargs) -> LLMProvider:
    """Instantiate and return the named :class:`LLMProvider`.

    Parameters
    ----------
    name:
        Case-insensitive provider identifier, e.g. ``"gemini"``.
    **kwargs:
        Forwarded verbatim to the provider's ``__init__``.  For example,
        pass ``model="gemini-2.0-flash"`` or ``api_key="..."``
        to override defaults or skip environment-variable lookup.

    Returns
    -------
    LLMProvider
        A fully initialised concrete provider ready to call.

    Raises
    ------
    ProviderError
        If *name* is not registered or the provider fails to initialise
        (e.g. missing API key).

    Examples
    --------
    >>> p = get_provider("gemini")
    >>> p = get_provider("gemini", model="gemini-2.0-flash", api_key="sk-...")
    """
    key = name.strip().lower()
    provider_cls = _REGISTRY.get(key)

    if provider_cls is None:
        supported = ", ".join(f'"{k}"' for k in sorted(_REGISTRY))
        raise ProviderError(
            f"Unknown provider {name!r}. Supported: {supported}.",
            provider=key,
        )

    try:
        return provider_cls(**kwargs)
    except ProviderError:
        raise  # already well-formed, don't double-wrap
    except Exception as exc:
        raise ProviderError(
            f"Failed to initialise provider {name!r}: {exc}",
            provider=key,
            original=exc,
        ) from exc
