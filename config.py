"""
config.py
=========
Central configuration loader for the Battery Research Multi-Agent System.

Reads from the .env file (if present) via python-dotenv,
then exposes the primary settings as module-level constants.

Usage
-----
>>> from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (the directory containing this file)
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)

# ---------------------------------------------------------------------------
# Anthropic / Claude (default provider)
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
"""Anthropic API key. Required for ClaudeProvider."""

ANTHROPIC_MODEL: str = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
"""Claude model for sub-agents (fast, cost-effective)."""

ANTHROPIC_SYNTHESIS_MODEL: str = os.environ.get(
    "ANTHROPIC_SYNTHESIS_MODEL", "claude-sonnet-4-5"
)
"""Claude model for synthesis (higher quality)."""

# ---------------------------------------------------------------------------
# Google Gemini (alternative provider)
# ---------------------------------------------------------------------------

GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
"""Google AI Studio / Vertex API key. Required for GeminiProvider."""

LLM_MODEL: str = os.environ.get("LLM_MODEL", "gemini/gemini-2.5-flash")
"""Default LLM model string. Format: 'provider/model-name'."""

# Strip the provider prefix if present (google-genai SDK only wants the model name)
# e.g. "gemini/gemini-2.5-flash" -> "gemini-2.5-flash"
_model_parts = LLM_MODEL.split("/", 1)
GEMINI_MODEL: str = _model_parts[-1]
"""Bare Gemini model name for direct use with google-genai SDK."""
