"""
agents/__init__.py
==================
Battery Research Multi-Agent System — Agents package.
"""
from agents.base_agent import AgentConfig, ResearchAgent
from agents.synthesizer import ResearchSynthesizer
from agents.factory import build_all, build_agent_for_role, build_synthesis_agent

__all__ = [
    "AgentConfig",
    "ResearchAgent",
    "ResearchSynthesizer",
    "build_all",
    "build_agent_for_role",
    "build_synthesis_agent",
]
