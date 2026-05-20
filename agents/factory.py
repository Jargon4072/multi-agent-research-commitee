"""
agents/factory.py
=================
Factory functions for building research sub-agents.

Public surface
--------------
build_all(provider)          → dict[str, ResearchAgent]  (all 5 sub-agents)
build_synthesis_agent(provider) → ResearchSynthesizer
"""

from __future__ import annotations

from typing import Dict

from providers.base import LLMProvider
from agents.base_agent import AgentConfig, ResearchAgent
from agents.synthesizer import ResearchSynthesizer
from agents.prompts import (
    CHEMISTRY_SYSTEM_PROMPT,
    POLICY_SYSTEM_PROMPT,
    GEOGRAPHY_SYSTEM_PROMPT,
    RETRIEVAL_SYSTEM_PROMPT,
    COMPARISON_SYSTEM_PROMPT,
)
from models.schemas import SubtaskType

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

_AGENT_DEFINITIONS = [
    AgentConfig(
        id="chemistry",
        name="Dr. Chen",
        role=SubtaskType.CHEMISTRY,
        system_prompt=CHEMISTRY_SYSTEM_PROMPT,
        display_name="Dr. Chen · Chemistry Expert",
    ),
    AgentConfig(
        id="policy",
        name="Dr. Amara",
        role=SubtaskType.POLICY,
        system_prompt=POLICY_SYSTEM_PROMPT,
        display_name="Dr. Amara · Policy Expert",
    ),
    AgentConfig(
        id="geography",
        name="Dr. Okonkwo",
        role=SubtaskType.GEOGRAPHY,
        system_prompt=GEOGRAPHY_SYSTEM_PROMPT,
        display_name="Dr. Okonkwo · Geography & Supply Chain",
    ),
    AgentConfig(
        id="retrieval",
        name="Alex",
        role=SubtaskType.RETRIEVAL,
        system_prompt=RETRIEVAL_SYSTEM_PROMPT,
        display_name="Alex · Cross-Domain Retrieval",
    ),
    AgentConfig(
        id="comparison",
        name="Dr. Rivera",
        role=SubtaskType.COMPARISON,
        system_prompt=COMPARISON_SYSTEM_PROMPT,
        display_name="Dr. Rivera · Structured Comparison",
    ),
]

# Fast lookup by role value
_AGENT_BY_ROLE: Dict[str, AgentConfig] = {
    cfg.role.value: cfg for cfg in _AGENT_DEFINITIONS
}


def build_all(provider: LLMProvider) -> Dict[str, ResearchAgent]:
    """Build all research sub-agents keyed by agent ID.

    Returns
    -------
    dict mapping agent_id → ResearchAgent
    """
    return {cfg.id: ResearchAgent(cfg, provider) for cfg in _AGENT_DEFINITIONS}


def build_agent_for_role(role: SubtaskType, provider: LLMProvider) -> ResearchAgent:
    """Build a single research agent for the given role.

    Parameters
    ----------
    role:
        The SubtaskType to build an agent for.
    provider:
        Live LLM provider.

    Returns
    -------
    ResearchAgent

    Raises
    ------
    ValueError
        If no agent is registered for the given role.
    """
    cfg = _AGENT_BY_ROLE.get(role.value)
    if cfg is None:
        raise ValueError(f"No agent registered for role {role.value!r}")
    return ResearchAgent(cfg, provider)


def build_synthesis_agent(provider: LLMProvider) -> ResearchSynthesizer:
    """Build the synthesis agent that merges all findings."""
    return ResearchSynthesizer(provider)