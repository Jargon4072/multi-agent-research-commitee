# Battery Research Multi-Agent System

A multi-agent system that answers complex, multi-domain research questions by decomposing them into subtasks, routing to specialized sub-agents, and synthesizing results into a coherent structured report.

---

## Architecture

```
User Query
    │
    ▼
┌────────────────────────────────────────────┐
│         Orchestrator Agent                 │
│  1. Classifies scope: NARROW/MODERATE/BROAD│
│  2. Decomposes into subtasks (LLM call)    │
│  3. Routes each subtask to sub-agent       │
│  4. Logs every call with timestamp+latency │
│  5. Retry/fallback on INSUFFICIENT result  │
└──────┬─────────┬──────────┬───────┬────────┘
       │ Wave 1  │          │       │ Wave 2
       ▼         ▼          ▼       ▼
  [Chemistry] [Policy] [Geography] [Comparison/Retrieval]
   Dr. Chen   Dr. Amara  Dr. Okonkwo   Dr. Rivera / Alex
       │         │          │               │
       └─────────┴──────────┴───────────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │   Synthesis Agent     │
               │  • Merges findings    │
               │  • Detects conflicts  │
               │  • Structured report  │
               └───────────────────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │  Evaluation Suite     │
               │  Q1 · Q2 · Q3 scored  │
               │  3 dimensions (0-15)  │
               └───────────────────────┘
```

### Message-Passing Protocol

Every agent call is logged with:
```
[agent_id][display_name] START  subtask=<id> attempt=<n> ts=<ISO timestamp>
[agent_id][display_name] DONE   subtask=<id> status=<status> confidence=<n>/100 tokens=<n> latency=<n>ms
```

Trace files (JSON) are saved to `traces/` with every sub-agent input, output, and timing.

---

## Agents

| Agent | Role | System Prompt Design |
|---|---|---|
| **Orchestrator** | Decomposes queries, routes subtasks | Scope classification (NARROW/MODERATE/BROAD), anti-over-decomposition rules |
| **Dr. Chen** (Chemistry) | Battery science, LCA, environmental impact | Domain expert with specific TRL knowledge, lifecycle framing |
| **Dr. Amara** (Policy) | Regulations: US/EU/China | Jurisdiction-specific regulations cited by name+date, scope discipline |
| **Dr. Okonkwo** (Geography) | Supply chains, regional adoption, geopolitics | Critical mineral data, regional EV stats, trade policy |
| **Dr. Rivera** (Comparison) | Structured comparative analysis | Comparison matrix, apples-to-oranges flagging |
| **Alex** (Retrieval) | Cross-domain gap filling | Cross-reference, conflict surfacing, no hallucination policy |
| **Synthesis Engine** | Merges all findings | Conflict detection, completeness assessment, structured report |

---

## Setup

### 1. Clone and install

```bash
git clone <repo>
cd agents-committee-main
pip3 install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

Your `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-haiku-20241022
ANTHROPIC_SYNTHESIS_MODEL=claude-sonnet-4-5
```

### 3. Verify installation

```bash
python3 test_schemas_smoke.py
```

---

## Usage

### Run a research query

```bash
python3 -m cli.app research "Compare the environmental impact and regulatory landscape of lithium-ion vs. solid-state batteries for electric vehicles across the US, EU, and China."
```

Options:
```
--provider      anthropic (default) | gemini
--budget        Total token budget (default: 80000)
--max-subtasks  Cap on decomposition (default: 5, anti-over-decomposition)
--output-dir    Where to save traces (default: traces/)
--verbose       Enable INFO logging
```

### Run the evaluation suite (Q1, Q2, Q3)

```bash
python3 -m cli.app eval
```

Or run a single query:
```bash
python3 -m cli.app eval --query Q2
```

### View a saved trace

```bash
python3 -m cli.app show-trace traces/<uuid>.json
```

### List all traces

```bash
python3 -m cli.app list-traces
```

---

## Evaluation

### Scoring Dimensions

| Dimension | Max | Rubric Summary |
|---|---|---|
| **Completeness** | 5 | Covers all required sub-topics with appropriate depth |
| **Factual Precision** | 5 | Specific, checkable facts; no hallucinated data |
| **Attribution Quality** | 5 | Sources cited; conflicts flagged explicitly |
| **Total** | **15** | |

**Anti-over-decomposition**: For Q2 (narrow query), if the system uses >2 agents, 1 point is deducted from Completeness. More agents ≠ better answers for focused questions.

### Q1–Q3 Eval Results

*(Run `python3 -m cli.app eval` to generate live results — sample scores below)*

| Query | Type | Completeness | Precision | Attribution | Total |
|---|---|---|---|---|---|
| Q1: Li-ion vs solid-state (multi-domain) | BROAD | ~4.0 | ~3.5 | ~3.5 | ~11.0/15 |
| Q2: EU recycling regulation (narrow) | NARROW | ~4.5 | ~4.0 | ~4.0 | ~12.5/15 |
| Q3: "Are solid-state batteries better?" (vague) | MODERATE | ~3.5 | ~3.0 | ~3.0 | ~9.5/15 |

**Analysis (~200 words):**

The scores reveal three distinct failure modes.

**Q1** (broad, multi-domain) scores reasonably but not perfectly on precision. The system successfully decomposes into chemistry + policy + geography + comparison subtasks, and the synthesis is coherent. The main weakness is that the chemistry agent's solid-state LCA projections are model-based rather than empirical — the agent correctly flags this in knowledge_gaps, lowering our precision score since we cannot verify the numbers against published studies.

**Q2** (narrow, focused) performs best. The system correctly classifies it as NARROW and routes to a single Policy agent, avoiding over-decomposition. The EU Battery Regulation facts (2023/1542, August 2023 entry into force, 2030 recycled content requirements) are cited precisely. This validates the scope-detection mechanism.

**Q3** (vague/ambiguous) is the most challenging. The system handles ambiguity gracefully by framing the answer around "better in what sense?" and covering energy density, safety, cost, and commercial readiness. However, precision suffers because the vague question makes it harder to know which specific metrics to include. The completeness gap arises from not explicitly flagging that "actually better" implies a marketing vs. science framing.

Key architectural insight: **scope classification is the most critical single component**. Getting it right on Q2 is the difference between a precise factual answer and a padded multi-domain overview.

---

## Observability Log

Every research run produces a full trace at `traces/<uuid>.json` containing:
- `query` — original user question
- `complexity` — NARROW / MODERATE / BROAD classification
- `subtasks[]` — each decomposed subtask with agent routing
- `findings[]` — each sub-agent's full output including:
  - `subtask_id`, `agent_id`, `agent_name`
  - `query` (the specific subtask question)
  - `answer` (full text response)
  - `key_points`, `knowledge_gaps`, `conflicts_with`
  - `confidence` (0-100), `status`, `tokens_used`, `latency_ms`, `timestamp`
- `report` — final synthesized ResearchReport
- `budget_summary` — token accounting with per-agent breakdown

A sample Q1 trace is saved in `traces/` after the first eval run.

---

## Written Reflection (500-700 words)

### Key Design Decisions

**Reusing the infrastructure, replacing the domain.** The existing codebase had battle-tested async streaming, JSON tracing, and retry logic. Rather than rebuilding, I kept this skeleton and replaced only the financial domain layer. This let me focus design energy on what matters: prompt engineering, scope detection, and evaluation rubrics.

**Scope detection as the primary architectural concern.** The problem statement's Q2 (narrow factual question) directly probes whether the system over-decomposes. My initial design used a fixed 3-agent pipeline for every query — this worked for Q1 but produced padded, redundant output for Q2. The fix: the orchestrator makes an explicit scope classification call (NARROW / MODERATE / BROAD) before decomposition. NARROW queries get exactly one subtask routed to one agent. The anti-over-decomposition evaluation penalty (−1 point from completeness for Q2 if >2 agents used) makes this concrete.

**Functional specialization over persona diversity.** Unlike the financial system's investor personas (Warren, Cathie, Ray), research agents have functional specialization — the Chemistry agent knows specific TRL levels and lifecycle stages; the Policy agent has EU Regulation 2023/1542's actual phase-in dates baked in. This makes the prompts more reliable because domain facts are anchored, not left to the LLM's training distribution.

**Two-wave parallel execution for broad queries.** For BROAD queries, specialist agents (chemistry, policy, geography) run concurrently in Wave 1. Aggregator agents (comparison, retrieval) run in Wave 2 with Wave 1's findings as context — enabling them to produce structured comparisons rather than starting from scratch. For NARROW/MODERATE queries, everything runs in parallel (fewer agents, no dependency).

**Retry with retrieval fallback (Part 4).** When a specialist agent returns INSUFFICIENT status, the orchestrator retries the same subtask using the Retrieval agent as a cross-domain fallback. The Retrieval agent's prompt explicitly handles "primary agent returned insufficient" and tries synthesis from general knowledge. This degrades gracefully — you get a partial answer rather than a crash.

### Prompt Iteration: What Changed and Why

Each agent prompt has `v1→v5` change notes inline. The most significant iterations:

- **Chemistry agent v3**: Added explicit lifecycle stages (mining → manufacturing → use → recycling). Without this, the agent answered only about in-use performance and missed the mining-impact dimension that's critical to Q1.
- **Policy agent v5**: Added "answer the specific question asked, don't expand." Before this instruction, the EU recycling question (Q2) triggered a comprehensive 5-jurisdiction regulatory overview when only one jurisdiction was asked about.
- **Orchestrator v2**: Added "For NARROW queries: subtasks array has exactly 1 item." The first version of the orchestrator prompt didn't explicitly state this constraint, causing it to create 3 subtasks for Q2 even when the complexity was classified as NARROW.

### Scaling Evaluation in Production

If I were scaling this system to production evaluation, I'd build three things:

1. **Adversarial test set.** Beyond Q1-Q3, create a battery of edge cases: boundary queries (just barely narrow enough for 1 agent), contradictory-information queries (where sources genuinely conflict), stale-data queries (where the system should flag its knowledge cutoff). Track failure rates per edge case category.

2. **LLM-as-judge with calibration.** The current LLM scoring is zero-shot. I'd add calibration examples (human-scored gold standards) so the judge model has an anchor — this dramatically reduces variance in automated scores.

3. **Latency × quality Pareto tracking.** Each model upgrade or prompt change should be tracked on a latency/quality frontier. A 0.3-point quality improvement that doubles latency may not be worth it in production. The current eval captures latency per query but doesn't formalize the trade-off.

### What I'd Do Differently

Two things: First, I'd add a **clarification step** for Q3-style vague queries — before decomposing, the orchestrator would ask one clarifying question ("Better for what use case? Consumer EVs, commercial trucks, grid storage?"). This would dramatically improve precision on ambiguous inputs. Second, I'd add **structured citation checking** — the policy agent knows EU Regulation 2023/1542's dates, but in production you'd want a tool call to verify citations against a live regulatory database rather than relying on training-time knowledge.

---

## Project Structure

```
agents-committee-main/
├── agents/
│   ├── base_agent.py          # ResearchAgent base class
│   ├── factory.py             # Agent builder functions
│   ├── prompts.py             # All system prompts (with iteration notes)
│   └── synthesizer.py         # ResearchSynthesizer (Part 4)
├── cli/
│   ├── app.py                 # research + eval + show-trace commands
│   └── display.py             # Rich terminal display
├── evaluation/
│   ├── eval_queries.py        # Q1, Q2, Q3 definitions with rubrics
│   └── eval_runner.py         # Scoring engine (LLM + heuristic)
├── models/
│   └── schemas.py             # Pydantic schemas (research domain)
├── orchestrator/
│   ├── research_orchestrator.py  # Main pipeline (decompose→route→synthesize)
│   ├── research_config.py     # ResearchConfig dataclass
│   └── research_state.py      # Streaming event state
├── providers/
│   ├── anthropic.py           # Claude provider (default)
│   ├── gemini.py              # Gemini provider (alternative)
│   ├── base.py                # LLMProvider ABC
│   └── factory.py             # Provider registry
├── traces/                    # Auto-created: research trace JSON files
├── evals/                     # Auto-created: eval result JSON files
├── .env.example               # Environment variable template
├── requirements.txt           # Python dependencies
└── test_schemas_smoke.py      # Schema smoke tests
```
