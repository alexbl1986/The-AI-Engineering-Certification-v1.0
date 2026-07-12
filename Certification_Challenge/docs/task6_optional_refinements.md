# Task 6 — Optional Refinements to Consider

> A running list of refinements that surfaced during the build, each considered
> but **not** settled. These are candidate improvements — some may land before
> submission, some become Demo-Day items, some are deliberately declined. Each
> entry states the current behavior, the proposed change, the trade-off, and a
> status so the decision is explicit rather than implicit.

---

## R1. Deterministic ticker extraction in the scoper

**Status:** open — not yet decided.

**Current behavior.** The intake/scoper node (`app/graphs/trading_assistant/scope.py`)
asks the LLM to produce the whole `Scope` in one structured-output call, including
`tickers`. So the *symbols* are extracted by the model alongside the intent.

**Observation.** The symbols the scoper returns drive real, consequential downstream
work: `trade_history` calls `get_trades(trades, ticker)` per ticker, and
`trade_signal_eval` sizes a position off the parsed symbol. A wrong or missed ticker
there is not a cosmetic error — it queries the wrong campaign or sizes the wrong name.
Meanwhile we **already have** deterministic ticker extraction in the RAG layer:
`app/rag/chunk.py::_extract_tickers` (regex `\b[A-Z]{1,5}\b` intersected with a
ticker whitelist). Symbol recognition is exactly the kind of narrow, verifiable task
that code does more reliably than a language model.

**Proposed refinement.** Split the responsibility: keep the **LLM for intent**
(fuzzy language → one of the 10 routes, the hypothetical flag, clarification), but
extract **symbols in code** — regex + a whitelist of the user's actual holdings /
known desk-universe symbols — either replacing or cross-checking the model's
`tickers`. This mirrors the graph's guiding principle (ADR-0006): the LLM interprets
language; deterministic code owns anything safety- or correctness-critical.

**Trade-off / nuance (why it isn't a clean win).**
- A pure regex+whitelist can't resolve a *company name* to a ticker ("Taiwan Semi" →
  `TSMC`) or an *implied* holding, which the LLM handles for free.
- The whitelist has to come from somewhere current — the parsed snapshot/ledger
  symbols plus the desk-review ticker set — so it's per-user state, not a static list.
- A hybrid (code extracts + validates against the whitelist; LLM fills gaps for
  names not matched) captures most of the reliability gain without losing name
  resolution, at the cost of a second reconciliation step.

**Recommendation.** Defer past the base agent build; revisit alongside the Task 5
eval harness. The routing rubric will quantify how often the LLM actually mis-extracts
a ticker on real phrasings — if that rate is low, the added machinery isn't worth it;
if it's material (especially on `trade_signal_eval`), do the hybrid version. Decide
with data, not a priori.

---

## R2. Automated semantic-accuracy check for scoper routing

**Status:** planned (folds into Task 5), noted here for completeness.

**Current behavior.** The scoper's output *shape* is guaranteed by the Pydantic schema
(`with_structured_output(Scope)` — intents can only be the 10 `Literal` values), and
the one safety-critical semantic rule is enforced deterministically (`_normalize_scope`
strips `policy_change` from a hypothetical). But whether the LLM *labels correctly*
(e.g. "sell everything and buy dogecoin" → should be `off_topic`, was observed as
`rebalance_advice`) has **no assertion** — the unit tests use a stub model and check
only the graph wiring.

**Proposed refinement.** The Task 5 eval layer-3 routing rubric (paraphrase variants,
daily-vs-weekly collision, cold-start, off-topic) becomes the semantic check. This is
already in the plan; the note here is to make sure the routing cases explicitly cover
misroutes like the dogecoin example and any ticker mis-extraction from R1.

**Recommendation.** Build with Task 5; no separate work.
