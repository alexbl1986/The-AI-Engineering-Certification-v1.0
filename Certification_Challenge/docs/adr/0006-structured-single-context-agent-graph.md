# Structured single-context agent graph; subagents rejected

The agent is one reasoning context wrapped in deterministic scaffolding, not a plain
tool-calling loop and not a multi-agent system. The LangGraph shape:

1. **Intake/scoper node** — structured output: **multi-label** intent ({status_check,
   desk_question, rebalance_advice, trade_history, performance_review, policy_change,
   market_regime, trade_signal_eval, daily_briefing, off_topic} — the last two added by
   ADR-0007 from the trader's real-usage transcripts) plus extracted entities and a
   `hypothetical` flag ("if I raised my cap…" must analyze, not write). Reads thread history; ambiguity gets one
   clarification round, then best-effort with stated assumptions; off_topic is refused.
2. **Deterministic pre-fetch by route** — the calls that must happen are graph code, not
   LLM discretion: status/rebalance always runs exposure + scale-out + positions + quotes;
   desk questions always retrieve. Multi-label intent fetches the union.
3. **Agent node** with the full tool roster for the long tail, under an iteration cap.
   Web-search results are demoted to quoted evidence (prompt-injection surface).
4. **Synthesis node** — per-route typed outputs (an AssessmentReport with per-rule
   pass/breach, actions, and evidence refs for status/rebalance; simpler types for e.g.
   trade history). Synthesis copies numbers verbatim from the evidence table — it never
   recomputes.
5. **Deterministic audit node** — code, not LLM: every number in the answer must match a
   tool output, every cited rule must exist in the policy record, no price may appear if
   the quote was unavailable, and no portfolio number may appear if its backing store was
   never uploaded. One bounce-back, then deliver with a warning banner.
6. **policy_change routes through a LangGraph `interrupt()`** — the human-approval gate
   required by ADR-0003.

**Cold-start contract.** Snapshot, ledger, and corpus reads must distinguish *"never
uploaded"* from *"empty result"*. A pandas query over a never-populated snapshot returns
an empty DataFrame — which `check_exposure` would read as "options at 0%, fully within
policy": a confident, audit-passing, wrong answer. So tools and pre-fetch return a typed
`MissingData(store, remedy)` value instead of an empty frame when the backing store was
never uploaded (generalizing the existing `scan_scaleout`-refuses-without-ledger rule);
synthesis renders what *is* answerable and adds an explicit "to answer the rest, upload
X" line rather than refusing the whole question; the audit rule above enforces it.
Routes needing no uploads work on day one: `market_regime` (live quotes/Tavily) and
policy questions/edits (record is seeded from default config at first login).

**Subagents-as-tools were considered and rejected.** The product's value is the synthesis
across portfolio, policy, quotes, and desk thesis; subagent boundaries cut exactly at that
seam — the supervisor would reason over summaries-of-summaries and lose the specific
contracts and tickers that make a recommendation actionable. None of the conditions that
justify subagents hold here (small tool outputs, convergent not parallel questions, ~10
tools). Multi-agent remains the Demo Day evolution if a large research-library corpus
lands (context-bloat condition would then hold).

**Deliberate exclusions** (each considered, each a permanent or scope-level "no"):
chart-pattern recognition in the synthesis path (subjective; vision-LLM chart reading is
confidently wrong — dangerous in a trading tool); counterfactual backtests of the trading
rules (no free historical option data; approximating from the underlying produces
misleading results); statistical "validity" claims about the strategy (six months of one
trader's trades cannot support them — the agent states this caveat instead); desk-signal
profitability attribution (no trade-to-signal log exists); and, per ADR-0007: IV-rank
computation (no free 52-week IV history — fail-loud manual-check reminder instead),
machine-graded rolls (no free Greeks; a computed-from-stale-IV delta is false
confidence), path-dependent moonshot tracking (proxy check on current gain only), and
compliance verdicts from screenshots (redirect to re-uploading the tactical export).
