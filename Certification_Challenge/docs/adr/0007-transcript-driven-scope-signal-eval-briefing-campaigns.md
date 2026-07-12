# Transcript-driven scope: pre-trade signal eval, daily briefing, campaign-scoped history

> **Superseded in part (2026-07-11):** the `scan_deadlegs` tool and `dead_leg_floor`
> ($0.50) named below were dropped after the trader confirmed the Thursday-flush rule
> wasn't worth building as a machine check. The rest of this ADR stands.

Reviewing the trader's real transcripts with his previous chat-agent setup revealed that
his highest-frequency use case — evaluating an incoming trade signal — was missing from
the route list, and that his rulebook is larger and more precise than the policy record
we had scoped. The transcripts also validated the architecture twice: the agent read a
14,700 SEK position as USD and ordered an immediate trim ("confidently wrong compliance
verdict from LLM arithmetic" — the exact failure ADR-0002 exists to prevent), and its
per-question search of the 9,489-line annual statement was slow enough that the trader
asked it to stop (the parse-once ledger, justified in his own words).

## Decisions

**1. New route: `trade_signal_eval` (in cert scope, built last behind a clean seam —
first cut if the schedule slips).** Input: shorthand signal pasted into chat ("AAOI 150
NEXT WEEK 3.1"). v1 scope: intake parses ticker/strike/type/premium (defaults per the
trader's own protocol: call if unspecified, nearest weekly/monthly **verified against
the live yfinance option chain**); sizing from the policy record (options 1% / stocks 3%
of NAV → max contracts at the signal premium); exit plan from the DTE-tier matrix;
desk bias + tier cross-ref via RAG (the desk's Tier-2 "high-beta optionality" label caps
sizing at standard — a deterministic rule keyed on a retrieved label); existing-inventory
conflict check against the snapshot; **fail-loud on IV rank** ("not available from free
data — check manually; if >70% your rules mandate a spread" — today's IV from the chain
is shown, the 52-week rank cannot be computed without paid IV history). Output: a typed
`TradePlan`. Out even within this route: IV-rank computation, automated roll grading,
Discord ingestion (paste-only for cert).

**2. New route: `daily_briefing` (in — a composition, not a feature).** Trigger phrases
("morning briefing", "start my day"); pre-fetch = union of existing calls: exposure
check + scale-out scan + dead-leg scan (Thursday-aware) + hedge ratio + desk-summary
retrieval over the latest daily & weekly + index technical snapshot. One synthesis type
(`DailyBriefing`): book status → rule flags → desk's read → what to watch. Replaces the
auto-boot sequence the trader hand-built in his previous setup.

**3. `get_trades` defaults to the current active campaign.** Campaign v1 = all ledger
fills in the same contract (symbol+strike+expiry) while the position has been
continuously open; closes when quantity returns to zero; stocks analogously (fills since
last flat). Default answer: entry date/premium, partial scales, net cost basis,
realized-so-far, house-money status. Full history (closed campaigns) only on explicit
request — an intake flag, not a separate tool. Note: his latency complaint doesn't apply
to a Postgres ledger (both queries are milliseconds); we adopt his preferred *answer
shape*, not a performance trade-off. Roll chains do NOT merge campaigns in v1.

**4. Policy record scope: rules a cert route reads, plus three cheap checks.** In:
sizing caps (options 1% / stocks 3% new / **6% existing-holding cap**), 20% max active
offensive exposure, options ≤10% NAV, **hedge ratio = put value ÷ call value, target
10–15%** (the precise formula from his rules — not a %-of-NAV figure; whether the
cross-hedge ~15% uses the same ratio logic must be confirmed with the trader before
Day 3), cross-hedge, scale-out (+100%/+200%), the DTE exit matrix (0–2 / 3–7 / 8–30 /
31+: size tier, SL, TP, scale/roll clauses), IV-shield threshold (70% — enforced as a
manual-check reminder), `dead_leg_floor` ($0.50, commission-aware) powering a
`scan_deadlegs` tool, and moonshot thresholds (+150% trigger / +50% stop) as a
**labeled proxy** flag in the scale-out scan (true rule is path-dependent — "ever
touched +150%" — which needs option price history that free data doesn't have and the
app is upload-blind between snapshots). House-money status is a derived field of
campaign history (descriptive, not a rule check). **Excluded from machine checking:**
roll rules (0.50-delta target and 2×-cash: no free Greeks; computing delta = hand-rolled
Black-Scholes over yfinance IV that is often stale/zero on thin strikes — a
precise-looking wrong delta feeding a compliance verdict is the false-confidence failure
this app exists to prevent), roll-chained campaigns, post-trade **screenshot** audits
(compliance verdicts never hang off pixels; the synthesis prompt redirects to re-upload
the tactical export — the trader's existing habit — after which `status_check` on the
fresh snapshot IS the post-trade audit). An untested rule field that looks
machine-enforced is a liability, so nothing enters the record that no code reads; the
record carries a `version` field for Demo Day extension.

Demo Day items this creates: Discord signal bridge (push → auto-eval), IBKR API replacing
CSV uploads, paid options data (unlocks honest IV rank, Greeks/roll grading, historical
option prices), roll-aware campaign chaining. The full plain-language gap list for the
trader lives in the gitignored `docs/PROTOTYPE_GAPS.md`.
