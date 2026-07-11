# Data classification and access policy for a public repo and endpoint

The cert submission requires a public GitHub repo and a public deployment, but the working
data is someone else's real financial records (the trader's IBKR statements: name, account
number, full trade history) and the desk's paid research PDFs. Policy, by class:

1. **Real IBKR statements — never committed.** Gitignored from day one. The repo ships a
   **synthetic sample statement** (same 27-section IBKR format, fake identity, scaled/
   shuffled trades) as the unit-test fixture and the file graders upload to try the app.
2. **Real desk-review PDFs — never committed** (conservative default; may relax if the
   desk confirms the content is not proprietary — pending). The repo ships **synthetic
   Hebrew desk reviews** in the desk's template styles (one daily, one weekly — the
   runtime corpus holds only the latest of each; stance-shift history is Demo Day scope).
3. **Eval artifacts — split by sensitivity.** Committed: question sets, ground-truth
   judgments in our own words, summary metrics tables. Not committed: raw retrieval traces
   and quoted chunks — those live in the private LangSmith project (EU endpoint), which is
   the full evidence record. Tracing sits behind an env flag so real-use sessions can run
   untraced.
4. **The deployed endpoint is gated.** The running app — not the repo — is where real data
   lives (uploads land in Postgres/Qdrant). Simple credential login (no OAuth; two known
   users, server-side check, session token). The authenticated `user_id` scopes every
   LangGraph store/checkpointer namespace, the portfolio snapshot, **and the desk-review
   corpus** (a `user_id` payload filter in Qdrant), so the demo user (synthetic data) and
   the real trader (real data) are fully isolated — a shared corpus would let the demo
   user's synthetic desk reviews poison the real trader's answers. "Public endpoint"
   means reachable, not unauthenticated — graders get credentials privately in the
   submission.

Rationale: a push to a public repo is irreversible, and an ungated deployment holding a
real book is a larger leak than the repo could ever be. Synthetic fixtures beat excerpts
because excerpting paid research is still redistribution.
