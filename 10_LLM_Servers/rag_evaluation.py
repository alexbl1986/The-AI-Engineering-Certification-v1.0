"""Incremental RAG evaluation — open-source stack vs OpenAI stack, scored with RAGAS.

Run:
    uv run python rag_evaluation.py

Built in layers. Expensive/curated artifacts live in ``artifacts/`` and are reused
on later runs. To rebuild a stage, delete its artifact file and run again.

The evaluation dataset is reused from Assignment 5 (same cat-health PDF —
verified identical by checksum), curated with the Activity #1 review decisions,
and frozen to ``artifacts/eval_dataset.jsonl``.

Both pipelines share the SAME open-source retrieval (Ollama ``qwen3-embedding:4b``);
only the generator differs (Groq ``gpt-oss-20b`` vs OpenAI ``gpt-4.1-mini``), so
end-to-end differences are attributable to the model.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langsmith import tracing_context

from app.rag import _build_rag_graph

load_dotenv()  # reads 10_LLM_Servers/.env

# --- paths ------------------------------------------------------------------
ARTIFACTS_DIR = Path("artifacts")  # kept in the repo (not gitignored)
RAW_TESTSET = ARTIFACTS_DIR / "cat_health_synthetic_testset.jsonl"  # copied from Assignment 5
EVAL_DATASET = ARTIFACTS_DIR / "eval_dataset.jsonl"  # curated + frozen; delete to re-curate
RUNS_OPENSOURCE = ARTIFACTS_DIR / "runs_opensource.jsonl"  # Groq gpt-oss-20b; delete to re-run
RUNS_OPENAI = ARTIFACTS_DIR / "runs_openai.jsonl"  # OpenAI gpt-4.1-mini; delete to re-run
SCORES_OPENSOURCE = ARTIFACTS_DIR / "scores_opensource.jsonl"  # RAGAS scores; delete to re-score
SCORES_OPENAI = ARTIFACTS_DIR / "scores_openai.jsonl"  # RAGAS scores; delete to re-score

DATA_DIR = os.environ.get("RAG_DATA_DIR", "data")
JUDGE_MODEL = os.environ.get("RAGAS_JUDGE_MODEL", "gpt-5.4-mini")

# LangSmith projects: keep generation traces and RAGAS-eval (judge) traces in
# separate projects instead of everything landing in "default".
GEN_PROJECT = os.environ.get("LANGSMITH_PROJECT", "aie-s10-generation")
EVAL_PROJECT = os.environ.get("LANGSMITH_EVAL_PROJECT", "aie-s10-ragas-eval")


# --- helpers ----------------------------------------------------------------
def as_context_list(value) -> list[str]:
    """Coerce a contexts field into a clean list[str] for RAGAS.

    Guards against a lone string (wrapped, not exploded into characters) and
    coerces each element to str (e.g. stray numpy types from pandas).
    """
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


# --- Layer 1: reuse + curate the evaluation dataset -------------------------
def curate_testset() -> pd.DataFrame:
    """Apply the Activity #1 review decisions to Assignment 5's testset.

    - Drop examples 1 & 2 (author / meta questions a real user wouldn't ask).
    - Repair example 4's wording (index 1 after the drop+reset) to read naturally;
      its reference / reference_contexts still support the reworded question.
    """
    testset_df = pd.read_json(RAW_TESTSET, orient="records", lines=True)

    approved = testset_df.drop(index=[0, 1]).reset_index(drop=True)
    approved.loc[1, "user_input"] = (
        "How to address dental care and oral health as part of an "
        "individualized lifelong feline healthcare strategy?"
    )
    return approved


def load_eval_dataset() -> pd.DataFrame:
    """Load the frozen eval dataset, or curate + freeze it on first run."""
    print("\n=== Layer 1: evaluation dataset ===")

    if EVAL_DATASET.exists():
        df = pd.read_json(EVAL_DATASET, orient="records", lines=True)
        print(f"  reusing {EVAL_DATASET}  ({len(df)} examples)")
    else:
        df = curate_testset()
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        df.to_json(EVAL_DATASET, orient="records", lines=True)
        print(f"  curated from {RAW_TESTSET.name} -> {EVAL_DATASET}  ({len(df)} examples)")
        print("  (delete that file to re-curate from the raw testset)")

    print(f"  columns: {list(df.columns)}")
    for i, row in df.iterrows():
        n_ctx = len(row["reference_contexts"]) if row.get("reference_contexts") is not None else 0
        print(f"\n  [{i}] synthesizer: {row.get('synthesizer_name', '?')}  ({n_ctx} reference context(s))")
        print(f"      Q: {row['user_input']}")
        print(f"      reference: {str(row['reference'])[:160]}...")

    return df


# --- Layer 2: run both pipelines over the eval questions --------------------
def _ollama_embeddings():
    """Shared open-source retrieval embeddings (same for both pipelines)."""
    return OllamaEmbeddings(model="qwen3-embedding:4b", base_url="http://localhost:11434")


def _groq_generator():
    """Open-source generator: Groq-hosted gpt-oss-20b."""
    return ChatOpenAI(
        model=os.environ.get("GROQ_CHAT_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
        openai_api_key=os.environ["GROQ_API_KEY"],
        openai_api_base="https://api.groq.com/openai/v1",
    )


def _openai_generator():
    """OpenAI generator: gpt-4.1-mini (uses OPENAI_API_KEY from env)."""
    return ChatOpenAI(
        model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        temperature=0,
    )


def run_pipeline(label: str, provider: str, generator_llm, artifact_path: Path, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Run one RAG pipeline (shared Ollama retrieval + given generator) over the
    eval questions, recording retrieved_contexts + response. Frozen/reused.

    Generation traces are routed to the ``GEN_PROJECT`` LangSmith project and
    tagged with ``provider`` (e.g. "open_source" / "openai") so the two pipelines
    are cleanly separable in the dashboard and the later cost pull."""
    print(f"\n=== Layer 2: pipeline [{label}] ===")

    if artifact_path.exists():
        df = pd.read_json(artifact_path, orient="records", lines=True)
        print(f"  reusing {artifact_path}  ({len(df)} rows) — delete to re-run")
        return df

    graph = _build_rag_graph(
        DATA_DIR,
        embedding_model=_ollama_embeddings(),
        generator_llm=generator_llm,
    )

    rows = []
    with tracing_context(project_name=GEN_PROJECT, tags=[provider, "layer2-generation"]):
        for i, row in eval_df.iterrows():
            result = graph.invoke(
                {"question": row["user_input"]},
                config={
                    "run_name": f"gen-{provider}-q{i}",
                    "metadata": {"provider": provider, "q_index": int(i)},
                },
            )
            retrieved = [doc.page_content for doc in result.get("context", [])]
            response = result.get("response", "")
            rows.append(
                {
                    "user_input": row["user_input"],
                    "reference": row["reference"],
                    "reference_contexts": as_context_list(row["reference_contexts"]),
                    "retrieved_contexts": as_context_list(retrieved),
                    "response": response,
                }
            )
            print(f"\n  Q: {row['user_input'][:75]}")
            top = (retrieved[0][:120].replace(chr(10), ' ') + " ...") if retrieved else "(none)"
            print(f"     retrieved {len(retrieved)} ctx | top: {top}")
            print(f"     response: {response[:150].replace(chr(10), ' ')} ...")

    df = pd.DataFrame(rows)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    df.to_json(artifact_path, orient="records", lines=True)
    print(f"\n  saved -> {artifact_path}")
    return df


# --- Layer 3: score both pipelines with RAGAS ------------------------------
_METRIC_COLS = None  # populated after first scoring; the non-input columns


def _apply_vertex_shim():
    """Patch ragas 0.4.x's eager, dead ``langchain_community.chat_models.vertexai``
    import (that module was removed in langchain-community v1). Vertex is never
    used here — this just lets ``import ragas`` succeed."""
    import sys
    import types

    if "langchain_community.chat_models.vertexai" not in sys.modules:
        vx = types.ModuleType("langchain_community.chat_models.vertexai")
        vx.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules["langchain_community.chat_models.vertexai"] = vx
    import langchain_community.llms as _llms

    if not hasattr(_llms, "VertexAI"):
        _llms.VertexAI = type("VertexAI", (), {})


def _judge():
    """Judge LLM (``JUDGE_MODEL``) + OpenAI embeddings, wrapped for RAGAS."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import OpenAIEmbeddings

    llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0))
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    )
    return llm, emb


def score_pipeline(label: str, provider: str, runs_df: pd.DataFrame, scores_path: Path) -> pd.DataFrame:
    """Score one pipeline's runs with the five RAGAS metrics. Frozen/reused.

    Judge/eval traces are routed to the separate ``EVAL_PROJECT`` LangSmith
    project so eval overhead is not mixed in with generation cost."""
    global _METRIC_COLS
    print(f"\n=== Layer 3: score [{label}] ===")

    if scores_path.exists():
        scores_df = pd.read_json(scores_path, orient="records", lines=True)
        print(f"  reusing {scores_path}  ({len(scores_df)} rows) — delete to re-score")
    else:
        _apply_vertex_shim()
        os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import (
            AnswerAccuracy,
            ContextEntityRecall,
            Faithfulness,
            LLMContextRecall,
            NoiseSensitivity,
        )

        samples = [
            {
                "user_input": row["user_input"],
                "reference": row["reference"],
                "reference_contexts": as_context_list(row["reference_contexts"]),
                "retrieved_contexts": as_context_list(row["retrieved_contexts"]),
                "response": row["response"],
            }
            for _, row in runs_df.iterrows()
        ]
        dataset = EvaluationDataset.from_list(samples)
        judge_llm, judge_emb = _judge()
        metrics = [
            LLMContextRecall(),
            Faithfulness(),
            AnswerAccuracy(),
            ContextEntityRecall(),
            NoiseSensitivity(),
        ]
        print(f"  scoring {len(samples)} examples with judge={JUDGE_MODEL} ...")
        with tracing_context(project_name=EVAL_PROJECT, tags=["ragas-eval", provider]):
            result = evaluate(dataset=dataset, metrics=metrics, llm=judge_llm, embeddings=judge_emb)
        scores_df = result.to_pandas()
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        scores_df.to_json(scores_path, orient="records", lines=True)
        print(f"  saved -> {scores_path}")

    # Identify the metric columns (everything that isn't an input field)
    input_cols = {"user_input", "reference", "reference_contexts", "retrieved_contexts", "response"}
    metric_cols = [c for c in scores_df.columns if c not in input_cols]
    _METRIC_COLS = metric_cols

    for i, row in scores_df.iterrows():
        scores = "  ".join(f"{c}={row[c]:.3f}" if pd.notna(row[c]) else f"{c}=NaN" for c in metric_cols)
        print(f"  [{i}] {scores}")
        print(f"      Q: {str(row['user_input'])[:70]}")

    return scores_df


def compare(os_scores: pd.DataFrame, oai_scores: pd.DataFrame) -> None:
    """Print the aggregate metric-by-provider comparison table. Token usage, cost
    and latency are read directly from the LangSmith dashboards (GEN_PROJECT for
    generation, EVAL_PROJECT for the judge), not computed here."""
    print("\n=== Layer 3: comparison (mean per metric) ===")
    summary = pd.DataFrame(
        {
            "open_source": os_scores[_METRIC_COLS].mean(numeric_only=True),
            "openai": oai_scores[_METRIC_COLS].mean(numeric_only=True),
        }
    )
    summary["delta (openai - os)"] = summary["openai"] - summary["open_source"]
    print(summary.round(3).to_string())
    print(
        "\n  (context_recall / context_entity_recall are retrieval-only, so they match "
        "across pipelines;\n   faithfulness / answer_accuracy / noise_sensitivity are where "
        "the generator difference shows.)"
    )


def main():
    eval_df = load_eval_dataset()

    os_runs = run_pipeline(
        "open-source: Ollama qwen3-embedding + Groq gpt-oss-20b",
        "open_source",
        _groq_generator(),
        RUNS_OPENSOURCE,
        eval_df,
    )
    oai_runs = run_pipeline(
        "openai-gen: Ollama qwen3-embedding + OpenAI gpt-4.1-mini",
        "openai",
        _openai_generator(),
        RUNS_OPENAI,
        eval_df,
    )

    os_scores = score_pipeline("open-source (Groq gpt-oss-20b)", "open_source", os_runs, SCORES_OPENSOURCE)
    oai_scores = score_pipeline("openai-gen (gpt-4.1-mini)", "openai", oai_runs, SCORES_OPENAI)
    compare(os_scores, oai_scores)


if __name__ == "__main__":
    main()
