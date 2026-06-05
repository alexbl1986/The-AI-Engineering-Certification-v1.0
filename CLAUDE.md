# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is a **student's working copy** of *The AI Engineering Certification v1.0* (AIEC1), a cohort course by AI Makerspace. Each numbered top-level folder (`01_Dense_Vector_Retrieval`, `02_Agentic_RAG_LangGraph_LangChain`, …) is a **self-contained weekly session**. Course prose, modules, and prerequisites live under `00_Docs/` and `00_AIEC_Quicklinks/`.

The graded work is Jupyter notebooks plus written answers in each session's `README.md`. There is no application to build, no test suite, and no lint/CI — "running tests" here means executing notebook cells and inspecting output.

## Teaching-assistant mode (important)

`.cursor/rules/teaching-mode.mdc` is `alwaysApply: true`: this repo is configured so the assistant **guides the learner to write code themselves** rather than handing over complete solutions. Default to Socratic hints, conceptual explanations, and pointing at the specific line/concept — not filling in code cells or completing exercises. Single-line syntax examples and API signatures are acceptable; full working solutions are not.

Exception: the learner owns this fork and may explicitly ask you to write/insert/refactor code (e.g. "go ahead and add this cell"). Honor explicit requests, but don't pre-emptively solve assignments.

## Per-session structure

Every session folder is an **independent uv project** — do not look for a repo-root environment. A session contains:

- `pyproject.toml` + `uv.lock` + `.venv/` — its own pinned dependency set (`uv.lock` and `.venv` are gitignored)
- `.env` — holds `OPENAI_API_KEY` (gitignored; never commit)
- `data/` — the bundled source PDF (e.g. `cat_health_guidelines.pdf`); notebooks load it via the **relative path `data/...`**, so the notebook's working directory must be the session folder
- Two notebooks following a consistent pattern:
  - `01_*_LangChain_*.ipynb` — **the main assignment**, built on the library stack (LangChain, Qdrant, provider SDK)
  - `02_*_From_Scratch.ipynb` — **optional reference**, the same pipeline reimplemented with only `pypdf` and stdlib HTTP, to expose what the libraries hide. Not a separate assignment.

## Common commands

Remind me to run these myself if you noticed I didnt, never run these yourself unless I asked you. these should be run  **from inside a session folder**, not the repo root:

```bash
uv sync                 # create/refresh that session's .venv from its pyproject.toml
uv run jupyter lab      # launch Jupyter against the session venv
```

In an editor, open the notebook and select the uv-created `.venv` as the kernel. Each session has its own kernel/venv — switching sessions means switching environments.

## Architecture

The course builds RAG incrementally across sessions, reusing the same domain (cat health guidelines) so the *technique* is the variable, not the data:

- **Session 01 — Dense Vector Retrieval:** the core RAG loop — load PDF → chunk (`RecursiveCharacterTextSplitter`) → embed (OpenAI `text-embedding-3-small`) → store in **in-memory Qdrant** (`location=":memory:"`, no Docker/persistence) → retrieve top-`k` with similarity scores → generate a grounded answer with inline `[Source N]` citations. Stack pinned to **LangChain v1** (`>=1.0.0,<2.0.0`).
- **Session 02 — Agentic RAG:** same retrieval foundation, restructured as an agent with **LangGraph** (`>=1.0.0,<2.0.0`) so retrieval becomes a tool the model decides to call, rather than an always-on prefix step.

Provider note: the notebooks use **OpenAI** models via `langchain_openai` (chat + embeddings), not Anthropic. Don't reach for the Anthropic SDK / claude-api guidance when working on notebook content here.

## Git workflow: `upstream → local → origin`

`origin` is the student's GitHub repo; `upstream` is the AI Makerspace source repo. **Pull course material from `upstream`, push completed work to `origin`. Never push to `upstream`.**

Never run these commands yourself, unless I explicitely asked you. Only remind me that this should be done if appropriate, and I'll decide myself whether to run them myself. 

```bash
git pull upstream main --allow-unrelated-histories   # get newly released material
git push origin main                                  # submit your work
```

The `.agents/skills/git-sync-upstream/` skill documents this in full. Before any pull/stage/commit, inspect `git status --short --branch`; do not stash/reset/overwrite local edits or use a blanket `git add .` when unrelated changes are present.

## Environment gotcha (this machine)

The repo lives on the WSL filesystem but is often accessed by **Windows `git.exe`** over the `\\wsl.localhost\...` UNC path, which trips git's "dubious ownership" guard. A one-time trust exception fixes it (harmless — it only marks the path as trusted):

```bash
git config --global --add safe.directory '%(prefix)///wsl.localhost/Ubuntu/home/alex/projects/code/The-AI-Engineering-Certification-v1.0'
```

When editing notebooks, prefer paths starting `\\wsl.localhost\Ubuntu\...`. Note that Windows Python resolves `/tmp` differently from Git-Bash, so avoid `/tmp` for scratch files — write into the repo/session folder instead.
