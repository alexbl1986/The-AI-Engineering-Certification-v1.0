# PyMuPDF + deterministic RTL-repair pass for desk-review extraction

The desk reviews are Hebrew (RTL) PDFs, and no extractor handles them cleanly out of the
box. A head-to-head on the real 08/07/26 daily review showed: **pypdf** extracts content
perfectly (correct punctuation and prefix order, zero artifacts) but flattens the document
to one word per line with no paragraph signal; **PyMuPDF** preserves the full structural
map (blocks + font sizes: 13pt section headers, 11pt sub-headers) that section-aware
chunking needs, but introduces ~173 systematic artifacts — leading punctuation that
belongs at the end of the same token (`,זיכרון` → `זיכרון,`) and a maqaf misplaced around
one-letter Hebrew prefixes on Latin acronyms (`-הAI` → `ה-AI`). Both artifact classes are
mechanical, so we chose **PyMuPDF plus a small deterministic repair pass**, unit-tested
against pypdf's output as ground truth. We rejected pypdf (structure loss is unrecoverable,
and structure is what chunking needs most) and vision-LLM page parsing (cleanest output,
but non-deterministic ingestion would muddy the Task 5/6 eval comparisons; also an API
dependency for a job that runs offline in seconds). PyMuPDF is AGPL: acceptable because
the cert repo is public; if commercialized later, the extraction step is one isolated
ingestion module and can be swapped without touching the agent.
