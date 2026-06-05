# Session 1: Dense Vector Retrieval

### [Quicklinks]()


| 📰 Module Sheet                                                                  | ⏺️ Recording                                                                                                                                           | 🖼️ Slides                                             | 👨‍💻 Repo    | 📝 Homework                                                 | 📁 Feedback                                         |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------ | ------------- | ----------------------------------------------------------- | --------------------------------------------------- |
| [Dense Vector Retrieval](../00_Docs/Modules/01_Dense_Vector_Retrieval/README.md) | [Recording!](https://us02web.zoom.us/rec/share/sHWvo0Nd1aI0SEhKecOLEX9kFGVJJAdYfsKiuTmm8t85W48Z2lnjpnzTy8jAd8R5.PwuqibGwAZhvDd8c) passcode: `C62n^@Q!` | [Session 1 Slides](https://canva.link/htfqf8i39yejyhn) | You are here! | [Session 1 Assignment](https://forms.gle/Z9qskfVaAvPjn6gz8) | [Feedback 6/2](https://forms.gle/21a2uoL9DVZPwgJP6) |


## 🏗️ How AIM Does Assignments

> 📅 **Assignments will always be released to students as live class begins.** We will never release assignments early.

Each assignment will have a few of the following categories of exercises:

- ❓ **Questions** - these will be questions that you will be expected to gather the answer to. These can appear as general questions, or questions meant to spark a discussion in your breakout rooms.
- 🏗️ **Activities** - these will be work or coding activities meant to reinforce specific concepts or theory components.
- 🚧 **Advanced Builds (optional)** - Take on a challenge. These builds require you to create something with minimal guidance outside of the documentation.

## Main Assignment

In this assignment, you will build a vector RAG application using LangChain v1, OpenAI embeddings, and Qdrant.

The main notebook is:

```text
01_Cat_Health_Vector_RAG_LangChain_Qdrant.ipynb
```

The notebook uses the bundled cat health guideline PDF in `data/cat_health_guidelines.pdf`.

### Setup

From this folder, install the environment with uv:

```bash
uv sync
```

Then open the notebook in Cursor or VS Code and select the Python/Jupyter environment created by uv.

You will also need an OpenAI API key available when running the notebook.

---

## 🏗️ Activity #1: Embedding Similarity

Run the embedding similarity primer in the notebook.

You will compare embeddings for terms like:

- `king`
- `queen`
- `banana`
- `cat`
- `veterinarian`
- `cat health guidelines`

#### ❓Question #1

Why is cosine similarity useful for dense vector retrieval?

##### ✅ Answer:

Cosine similarity is useful for dense vector retrieval because we know that vectors with similar angles have similar textual meaning regardless of their length, and if we run cosine similarity between a prompt (in embedded form) and parts [chunks] of a reference document (also in embedded form), we can find this way only the parts of the document which should have the most similar meaning, while normalizing the short question with the much longer parts of the document.
The cosine gives a clean and bounded score (between -1 and 1), which means its easy to rank the retrieved vectors. This is all provided that the embedding was done well. Also, when vectors are normalized to length 1 before the operation, the Cosine similarity becomes very efficient computationally, (simple dot product, multiply and add, without square roots).

---

## 🏗️ Activity #2: Build the Vector RAG Pipeline

Run the notebook sections that:

1. Load the PDF into LangChain `Document` objects
2. Split the document into chunks
3. Embed the chunks
4. Store the chunk embeddings in in-memory Qdrant
5. Retrieve relevant chunks with similarity scores
6. Generate an answer grounded in retrieved context

#### ❓Question #2

Why is metadata important for a RAG application?

##### ✅ Answer:

Metadata is important for rag applications because embedded chunks for documents don't hold information like recency (the date of the document), version numbers, document page numbers, etc. This means, that once you augment your prompt, without metadata you can't cite sources or filter out data that might have similar meaning to the query, but is outdated or irrelevant to the topic. Another important aspect is permissions. Some data shouldn't be available to users depending on their clearance. By attaching metadata to embedded chunks, you can have control over what users are allowed to retrieve. If RAG uses a metadata filter to narrow down to the chunks that are current, on-topic, and permitted, and then runs cosine similarity, the most meaningful matches within that filtered set can be found.

#### ❓Question #3

What tradeoff do we make when choosing chunk size and chunk overlap?

##### ✅ Answer:

When choosing chunk size, a large chunk means that even though it will more likely hold relevant contextual information to the query, it will also hold much irrelevant information, and when embedding it, it might average out in vector similarity, and get ranked low. Also, it will inflate the context window. On the other hand, a small chunk might be too small to have more than a single narrow idea, but it could potentially point to a similar direction to the query, resulting in a a highly ranked vector, but not enough context to stand on its own, rendering it useless.  When choosing chunk overlap, if the overlap is too small, ideas might get cut-off at the  beginning or end of chunks, losing valuable context. but if the overlap is too big, then it introduces redundancy. The same text will be introduced multiple times, costing with inefficient context window.

#### ❓Question #4

What does a similarity score help you understand, and what does it not prove by itself?

##### ✅ Answer:

A similarity score helps me understand whether two texts have similar textual meaning. A high score means their embedded vector representations are angularly close. But it doesn't prove the following:

1. It doesn't check if a retrieved chunk has true, false, accurate or recent information.
2. It can't check if retrieved chunks with high scores have contradictory information.
3. A similarity score doesn't take into account model biases or blind spots. Also, it depends on the model what is considered a high score.
4. A similarity score doesn't take metadata into account, which we already know is crucial for relevance of retrieved chunks.

---

## 🏗️ Activity #3: Vibe Check Retrieval Quality

Run the notebook's vibe check queries and inspect both:

- The retrieved context
- The generated answer

#### ❓Question #5

For the vibe check queries, did the retrieved context seem relevant before generation? Why or why not?

##### ✅ Answer:

---

## 🏗️ Activity #4: Tune Retrieval

Improve retrieval quality by changing one or more of:

- Chunk size
- Chunk overlap
- Retrieval `k`
- Query wording

Document what changed and whether retrieval improved.

##### Settings Changed:

- 

##### Results:

---

## Optional Deep Dive: RAG From Scratch

If you want to look underneath the library abstractions, run the optional reference notebook:

```text
02_Cat_Health_Vector_RAG_From_Scratch.ipynb
```

It builds the same retrieval pipeline again with only:

- `pypdf` for extracting text from the PDF
- Python standard-library HTTP requests for calling OpenAI
- Handcrafted document, chunking, embedding, similarity-search, vector-store, and generation primitives

This notebook is a reference walkthrough, not an additional assignment. Its purpose is to make the responsibilities hidden by LangChain, Qdrant, and provider SDKs visible.

---

## Submitting Your Homework

### Main Assignment

Follow these steps to prepare and submit your homework:

1. Pull the latest updates from upstream into the main branch of your AIE9 repo:

```bash
git checkout main
git pull upstream main
git push origin main
```

1. Start Cursor from the `01_Dense_Vector_Retrieval` folder.
2. Complete the notebook.
3. Answer the questions in this `README.md`.
4. Add, commit, and push your modified work to your origin repository.

When submitting your homework, provide the GitHub URL to your AIE9 repo.