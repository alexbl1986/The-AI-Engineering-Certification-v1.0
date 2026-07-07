# BODY TEST

If you can read this, the body Markdown is fine.

| 📰 Session Sheet                                  | ⏺️ Recording                           | 🖼️ Slides                                   | 👨‍💻 Repo       | 📝 Homework                                              | 📁 Feedback                        |
| ------------------------------------------------- | -------------------------------------- | ------------------------------------------- | ------------- | -------------------------------------------------------- | ---------------------------------- |
| [Session 10: LLM Servers](https://github.com/AI-Maker-Space/The-AI-Engineering-Certification-v1.0/tree/main/00_Docs/Modules/10_LLM_Servers) |[Recording!](https://us02web.zoom.us/rec/share/zXd6__uO2RwCmJUmNyGKY01sbwYjjrkpDDNPbfK_Es0MANaqRpFOqqYX4sEVYY1d.gJwTZk1729siXnjj) <br> passcode: `^1$@$R@.`| [Session 10 Slides](https://canva.link/953giejzt5igxvw) |You are here! | [Session 10 Assignment](https://forms.gle/hc1B1bkTuXzNVrZU) | [Feedback 7/2](https://forms.gle/uj2QvYjHfHKFFQ8a6) |

**⚠️!!! PLEASE BE SURE TO SHUTDOWN YOUR DEDICATED ENDPOINT ON FIREWORKS AI WHEN YOU'RE FINISHED YOUR ASSIGNMENT !!!⚠️**

# Build 🏗️

In today's assignment, we'll be creating Fireworks AI endpoints, and then building a RAG application.

- 🤝 Breakout Room #1
  - Set-up Open Source Endpoint (Instructions [here](./ENDPOINT_SETUP.md)) ((This process may take 15-20min.))
  - Test Endpoint and Embeddings with the `endpoint_slammer.ipynb` notebook.

- 🤝 Breakout Room #2
  - Use the Open Source Endpoints to build a RAG LangGraph application

# Ship 🚢

The completed notebook and your RAG app/notebook!

### Deliverables

- A short Loom of either:
  - the notebook and the RAG application you built for the Main Homework Assignment; or
  - the notebook you created for the Advanced Build

# Share 🚀

Make a social media post about your final application!

### Deliverables

- Make a post on any social media platform about what you built!

Here's a template to get you started:

```
🚀 Exciting News! 🚀

I am thrilled to announce that I have just built and shipped a RAG application powered by open-source endpoints! 🎉🤖

🔍 Three Key Takeaways:
1️⃣
2️⃣
3️⃣

Let's continue pushing the boundaries of what's possible in the world of AI and question-answering. Here's to many more innovations! 🚀
Shout out to @AIMakerspace !

#LangChain #QuestionAnswering #RetrievalAugmented #Innovation #AI #TechMilestone

Feel free to reach out if you're curious or would like to collaborate on similar projects! 🤝🔥
```

# Submitting You Homework

## Main Homework Assignment

Follow these steps to prepare and submit your homework assignment:

1. Follow the instructions in `ENDPOINT_SETUP.md`
2. Replace both `model` values in `endpoint_slammer.ipynb` with the `gpt-oss` endpoint you created in Step 1
3. Run the code cells in `endpoint_slammer.ipynb`
4. Respond to the questions in the section below
5. Build a sample RAG
6. Record a Loom video reviewing what you have learned from this session

**⚠️!!! PLEASE BE SURE TO SHUTDOWN YOUR DEDICATED ENDPOINT ON FIREWORKS AI WHEN YOU HAVE FINISHED YOUR ASSIGNMENT !!!⚠️**

## Questions

### ❓ Question #1:

What is the difference between serverless and dedicated endpoints?

#### ✅ Answer:

- A serverless endpoint allows me to call an open source model similarly to a commercial model, paying only per token usage. But I have no quality-of-service guarantee: On rush hours I can get rate/throughput limited and get higher latency, and I can't customize the inference server's configuration. Also, If I'm a heavy user, the costs could quickly get out of hand.
- A dedicated endpoint means I hire a personal dedicated server, which only I use, so I get guaranteed QoS. I can set different knobs, like choose the GPU type to improve latency/throughput and support stronger models, number of replicas for scaling to many users, quantization to control the precision-memory-throughput tradeoff, and Scale-to-Zero to auto idle after a set period of time to control costs (but also introduce cold-starts). But when its up, it costs an hourly rate regardless of my usage, which is great if I call the model frequently (past the break-even usage level), but it also means I need to control its up-time or I'll get charged whenever I don't use it. 

### ❓ Question #2:

Why is it important to consider token throughput and latency when choosing an LLM for user-facing applications?

#### ✅ Answer:

I have experienced this first-hand in the assignment, when I was throughput-limited by groq on the third consecutive model call. When I passed 8K tokens (each call's input was RAG augmented with which causes a spike in the number of input-tokens), the subsequent calls took over 30 seconds instead of ~1 second to get a response. For a user-facing application, under normal conditions a user expects a latency TTFT (time-to-first-token) of no more than a few seconds to get a response from an agent, (a regular, non-deep-research call), and then a smooth streaming response. If the throughput ceiling is reached, the streaming rate might drop to below the user's reading speed, which will be a very negative experience, failing to meet expecations and hurt the user's trust. Also, for multiple concurrent users, throttling might occur if the the throughput ceiling is reached, causing an unexpected higher TTFT.

## Activity 1: RAGAS Evaluation with Cost Analysis

Use RAGAS to evaluate your open-source Fireworks AI powered RAG app against an OpenAI `gpt-4.1-mini` powered equivalent. Compare retrieval quality, answer faithfulness, and end-to-end accuracy across both providers.

Additionally, instrument both pipelines with **LangSmith** to capture token usage and cost per query. Use LangSmith's tracing and cost dashboards to compare the total cost of running each provider at scale. Include your evaluation results, cost breakdown, and analysis in your Loom video.


#### ✅ Answer:

- Ran an open-source pipeline (Ollama qwen3-embedding:4b + Groq gpt-oss-20b) vs the 'gpt-4.1-mini' as the chat model (same Ollama qwen3-embedding:4b embedding model to isolate the RAGAS end-to-end evaluation metrics differences to the chat model change).
- Ragas judge model was chosen to be gpt-5.4-mini for stronger reasoning for the judge and an independent model from the evaluated models (to negate "model narcissism").
- The dataset chosen was the SDG generated in assignment 5, and validated in activity 1 of assignment 5. (A final curated dataset of 4 triples) This is a validated ground-truth reference which is critical for the RAGAS eval results.
- Metrics chosen were the RAGAS metrics from assignment 6.
- Full evaluation code is in rag_evaluation.py


- RAGAS eval results:


| Metric                           | open_source | openai | delta (openai - os) |
|-----------------------------------|-------------|--------|---------------------|
| context_recall                    | 0.750       | 0.750  | 0.000               |
| faithfulness                      | 0.455       | 0.736  | 0.281               |
| nv_accuracy                       | 0.562       | 0.688  | 0.125               |
| context_entity_recall             | 0.192       | 0.197  | 0.005               |
| noise_sensitivity (mode=relevant) | 0.288       | 0.359  | 0.071               |



- Results analysis:
  - context_recall the same and context_entity_recall scored almost the same (probably attributed to llm entity-extraction non-determinism between different runs), which is as expected, as the same retrieval was used for both pipelines.
  - gpt-4.1-mini's performance in faithfulness was significantly better (0.736 vs 0.455).
  - When tracing the answers from both models in langsmith, for question 2  "How to address dental care and oral health as part of an individualized lifelong feline healthcare strategy?" - the open source model responded with "I don't know" and scored 0 in faithfulness and answer accuracy. On the other hand, the openAI model gave a response and scored faithfulness=0.615  nv_accuracy=0.500. When analyzing the actual response, the answer was vague but mostly faithful to the to the retrieved context and also similar to the reference answer. For the other questions, the open source model did manage to answer the questions using the retrieved context, but over-elaborated or hallucinated data that wasn't in the retrieved context. (for example, for question 1, the open-source model made up that "The clinician reviews the cat’s current parasite‑prevention protocol (fleas, ticks, intestinal worms, etc.)" even though it wasn't in the retrieved context.) The open source model also hallucinated additional data when answering question 3. In conclusion, the openAI model was significantly more faithful which aligns with the score overall. 
  - regarding answer accuracy, even though for the first answer the open source hallucinated data, it still got a perfect accuracy score of 1.0 vs 0.75 for the openAI model, which significantly skews the result and when negating that, the accuracy gap widens to 0.25 in favor of the openAI model which is in line when reviewing the traced data.
  - Regarding the noise_sensitivity score, which is marginally lower for the open-source model:
    - For the question 1, which the open source model hallucinated deworming and flea-tics and the openAI model didn't, noise sensitivity was similar to the openAI model (0.4 vs 0.35), which skews the result.
    - For the question which the open-source model answered with "I don't know", had noise-sensitivity 0, even though it could have made correct claims based on the retrieved context, which also skews the result.
    - As these are multi-hop questions, dense retrieval is limited in its ability to provide context sufficient for the claims in the answer to be supported by the ground truth, which has the advantage of being based on relationships between different chunks of the original corpus.
    - For these reasons, I believe this metric is lacking in this current evaluation and no conclusions can be made of it, regardless of the skewed results.
  - In conclusion, gpt-4.1-mini significantly outperforms gpt-oss-20b when answering the multi-hop questions for this limited dataset of 4 (there were no single-hop-specific questions in the dataset). even though this dataset is small, such a big difference in faithfulness and accuracy is still of medium-confidence to make a conclusion because of the small dataset. 


  - Cost and latency breakdown:
    - A general caveat: I used Groq to call the open source chat model, which on one hand is free, on the other hand is rate limiting not calls per minute but tokens per minute, which is why the last 2 of the 4 calls took ~30 seconds of latency each to produce output. For this reason I will only take the first 2 calls into account when analyzing latency, and take the pricing of serverless Fireworks AI when analyzing cost:
    - PRICING:
    "gpt-4.1-mini":       {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "openai/gpt-oss-20b": {"input": 0.07, "cached_input": 0.04, "output": 0.30},
    - Cost: (numbers taken from langsmith tracing, only tokens usage is shown for the open-source so the cost was calculated using the table above)
      - gpt-4.1-mini: Average tokens usage: ~3.6K   Average cost: $0.00175
      - gpt-oss-20b: Average token usage: ~ 3.75k    Average cost: $0.00037  
    - Latency:
      - gpt-4.1-mini: Average latency: 4.6s
      - gpt-oss-20b: Average latency: 1.25s 

    - Conclusion: on average, openAI model costs over 4 times as much to generate output, while taking almost 4 times as much latency. 

    The tradeoff here is obvious, higher quality output for higher cost and latency. For an application heavy on Rag retrieval with multi-hop questions, I think the lower price isn't worth the final result.

## Advanced Activity: Local Models

Swap out the Fireworks AI endpoints for **locally-running open-source models** using [Ollama](https://ollama.com/) or another local inference server of your choice. Run both your embedding model and your chat model locally, and rebuild the RAG pipeline on top of them.

- Compare quality and latency between the local setup and your Fireworks AI hosted endpoint.
- Reflect: what are the trade-offs of local models vs. managed endpoints in a production setting?

Include your findings and a demo in your Loom video.

