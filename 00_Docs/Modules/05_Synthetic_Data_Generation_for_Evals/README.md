# Session 5: 🧪 Synthetic Data Generation for Evals

🎯 Understand how to generate test data automatically to test agentic RAG applications when you don’t have any eval datasets.

📚 **Learning Outcomes**
- Learn to generate high-quality synthetic test data for AI applications using LLMs
- Understand the knowledge graph approach used to generate data
- Understand the process of metrics-driven development
- Learn to load datasets into LangSmith when generated elsewhere

🧰 **New Tools**
- Testset Generation: [RAGAS](https://docs.ragas.io/en/stable/getstarted/rag_testset_generation/) 
- Evaluation: [RAG ASessment](https://docs.ragas.io/en/stable)[, LangSmith Evaluations](https://docs.langchain.com/langsmith/evaluation)

## 📛 Required Tooling & Account Setup
No new tools or accounts required for this session
   
## 📜 Recommended Reading
- [Synthetic Test Data generation](https://docs.ragas.io/en/v0.1.21/concepts/testset_generation.html) (Ragas docs)
- [Mastering LLM Techniques: Evaluation](https://developer.nvidia.com/blog/mastering-llm-techniques-evaluation/) (Jan 2025)

# 🎯 Overview

There are many ways that people can (and do) use Synthetic Data Generation (SDG). When we talk about evaluation techniques, we can think of SDG as a tool in our toolkit that helps us solve `the data scarcity` problem.

The key concepts we need to know are 1) how to go beyond vibe checking with systematic, metrics-driven development, and 2) how the data that we use for RAG applications can be leveraged to extract key features about it, creating a framework that we can use to generate synthetic test data. We also must keep in mind the drawbacks of using this kind of synthetic approach in a vacuum.

The key code we need to learn is the stuff we need to leverage SDG out of the box directly in an "input: raw unstructured data, output: question, context, answer test samples" way.

# ⚗️ Synthetic Data Generation for Evals

As a follow-up to a "vibe check" of your application, creating a "golden" test data set from scratch, informed by real usage of your application and that is perfectly aligned with what users are likely to ask in production, is the obvious next step.

Of course, this is hard and tedious, time-consuming, and sometimes can feel completely intractable.

So as developers, we end up in a situation where we have a scarcity of data samples that we can use to systematically evaluate our applications with useful data and insightful metrics.

If we could solve this problem, we could use this data and metrics calculated from it to drivethe  development of our application towards these key metrics.

This idea of Metrics Driven Development is all about creating an incentive structure for your application.

We might think of it as a way to _train our RAG and agent applications_. If loss or reward functions come to mind, you've got the right idea. We want to create objective measures that we can use to improve our applications.

In the case of AI Engineers leveraging SDG, we always want to ask ourselves a directional question:

<aside>
🤔

How can I use SDG to make my application better?

</aside>

Typically, when we refer to “better” when building AI applications, we’re often focused on either cost or performance.  With this in mind, we can think of a number of aims for SDG; namely:

- 🤓 Domain-Specific Data
    - Domain-Adapted Pretraining or Continued Pretraining
    - Domain-Adapted Language Models (e.g., fine-tuning an LLM chat model)
    - Domain-Adapted Retrieval (e.g., fine-tuning an embedding model)
    
    <aside>
    ❗
    
    We can use SDG to generate task-specific or language-specific data just as you can with domains!  We will discuss this more when we cover fine-tuning.
    
    </aside>
    
- ↔️ Alignment Data
    - Reinforcement Learning with Human Feedback (RLHF)
    - Reinforcement Learning with AI Feedback (RLAIF)
    - Direct Preference Optimization (DPO)
    - **Note that one can *align* models towards many targets!
- 📊 Evaluation Data
    - e.g., is my application “good” at what I want it to be good at?
    - For prompted LLM applications
    - For RAG applications
    - For agentic applications
    - … and more!

Welcome to the beginning of our investigation into how to best evaluate and assess RAG and agent applications beyond using the simple prompting techniques we’ve already covered!

# 🔍 Monitoring RAG Apps in Prod

When it comes to deploying RAG applications to production, there are some key aspects of our applications that we want to be tracking, including [[Ref](https://docs.ragas.io/en/latest/getstarted/monitoring.html)]:

1. **Hallucinations**:  Are answers based on factual reference material?  In the RAG ASsessment (RAGAS) framework, this is called `Faithfulness`
2. **Quality of Retrieval**: We can identify and define “poor context” or “bad” retrievals.
3. **Quality of Response**: We can identify and define “evasive, harmful, or toxic (a.k.a. bad)” responses.
4. **Quality of Format**: We can detect and quantify responses with incorrect formatting.

In short, we’re looking for fact-based, high-quality retrievals and responses with the correct formatting!  The first step towards getting there is SDG.  The second step, as we’ll cover in the next module, is understanding how to compute the most relevant metrics!

# 🛡️ Synthetic Data for Robustness and Adversarial Prompts

Once we understand synthetic data as a way to generate evaluation samples, we can use it for more than happy-path QA. We can also generate stress tests that ask:

> What happens when the user, the retrieved context, or an external tool result tries to push the system away from the behavior we want?

For agentic RAG applications, adversarial behavior can show up in a few places:

1. **Direct user prompts**: jailbreaks, prompt injections, unsafe requests, or requests that try to override the system prompt.
2. **Retrieved context**: poisoned documents, conflicting evidence, hidden instructions inside retrieved text, or misinformation that looks semantically relevant.
3. **Tool and agent workflows**: tool outputs that contain malicious instructions, orchestrator attacks, or multi-step tasks where the unsafe behavior only appears after several turns.

Synthetic data generation lets us deliberately manufacture these cases before users find them for us. A useful adversarial eval sample might include:

- A normal user query
- The expected answer or expected refusal behavior
- Clean reference context
- An adversarial context variant
- A label for the threat type
- A metric we care about, such as faithfulness, attack success rate, refusal correctness, retrieval quality, tool-call correctness, or task-performance retention

This matters because robust systems need to preserve useful behavior while resisting bad instructions. A system that refuses everything is not useful. A system that answers everything is not safe. The goal is to measure both:

1. **Task performance**: Does the application still answer legitimate questions well?
2. **Attack resistance**: Does the application avoid following malicious or misleading instructions?

The Ragas synthetic test data generation docs frame synthetic data as a way to reduce manual test-data creation and systematically create diverse question types, including reasoning, conditioning, and multi-context samples [[Ref](https://docs.ragas.io/en/v0.1.21/concepts/testset_generation.html)]. For modern RAG and agentic systems, we can extend that mindset into adversarial coverage: generate not only diverse questions, but diverse failure modes.

There is an important caveat: synthetic evals are not automatically representative of production. Recent work on synthetic RAG benchmarks suggests they can provide useful signal for tuning retrieval parameters when the synthetic task is well aligned with real tasks, but they can be less reliable when comparing broader generator architectures or when synthetic questions do not match real user behavior [[Ref](https://arxiv.org/html/2508.11758v1)]. In practice, we should combine synthetic adversarial evals with real user traces, human review, and production monitoring.

For deeper exploration of adversarial RAG evaluation, see work on RAG poisoning attacks and defensive techniques like skeptical prompting [[Ref](https://arxiv.org/html/2412.16708v2)], as well as larger red-team resources for agentic RAG systems such as the [Agentic RAG Red Teaming Bench](https://huggingface.co/datasets/Fujitsu/agentic-rag-redteam-bench).

# 〽️ Knowledge Graph Approach

The latest for test set generation **leverages a knowledge graph** in favor of the previous evolutionary approach (now deprecated, described below) based on the [WizardLM paper](https://arxiv.org/abs/2304.12244) that outlines the Evol-Instruct process!

So what is this about knowledge graphs for test set generation?

Instead of using different in-depth or in-breadth evolutionary techniques, we use knowledge graph transformations, also called [transforms](https://docs.ragas.io/en/latest/references/transforms/), to create rich starting data.  Then we build [scenarios](https://docs.ragas.io/en/latest/references/testset_schema/#ragas.testset.synthesizers.base.BaseScenario), also called schemas, based on the data.

Why do we need this more robust knowledge graph approach?

It’s simple: the previous method worked very well for *single-hop queries* that only need to retrieve data from one source.  In other words, evolutionary approach was sufficient for generating test data for RAG applications.

This new method is designed to work well with both *single-hop* and *multi-hop* queries.  Multi-hop queries involves multiple lookups across different sources or tables.

Think about it like this: a classic single-hop query might use a SQL database, whereas a classic multi-hop query might use a graph database.

In LLM terms, synthetic test data based on a single-hop query is good for assessing simple RAG applications, whereas multi-hop queries what you need to assess agentic applications with access to multiple tools.

Additionally, it’s worth noting that this new abstraction also signals a deeper structuring of the synthetic test data.  Now we can use transforms and schemas, rather than simple prompts to evolve data.

# 🧬 An Evolutionary Approach [Optional]

An approach that used to be favored over the Knowledge Graph approach is an evolutionary one, and it's worth considering due to its simplicity. For Question-Answer pairs, we _evolve_ data that increases the complexity and diversity of the initial data set.  After all, it is the case that:

1. The same question can be asked in different ways
2. The same answer can correspond to different questions

These two simple facts guide our path to evolution.  In the end, all we need are some relevant prompts!

Fundamentally, there are two dimensions that we will consider: `breadth` and `depth`.  That is, we can **evolve** our data in the direction of increasing depth or increasing breadth.

- 💬 In-Depth Evolver Prompt
    
    ```
    I want you act as a Prompt Rewriter.
    Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems
    (e.g., ChatGPT and GPT4) a bit harder to handle.
    But the rewritten prompt must be reasonable and must be understood and responded by humans.
    Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please
    do not omit the input in #Given Prompt#.
    You SHOULD complicate the given prompt using the following method:
    Please add one more constraints/requirements into #Given Prompt#
    You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only
    add 10 to 20 words into #Given Prompt#.
    ‘#Given Prompt#’, ‘#Rewritten Prompt#’, ‘given prompt’ and ‘rewritten prompt’ are not allowed to appear in
    #Rewritten Prompt#
    #Given Prompt#:
    <Here is instruction.>
    #Rewritten Prompt#:
    ```
    
- 💬 In-Breadth Evolver Prompt
    
    ```
    I want you act as a Prompt Creator.
    Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.
    This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.
    The LENGTH and difficulty level of the #Created Prompt# should be similar to that of the #Given Prompt#.
    The #Created Prompt# must be reasonable and must be understood and responded by humans.
    ‘#Given Prompt#’, ‘#Created Prompt#’, ‘given prompt’ and ‘created prompt’ are not allowed to appear in
    #Created Prompt#.
    #Given Prompt#:
    <Here is instruction.>
    #Created Prompt#:
    ```
    
    ```python
    I want you act as a Prompt Creator.
    Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.
    This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.
    The LENGTH and difficulty level of the #Created Prompt# should be similar to that of the #Given Prompt#.
    The #Created Prompt# must be reasonable and must be understood and responded by humans.
    ‘#Given Prompt#’, ‘#Created Prompt#’, ‘given prompt’ and ‘created prompt’ are not allowed to appear in
    #Created Prompt#.
    #Given Prompt#:
    <Here is instruction.>
    #Created Prompt#:
    ```

As you can see, the idea of simply evolving prompts is quite powerful. This idea of using the LLM to help us to generate data that is then fed into the LLM, potentially for evaluation, dovetails beautifully into the idea that I'm sure you've heard of called "LLM as a Judge," which we'll dive deep into in the next module!

