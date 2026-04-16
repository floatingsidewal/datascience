# Week 0: AI & LLMs — A Working Survey

## A Standalone 2-3 Hour Tour for Anyone Deciding How Deep to Go

**Time commitment:** 2:30–3:00 end-to-end. Segments are self-contained — stopping early is fine.

**Who this is for:** Managers, PMs, support leaders, ICs — anyone who has watched AI hype go by for two years and wants a real, honest mental model before they decide whether to invest time going deeper. You do not need to be technical. You do not need to have written code recently. You need to be curious.

**What you walk away with (even if you stop here):**
- A working mental model of how ChatGPT-style systems actually work, end to end.
- Enough vocabulary (tokens, context window, RAG, LoRA, evals, endpoints) to read an AI-related headline, spec, or vendor pitch without bouncing off.
- A clear-eyed answer to "what should I do next?" — which might be taking a deeper course, or might be "you're fine, keep reading newsletters."

**What this is NOT:**
- A sales pitch for any of the other courses in this repo. Segment 6 will genuinely route you away from them if that's the right answer.
- A deep technical treatment. This is knee-deep: named concepts, one diagram per segment, no math.
- A prerequisite for anything. It's a front door, not a gate.

---

## How to Use This Week

This curriculum has two tracks through the same material:

**Manager track (~90 min):** Read each segment's "Why you care" and "Reading" sections. Skim the vignettes. Skip the hands-on labs. Do Segment 6 at the end. You'll still get the mental model.

**IC track (~2:30-3:00):** Do everything. The hands-on labs are where the concepts lock in. The Segment 5 workshop in particular is where "what is training?" stops being hand-wavy.

**Course path through the week:**

| # | Segment | Time | Manager | IC |
|---|---|---:|:---:|:---:|
| 1 | What is an LLM and how does it work? | 25 min | Read | Read + lab |
| 2 | Harnesses: chat, agents, coding agents | 25 min | Read | Read + lab |
| 3 | Model hosting endpoints | 20 min | Read | Read (lab optional) |
| 4 | How do you test models? | 25 min | Read | Read + lab |
| 5 | Workshop: what is training? | 50-60 min | Watch demo | Full follow-along |
| 6 | Is this for you? Decision rubric | 10 min | Do it | Do it |

---

# Segment 1 — What Is an LLM and How Does It Work?

**Why you care:** Every product with "AI" on the label in 2026 has one of these in the middle. If you don't have a mental model for what an LLM is actually doing, every pitch, every failure, and every headline sounds like magic. Magic is where vendors sell to you at 10x.

## Reading (15 min)

An LLM (Large Language Model) is, in plain terms, **a very elaborate autocomplete trained on mountains of text.**

That's not a reductive joke. That's the actual mechanism.

Here's what happens every time you type into ChatGPT or Claude:

```
    Your message
         │
         ▼
   ┌───────────────┐
   │  Tokenizer    │   splits text into ~3-4 character chunks ("tokens")
   └───────────────┘
         │
         ▼
   ┌───────────────┐
   │  The Model    │   looks at all the tokens so far,
   │  (billions    │   produces a probability score for
   │   of weights) │   every possible next token
   └───────────────┘
         │
         ▼
   ┌───────────────┐
   │  Sampler      │   picks one next token (with some randomness
   └───────────────┘    controlled by "temperature")
         │
         ▼
    One more token added to the output
         │
         └─── loop back, predict the next one, until done
```

That's it. That's the whole trick. The model does not "think" about what you said. It predicts, token by token, what word should come next given everything it has seen so far.

The reason it feels like talking to a person is that "what word comes next in a helpful response to this question" is a pretty good proxy for "what a helpful person would say." When the proxy breaks, you get **hallucinations** — the model generates a confident-sounding sequence that happens to be wrong, because plausibility is what it optimizes for, not truth.

**Five terms you now own:**

- **Token** — a ~3-4 character chunk of text. "hello" is one token. "tokenization" is usually two. Models have a vocabulary of ~50,000–200,000 tokens.
- **Context window** — how many tokens the model can "see" at once (your prompt + its reply + any documents you paste). Modern frontier models are 200K–2M tokens. When you exceed it, older content falls off the edge.
- **Training vs. inference** — **training** is the (expensive, one-time, months-long) process of baking the weights. **Inference** is what happens every time you send it a message. These are completely different costs and failure modes.
- **Temperature** — a number between 0 and ~2 that controls how random the sampler is. 0 = always pick the most likely next token (deterministic, boring). 1 = natural variety. 2 = unhinged. Most chat products default around 0.7.
- **Hallucination** — a confident, fluent, wrong answer. Not a bug in the model. A direct consequence of "predict the next plausible token" being the goal.

## Vignette: the confidently-wrong feature lookup

A support engineer asks ChatGPT: "Does our product support SAML login?"

ChatGPT replies, with full confidence: "Yes, SAML 2.0 login is supported as of version 4.2, configurable under Admin → Security → SSO."

The engineer checks. There is no SAML support. Version 4.2 doesn't exist yet. There is no Admin → Security → SSO menu.

What happened:

1. The model has no idea what the engineer's product is. It has never seen the docs.
2. It has seen a *lot* of SaaS documentation that follows this exact pattern.
3. "The most plausible next tokens after that question" produced a perfectly reasonable-sounding SaaS documentation snippet — about a product that doesn't exist.

This is the base failure mode. The fix is **grounding** (feeding the model the actual docs — "RAG," which we'll see in Segment 2/3), not "a smarter model."

## Hands-on lab (10 min, hosted, no install)

Open ChatGPT, Claude.ai, or Copilot Chat — whichever you use. Run these three prompts and observe:

1. **Factual, common:** "What is the boiling point of water at sea level in Celsius?" → You should get "100°C" fast and correctly.
2. **Summarization of text you paste:** Paste any 3-paragraph article and ask "summarize this in two sentences." → Usually excellent, because the source material is right there in the context window.
3. **Outside its knowledge:** "What did [your company or team name] ship last week?" → Watch it hallucinate confidently. It has never heard of your company, or its training data is stale.

The point of the exercise: the model's *confidence* is nearly identical in all three cases. Only your external knowledge tells you which answers to trust. **This is the single most important thing to internalize about LLMs.**

## Deliverable

Write, in your own words, a one-sentence definition of an LLM. Then write one sentence about what you observed in the three-prompt exercise.

---

# Segment 2 — Harnesses: Chat, Agents, Coding Agents

**Why you care:** "ChatGPT," "Claude.ai," "Copilot agent," "deep research," "Cursor," "Claude Code," "automated AI SDR" — these are all different **harnesses** wrapped around the same kind of LLM. The harness decides more about what the product can do than the underlying model does. If you don't know what harness you're looking at, you can't tell why it succeeds or fails.

## Reading (15 min)

An LLM, by itself, is a **function:** text in → text out. That's all. It has no memory of your last conversation. It cannot open a web page. It cannot run code. It cannot read your files. It can only take text in, and return text out.

Every AI product you've ever used is an LLM with code wrapped around it — a **harness** — that turns that single function into something useful. The four families of harness, in order of complexity:

### 1. Chat UI

ChatGPT, Claude.ai, Gemini, Copilot Chat (the basic mode).

- Stores your conversation history so the model sees past turns.
- No tool use. No internet. No code execution. Just text in, text out.
- Great at: reasoning about text you paste in, brainstorming, writing, explaining.
- Bad at: anything requiring current information or acting on the world.

### 2. Chat + Retrieval (RAG)

Same as chat UI, but with one added step: before calling the model, a search is done (over your company's docs, or the web), and relevant snippets are pasted into the prompt.

- "ChatGPT with browsing," "Claude with web search," "Copilot grounded on SharePoint," almost every enterprise chatbot.
- The model answers using the retrieved text instead of its memory.
- This is the fix for the "confident hallucination" problem in Segment 1.
- **RAG** = Retrieval-Augmented Generation. You will hear this acronym a lot. It is not complicated.

### 3. Agent

An agent is an LLM in a **loop**, given a **goal** and a set of **tools.**

```
           ┌──────────────────────────────┐
           │                              │
           ▼                              │
   ┌───────────────┐                      │
   │ Prompt        │                      │
   │ (goal + prior │                      │
   │  steps + tool │                      │
   │  results)     │                      │
   └───────────────┘                      │
           │                              │
           ▼                              │
   ┌───────────────┐                      │
   │ Model decides │                      │
   │ next action   │                      │
   └───────────────┘                      │
           │                              │
           ▼                              │
   "Call tool X"  ──────►  execute tool, get result
           │                              │
           │                              │
           └──────────────────────────────┘
                    (loop until model says "done")
```

Tools can be: web search, code execution, file read/write, calendar, email, database queries, whatever the harness gives it.

Examples: ChatGPT's "deep research," Claude's computer use, an AI SDR that emails leads, an autonomous bug-fixer.

- Can do multi-step tasks that chat cannot (go research this topic across 20 pages and write a report).
- Cost and latency are higher — an agent run is often dozens of model calls, not one.
- Harder to debug: when it fails, it might have failed on step 7 of 12, and you have to read the whole trace.

### 4. Coding agent

A specialized agent whose tools are a shell, a filesystem, a code editor, and often git.

- Claude Code, Cursor agent, Copilot agent mode, Windsurf, Devin.
- The goal is usually "implement this," "fix this bug," "refactor this." The agent reads code, writes code, runs tests, iterates.
- This is the fastest-moving category in 2026.

### The key insight

**The model is only ever called with text in and text out. Everything interesting is the harness around it.**

When a product fails, ask: was it the model (wrong answer), or the harness (wrong tool, wrong retrieval, wrong loop)? The answer is usually the harness.

## Vignette: "our chatbot should be an agent"

A VP says in a meeting: "Our support chatbot is weak. It should be an agent."

What they mean: "It should do more than answer — it should fix things."

What "agent" actually buys you:
- Tool calling (the model can trigger a refund, reset a password, open a ticket).
- Multi-step loops (it can try something, see the result, try something else).

What it costs you:
- Every tool call is a point where something can go wrong with real-world impact (a wrongly-triggered refund is worse than a wrong answer).
- Unbounded loops can burn tokens and money quickly.
- The evaluation story gets much harder — you're no longer grading answers, you're grading *sequences of actions.*

Going from chatbot-with-RAG to agent is a 10x jump in system complexity, not a setting. The right question isn't "should this be an agent?" It's "what specific action can we safely let it take, and how will we know when it goes wrong?"

## Hands-on lab (10 min, hosted)

**Exercise A — chat without tools:** In a plain chat (ChatGPT GPT-4, Claude.ai without search enabled, whatever you've got), ask: "What is the current price of a basic OpenAI API plan?" Note the answer.

**Exercise B — chat with tools:** Turn on web search (ChatGPT: "search the web" toggle; Claude.ai: web search setting; Copilot: web grounding). Ask the exact same question. Note the answer.

Same model. Different harness. One knows what day it is, the other lives in its training cutoff.

**Bonus (IC only):** If you use a coding agent (Claude Code, Cursor agent mode), notice the next time it works on your codebase: it's running the exact loop in the diagram above, with your repo as its filesystem.

## Deliverable

One sentence distinguishing a chatbot from an agent. Then name one AI product you use every week and identify which harness it is (chat, chat+RAG, agent, coding agent, or plain embedded API).

---

# Segment 3 — Model Hosting Endpoints: Foundry vs. Public APIs

**Why you care:** When a VP at your company says "we're moving to Azure AI Foundry," that is a **compliance and contracts** decision, not a model quality decision. If you don't know what endpoints are, you'll think they're picking a worse model. They're picking a different data path.

## Reading (15 min)

The same model weights — GPT-5, Claude 4.x, Llama 4, Gemma 4 — can be served from multiple places with very different contracts. The three categories:

### 1. Public / Frontier Provider API

- Endpoints: `api.openai.com`, `api.anthropic.com`, `generativelanguage.googleapis.com`.
- You get the newest models on day one.
- Data flows through the vendor. Some vendors offer "zero data retention" contracts, but your raw requests land on their infrastructure.
- Fastest path from idea to shipped app — one API key, go.
- Common for: startups, research, prototyping, small teams.

### 2. Enterprise-Hosted / Cloud Provider

- Endpoints: Azure AI Foundry (hosts OpenAI + Meta + others), AWS Bedrock (hosts Anthropic + Meta + Amazon + others), Google Vertex AI (hosts Gemini + open models).
- **Same model weights** — Azure's GPT-5 is GPT-5. But the request goes to your cloud tenant, bound by your company's existing cloud contract, data residency rules, and compliance posture.
- Usually 1-4 weeks behind the frontier provider on new model releases.
- Often required for enterprise use — if your company already trusts Azure with all its data, putting AI inside the same trust boundary is a one-signature decision. Putting it inside OpenAI's boundary is a multi-month legal review.
- Common for: Fortune 500, regulated industries, anything with real customer data.

### 3. Self-Hosted

- You run the GPUs. You run the inference server (vLLM, TGI, Ollama, SGLang). You download open-weights models (Llama 4, Qwen 3, Gemma 4, DeepSeek).
- Full control. No per-token cost. All the operational cost (GPUs, engineers, incidents).
- Smaller model ceiling — no frontier model is open-weights.
- Common for: ultra-sensitive data, high-volume use cases, regulated environments that can't send data anywhere.

### The table

| Factor | Public API | Enterprise-hosted | Self-hosted |
|---|---|---|---|
| Newest models | Day 0 | 1-4 weeks behind | Rarely frontier |
| Per-token cost | Lowest | Similar or slight markup | $0 marginal, $$$ fixed |
| Data boundary | Vendor's | Your cloud tenant | Your datacenter |
| Compliance lift | High (new contract) | Low (existing contract) | Medium (new ops) |
| Ops burden | Zero | Light | Heavy |
| Latency | Vendor network | Your cloud region | Your network |
| Common users | Startups, research | Enterprise | Regulated, high-volume |

### The key insight

**The endpoint decision drives 80% of an enterprise AI project.** It dictates which models you can use, how fast you can iterate, what your cost curve looks like, and who has to sign the contract. "We're moving to Foundry" usually means "legal approved Azure, not OpenAI." It is not a slight on the model.

A corollary: the gap between "what's possible on the frontier API" and "what your enterprise can actually use in production" is often the dominant story in any AI roadmap. Close readers of this gap win.

## Vignette: the switch that wasn't about capability

A product team demos a feature built on `api.anthropic.com`. It works great. They push to production and get stopped by security: "data can't leave our cloud."

Three months of negotiation later, they rebuild on AWS Bedrock's Claude endpoint. Same model, same prompts, same code (mostly). The demo-to-production gap was entirely legal.

Nobody was wrong. This is the normal shape of enterprise AI delivery.

## Optional hands-on (IC only, skip if you don't have creds)

Run the same prompt against two endpoints and compare. If you have an OpenAI key and an Azure Foundry endpoint:

```bash
# Public OpenAI
curl https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Name three colors."}]
  }'

# Azure Foundry (same underlying model family)
curl "https://$AZURE_ENDPOINT/openai/deployments/$DEPLOYMENT/chat/completions?api-version=2024-08-01-preview" \
  -H "api-key: $AZURE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Name three colors."}]
  }'
```

Near-identical output. Different hostname, different contract, different bill.

If you don't have enterprise creds, that's fine — reading the shape above is enough.

## Deliverable

Match each real-world statement to the endpoint strategy that solves it:

1. "We can't ship until legal approves the data path."
2. "We need the brand-new model on its release day."
3. "We process 50M prompts per day; API bills are eating our margin."
4. "Our data is HIPAA-regulated and cannot leave our VPC."

*(Answers: 1 → enterprise-hosted, 2 → public API, 3 → self-hosted or negotiated enterprise pricing, 4 → self-hosted or enterprise-hosted in your VPC.)*

---

# Segment 4 — How Do You Test Models? Benchmarks, RLI, LLM-as-Judge

**Why you care:** "This model scored 92% on MMLU" is a sentence you will hear in vendor pitches for the rest of your career. You need to know what it actually means, and — more importantly — when it is and isn't the right question.

## Reading (15 min)

"Is this model good?" has three different answers depending on what you mean. Pick the wrong one and you will deploy a model that scores great on paper and fails in production.

### 1. Academic benchmarks

Standardized, published tests. Examples:

- **MMLU** — Massive Multitask Language Understanding. 57 subjects of multiple-choice questions, high-school to professional level. Tests trivia + reasoning.
- **GSM8K / MATH** — grade-school and harder math word problems.
- **HumanEval / SWE-bench** — code generation. HumanEval is small functions; SWE-bench is real GitHub issues.
- **GPQA** — graduate-level physics, chem, bio. Hard.
- **ARC-AGI** — pattern abstraction. Hard for models.

Strengths: comparable across models and providers, reproducible, well-known. Good for "is model B better than model A across the board?"

Weaknesses: models are trained on (or adjacent to) these benchmarks. Scores inflate. Real tasks look nothing like multiple choice. **A high MMLU score tells you almost nothing about whether the model will help your support team.**

### 2. Task-based / operational evals

Custom test sets built from *your* actual work.

- **Remote Labor Index (RLI)** — evaluates models on real remote-work tasks sampled from freelance platforms. "Can you actually do the job?"
- **SWE-bench Verified** — real bug fixes that real humans made, tested by running the original test suite.
- **Your own eval set** — 100-500 examples of the exact task you want to automate, each with a known-good answer or a rubric.

Strengths: directly measures the thing you care about. Resistant to benchmark contamination. Predicts production behavior.

Weaknesses: expensive to build. Requires taste about what "good" looks like for your task.

**This is the category that matters for real deployment decisions.** If you take one thing from this segment: when a vendor quotes a benchmark, ask them "what's the task-based number?"

### 3. LLM-as-Judge

Use a strong model (GPT-5, Claude 4.x, Gemini 2.x) to grade another model's output against a rubric.

```
 Model A answer  ──┐
                   │
 Model B answer  ──┼──►  Judge model  ──►  "A is better because..."
                   │
 Rubric         ──┘
```

Strengths: cheap and scalable. You can run 10,000 comparisons overnight. Correlates reasonably well with human judgment on well-defined tasks.

Weaknesses: the judge has biases — it may prefer verbose answers, its own outputs, or answers in its house style. You have to validate the judge against human labels before trusting it.

### 4. Human eval + production A/B

The ground truth. Real users, real tasks, real measurement (CSAT, resolution time, conversion).

Strengths: the only measurement that actually matters for business outcomes.

Weaknesses: slow, expensive, requires production traffic, requires instrumentation.

### The hierarchy

A reasonable evaluation stack, from cheapest to most expensive:

```
         Academic benchmarks    (free, fast, indicative)
                 │
                 ▼
         LLM-as-Judge on your data    (cheap, fast, directional)
                 │
                 ▼
         Task-based eval set    (meaningful, modest cost)
                 │
                 ▼
         Human eval + A/B in production    (truth, slow, expensive)
```

Smart teams do all four. Bad teams only do the first.

## Vignette: MMLU deployment failure

A company picks its support chatbot model based on MMLU scores. Model X leads the leaderboard. They ship it.

Two weeks in, CSAT is down. Tickets that used to auto-resolve are now escalating. What happened?

MMLU rewards knowing trivia. Support deflection rewards:
- Knowing when to say "I don't know, let me escalate."
- Matching the user's tone.
- Resolving common issues in one turn, not giving a Wikipedia article.
- Not hallucinating product features.

None of those are on MMLU. The model they picked is a good trivia-answerer and a bad support agent. A task-based eval of 200 past tickets would have caught this before launch.

## Hands-on lab (10 min, hosted)

Open [Chatbot Arena](https://lmarena.ai) or the [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard). Pick two models that are close on the leaderboard. Now open the [Remote Labor Index](https://rli.scale.com) or skim SWE-bench Verified results for those same models.

Notice: the rankings often disagree. A model that tops MMLU may rank mid-pack on operational tasks. This is the benchmark-vs-reality gap made visible.

## Deliverable

Write the three questions you'll ask the next vendor who pitches an AI product at you. Suggested starters:
- "What task-based eval set did you test this on, and can I see a sample?"
- "How does this perform on tasks that look like mine specifically?"
- "Where does this model fail, and what's your plan for those cases?"

**Cross-link:** If this segment lit you up, **Week 08-Alt** of the Data Science for Fusion Engineers course (`../Week08-Alt/README.md`) is the deep dive on RLI, LLM-as-Judge, and simulated-persona chatbot testing.

---

# Segment 5 — Workshop: What Is Training?

**Bring your laptop.** This is the centerpiece of the week.

**Why you care:** "We need to fine-tune our own model" is the sentence that launches a thousand miserable projects. Before you agree, disagree, or fund one, you need to have actually *done* a tiny fine-tune and seen what it changes and what it doesn't. 45 minutes here saves months elsewhere.

## The three stages (5 min reading)

Every modern chat model goes through three stages. You need the names:

```
 Stage 1: PRETRAINING         → Base Model
    (trillions of tokens,        (a text completer;
     next-token prediction)       doesn't know "questions")

          ↓

 Stage 2: SUPERVISED           → Instruction-Tuned Model
          FINE-TUNING (SFT)       (helpful assistant)
    (10K-100K instruction pairs)

          ↓

 Stage 3: PREFERENCE TUNING   → Polished Chat Model
    (RLHF or DPO on ranked pairs)  (what you ship)
```

- **Pretraining** is a ~$100M, months-long, trillions-of-tokens job done by frontier labs. You are not doing this in this workshop. You are also not likely to do this at your company — it almost never makes business sense.
- **Supervised Fine-Tuning (SFT)** is what you *can* do. You take an existing model and train it further on your own curated examples. This is what the workshop does.
- **Preference tuning (RLHF / DPO)** adds tone, safety, and polish on top of SFT. This is what makes ChatGPT feel like ChatGPT.

A deeper treatment of these three stages exists in `../How-Tuning-Works/Week00/README.md`. For now, the one-paragraph mental model above is enough.

## Workshop setup (5 min)

**Recommended path:** Google Colab. Free GPU tier (T4), no install, works on any laptop.

**Two ways to run it:**

- **Fast path (recommended):** open [`workshop.ipynb`](workshop.ipynb) in Colab (`File → Open notebook → GitHub` and paste this repo URL, or upload the file). Set `Runtime → Change runtime type → T4 GPU` and run cells top to bottom. Everything below is baked in.
- **Copy-paste path:** follow the code blocks in this README into a blank Colab notebook. Same content, useful if you want to read the narrative inline and type things yourself.

Either way:

1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Open the notebook or create a new one.
3. Runtime → Change runtime type → T4 GPU → Save.
4. Run cells in order.

**Alternative paths (if Colab is blocked at your org):**
- **Apple Silicon Mac:** `pip install mlx-lm` and follow their LoRA docs — roughly twice as slow but fully local.
- **No GPU available:** you can still run inference (skip the training step) on a tiny model in CPU Colab. The conceptual lesson survives; the "before/after" demo doesn't.

## Step 1 — Install and load a small base model (2 min)

```python
# In Colab, run this first
!pip install -q transformers peft trl datasets accelerate bitsandbytes

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # ~1 GB, fits on free T4
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

Qwen 2.5 0.5B is a real instruction-tuned model from Alibaba. Small enough to fine-tune in a few minutes, big enough to show real before/after behavior.

## Step 2 — Inference BEFORE fine-tuning (5 min)

```python
def chat(prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

# Three support-ticket-style prompts
prompts = [
    "Customer says: 'my account is locked after too many login attempts.' Categorize this ticket in one word.",
    "Customer says: 'the checkout button is greyed out on the mobile app.' Categorize this ticket in one word.",
    "Customer says: 'I was charged twice for the same subscription.' Categorize this ticket in one word.",
]

for p in prompts:
    print("Q:", p)
    print("A:", chat(p))
    print("---")
```

Note the answers. They're fine, maybe verbose, maybe inconsistent about formatting ("Authentication" vs. "account access" vs. "The category is: billing").

**Save these outputs.** We're comparing against them after training.

## Step 3 — Build a tiny instruction dataset (3 min)

Normally you'd have thousands of examples. For the workshop, ~30 is enough to see the shape change.

```python
from datasets import Dataset

TRAINING_EXAMPLES = [
    {"user": "Customer says: 'I can't log in, keeps saying wrong password.' Categorize this ticket in one word.", "assistant": "auth"},
    {"user": "Customer says: 'forgot my password and reset link isn't coming.' Categorize this ticket in one word.", "assistant": "auth"},
    {"user": "Customer says: 'two-factor code never arrived.' Categorize this ticket in one word.", "assistant": "auth"},
    {"user": "Customer says: 'the dashboard is stuck loading.' Categorize this ticket in one word.", "assistant": "bug"},
    {"user": "Customer says: 'app crashes when I click settings.' Categorize this ticket in one word.", "assistant": "bug"},
    {"user": "Customer says: 'report export produces empty file.' Categorize this ticket in one word.", "assistant": "bug"},
    {"user": "Customer says: 'I was charged twice for last month.' Categorize this ticket in one word.", "assistant": "billing"},
    {"user": "Customer says: 'invoice doesn't match what I signed up for.' Categorize this ticket in one word.", "assistant": "billing"},
    {"user": "Customer says: 'please cancel my subscription and refund me.' Categorize this ticket in one word.", "assistant": "billing"},
    {"user": "Customer says: 'how do I invite a new team member?' Categorize this ticket in one word.", "assistant": "howto"},
    {"user": "Customer says: 'where do I find the API keys?' Categorize this ticket in one word.", "assistant": "howto"},
    {"user": "Customer says: 'what's the difference between the Pro and Team plans?' Categorize this ticket in one word.", "assistant": "howto"},
    # Add more — aim for 30+, roughly balanced across categories
]

def format_example(ex):
    messages = [
        {"role": "user", "content": ex["user"]},
        {"role": "assistant", "content": ex["assistant"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = Dataset.from_list(TRAINING_EXAMPLES).map(format_example)
print(dataset[0]["text"])
```

Notice the shape: every example enforces the same pattern — one-word answer from a small vocabulary (`auth`, `bug`, `billing`, `howto`). **This is the shape we're teaching the model to produce.**

## Step 4 — Run a LoRA fine-tune (8-12 min)

LoRA (Low-Rank Adaptation) is the cheap kind of fine-tuning. Instead of updating all billion+ weights in the model, you add a tiny set of "adapter" weights (~1% of the total) and only train those. Same effect for most purposes, 100x cheaper.

```python
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # you'll see ~1% trainable

training_args = SFTConfig(
    output_dir="./lora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="no",
    bf16=True,
    max_seq_length=512,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

Watch the loss number go down in the logs. That's the model getting better at producing the shape you trained it on.

## Step 5 — Inference AFTER fine-tuning (5 min)

Run the exact same three prompts from Step 2:

```python
for p in prompts:
    print("Q:", p)
    print("A:", chat(p))
    print("---")
```

Compare to your saved pre-training outputs.

**Expected observation:** answers are now short, one-word, from your trained vocabulary. `auth`, `bug`, `billing`, `howto`. Consistent. The shape locked in.

## Step 6 — Probe the limits (5 min)

Now try prompts that are *outside* the training distribution:

```python
extra_prompts = [
    "Customer says: 'the pricing page has a typo.' Categorize this ticket in one word.",
    "Customer says: 'I love your product!' Categorize this ticket in one word.",
    "What is the capital of France?",
]

for p in extra_prompts:
    print("Q:", p)
    print("A:", chat(p))
    print("---")
```

Observations you might see:
- The typo ticket gets forced into one of your four categories — maybe `bug`, maybe `howto`. It wants to produce the shape you trained, even when the shape doesn't fit.
- The compliment may get miscategorized — you didn't train a `feedback` category, so there's no slot for it.
- "What is the capital of France?" — the model may still answer "Paris," OR it may now try to return a one-word category. The fine-tune has narrowed the model's behavior.

## What you just learned

1. **Fine-tuning teaches a shape, not facts.** The model didn't learn what your product is. It learned a consistent response pattern from examples.
2. **The model generalizes within the shape you showed it.** Prompts that look like training data get handled well. Prompts outside it get forced into the wrong shape.
3. **LoRA is cheap.** A 1-3% trainable adapter captured almost all of the behavior change. Full fine-tuning is rarely necessary.
4. **Data is the lever, not compute.** The 30-example change was driven by the *examples*, not the training algorithm. If you want a better fine-tune, you almost always want better data, not more epochs.

## The decision framework (2 min reading)

When someone says "we need to fine-tune," run this before agreeing:

| Problem | Try this first | Fine-tune? |
|---|---|---|
| Model doesn't know our facts | RAG (retrieval) | Rarely |
| Tone doesn't match our brand | System prompt + few-shot | Maybe |
| Output format is wrong | Structured outputs / JSON mode | Rarely |
| Too expensive at scale | Smaller model, or distill | Yes, via distillation |
| Model refuses reasonable requests | Different base model or provider | Sometimes |
| Need a specialized tiny model | Yes | **Yes, this is the real use case** |

**Heuristic:** if a good system prompt gets you 80% of the way, fine-tuning is rarely worth the operational cost (training pipeline, eval pipeline, ongoing retraining as base models improve).

## Deliverable

One paragraph: **what changed** in the model after training, **what didn't change,** and **one problem you'd NOT solve with fine-tuning.**

**Cross-link:** If this workshop lit you up, the full `../How-Tuning-Works/` course goes deep on tokenization, scaling laws, SFT internals, DPO/RLHF, chat templates, and distillation. That course is built for the audience this workshop is for.

---

# Segment 6 — Is This For You? Decision Rubric

You've now walked the whole stack: what an LLM is, how harnesses wrap it, where models are hosted, how to test them, and what training actually does. This is a real working mental model. It is enough, by itself, to make better decisions.

The honest question now: **should you invest more time going deeper, and if so, where?**

## Five questions

Answer each to yourself. No scoring system, no points. Just pattern-match.

**1. In the next 12 months, do you need to evaluate, deploy, or be accountable for an AI product at work?**
- Yes → you benefit from one of the deeper courses.
- No, just curious → you probably don't need more structured learning. Newsletters will do.

**2. Are you comfortable reading and running Python code for 1-2 hours a week?**
- Yes → any course is fine.
- No → either start with a Python refresher first, or stick to the manager-track readings in this Week 0 and in How-Tuning-Works Week 0.

**3. What kind of problem are you actually trying to solve?**
- "I need to evaluate an AI feature, measure if it's working, spot when it breaks in production" → **Fusion Engineers course.**
- "I need to understand how models are built, when fine-tuning makes sense, and have credible conversations with ML teams" → **How Tuning Works course.**
- "Both, eventually" → start with whichever is closer to next week's work problem.
- "Neither applies to me" → you probably don't need more structured learning.

**4. Did the Segment 5 workshop energize you or exhaust you?**
- Energized → How Tuning Works.
- Exhausted → Fusion Engineers' operational focus is probably a better fit.
- Fine either way → you're flexible; use question 3.

**5. Do you have stats / data background (mean, median, distributions, A/B testing) or are you rusty?**
- Have it → either course works.
- Rusty → Fusion Engineers Week 1-3 will bring you back up. How Tuning Works assumes a bit more comfort with ML vocabulary but less stats.

## Four tracks

Based on your answers, here's where to go next:

### Track A — "You're set. Keep reading newsletters."

You now have the mental model. That's the 80/20. Structured courses are for people who need to operate, build, or evaluate — not for everyone who wants to be informed.

Recommended inputs, roughly 10-30 min/week:
- **[Import AI](https://jack-clark.net/)** — Jack Clark's weekly newsletter. Excellent policy + technical mix.
- **[The Batch](https://www.deeplearning.ai/the-batch/)** — Andrew Ng's newsletter. Balanced, practitioner-focused.
- **[Stratechery](https://stratechery.com/)** — Ben Thompson's business analysis, heavy AI coverage. Paid.
- **[Hacker News](https://news.ycombinator.com/)** — the AI-related comment threads are where the gossip lives.
- **[Simon Willison's blog](https://simonwillison.net/)** — the best single source on what's actually practical week-to-week.

"This isn't for me" is a correct outcome.

### Track B — "Take Data Science for Fusion Engineers" (`../DataScienceForFusionEngineers.md`)

If your job involves evaluating, deploying, monitoring, or championing AI in a support / customer-facing / operational context. 10 weeks, hands-on, starts with data literacy and ends with production monitoring. Week 08-Alt specifically covers real-world evaluation (RLI, LLM-as-Judge) — the Segment 4 deep dive.

Start with `../Week01/README.md`.

### Track C — "Take How Tuning Works" (`../How-Tuning-Works/README.md`)

If you want to understand how models are built at the mechanic level — pretraining, SFT, preference tuning, chat templates, scaling laws, cost and capability tradeoffs. Built for managers, PMs, and engineers who need to hold credible conversations with ML teams and make fine-tune-vs-buy-vs-prompt decisions.

Start with `How-Tuning-Works/Week00/README.md`.

### Track D — "Do something else first"

If the programming or stats gap feels real, that's useful information. Don't try to muscle through — prerequisites exist for a reason. A few free resources:

- **Python basics:** [Python for Everybody](https://www.py4e.com) (Dr. Chuck, free, excellent).
- **Stats refresher:** [Khan Academy Statistics](https://www.khanacademy.org/math/statistics-probability) or [StatQuest on YouTube](https://www.youtube.com/user/joshstarmer).
- **General AI literacy:** [Elements of AI](https://www.elementsofai.com/) — free online course, no coding required.
- **Enterprise context:** read the top-level `../../README.md` of this repo and pick the topic doc (e.g., `GenerativeAI.md`, `MLOps.md`) closest to your work.

## A note

The goal of this Week 0 is not to recruit you into a longer course. It's to give you an honest working picture of the space and point you at the next best step *for you* — including "you don't need another course." Everyone who finishes this week knows more than they did three hours ago. That is the win.

---

# Glossary

Terms introduced in this week, in the order they appeared:

- **Token** — a ~3-4 character chunk of text; the atomic unit models operate on.
- **Context window** — how many tokens the model can see at once (prompt + reply + pasted docs).
- **Training vs. inference** — the expensive one-time process of baking weights vs. each time you use the model.
- **Temperature** — sampler randomness knob, 0 = deterministic, ~1 = natural variety.
- **Hallucination** — confident, fluent, wrong output; a consequence of "predict plausible next token."
- **Harness** — the code wrapped around the LLM that gives it memory, tools, loops, etc.
- **RAG (Retrieval-Augmented Generation)** — search for relevant text, paste it into the prompt, model answers grounded on it.
- **Agent** — an LLM in a loop with tools and a goal.
- **Coding agent** — an agent specialized with shell + filesystem + editor tools.
- **Endpoint** — where the model is served from; public API, enterprise-hosted, or self-hosted.
- **Foundry / Bedrock / Vertex** — the three big enterprise-hosted endpoint families (Azure / AWS / GCP).
- **Benchmark** — a standardized test. Good for relative comparison, weak for predicting production fit.
- **Task-based eval** — a custom test set of real examples from your actual task.
- **LLM-as-Judge** — using a strong model to grade another model's outputs cheaply.
- **SFT (Supervised Fine-Tuning)** — train a base model on curated instruction-response pairs.
- **LoRA** — cheap fine-tuning that trains a small adapter instead of all weights.
- **Preference tuning (RLHF / DPO)** — the third training stage, teaches tone and polish from ranked pairs.

---

# Further Reading

General LLM intuition:
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar.
- [A Hacker's Guide to Language Models](https://www.youtube.com/watch?v=jkrNMKz9pWU) — Jeremy Howard, 90-min video survey.
- [Simon Willison on LLMs](https://simonwillison.net/tags/llms/) — ongoing practitioner commentary.

Harnesses and agents:
- [Anthropic — Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — design patterns.
- [LangChain concepts](https://python.langchain.com/docs/concepts/) — widely-used harness framework.

Endpoints and enterprise:
- [Azure AI Foundry docs](https://learn.microsoft.com/en-us/azure/ai-foundry/).
- [AWS Bedrock docs](https://docs.aws.amazon.com/bedrock/).
- [Ollama model library](https://ollama.com/library) — self-hosting made easy.

Evaluation:
- [Remote Labor Index](https://rli.scale.com).
- [SWE-bench](https://www.swebench.com).
- [Chatbot Arena](https://lmarena.ai).
- [Eugene Yan on evals](https://eugeneyan.com/writing/llm-patterns/).

Training:
- [Hugging Face — LoRA / PEFT docs](https://huggingface.co/docs/peft/).
- [InstructGPT paper (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) — the paper that set the now-standard SFT + RLHF pipeline.
- `../How-Tuning-Works/Week00/README.md` — the deeper treatment of the three-stage pipeline introduced in Segment 5.

---

**You're done with Week 0.** Whichever track you chose in Segment 6, you now have the working picture. That was the whole point.
