# Week 0: Why Tuning Matters

## The Manager's Week — What Every Leader Should Know About How LLMs Are Built

**Time commitment:** 1-2 hours (45 min concepts + 30-45 min hands-on lab + 15 min review)

**Who this is for:** Managers, PMs, support leaders, and anyone about to fund, buy, evaluate, or deploy an AI product. Also recommended as a prerequisite for anyone taking the rest of this course.

**What you'll walk away with:** A mental model of how a raw transformer becomes a useful assistant, enough vocabulary to have a real conversation with an AI team, and the ability to spot when "the model is broken" is actually "the integration is wrong."

---

## Learning Objectives

By the end of this week, you will be able to:

1. Explain the three stages that turn a pretrained model into a chat assistant
2. Describe what a **base model** is and why you almost never want to talk to one directly
3. Define **instruction tuning** and **preference tuning** and what each contributes
4. Recognize a **chat template** and explain why it matters
5. Make an informed decision about when your organization needs to fine-tune vs. prompt-engineer vs. buy
6. Hold your own in a conversation with an ML engineer

---

## Why This Matters (The Manager's Framing)

If you manage or fund AI work, you will hear statements like:

- "The model is hallucinating — we need a fine-tune."
- "We should switch from Chat Completions to the Responses API."
- "RLHF made the model refuse reasonable requests."
- "We got better results with DPO than PPO on our preference dataset."
- "The chat template broke after the model upgrade."

Each of these statements is either a correct diagnosis or a symptom of a different problem entirely. Knowing the pipeline lets you push back — or agree — with evidence. This week gives you just enough context to do that.

**A concrete payoff:** teams routinely spend weeks fine-tuning a model to fix a problem that was actually caused by a chat-template mismatch. A ten-minute check at the start would have saved the whole cycle. This course teaches you that ten-minute check.

---

## Core Concepts

### The Three-Stage Pipeline

Every modern chat model — GPT-5.x, Claude 4.x, Gemma 4, Llama 4 — goes through three stages. If you remember nothing else from this week, remember these three.

```
   [ trillions of tokens ]
            │
            ▼
   ┌────────────────────┐
   │  Stage 1:          │
   │  PRETRAINING       │   →  Base Model
   │  (next-token       │      (text completer)
   │   prediction)      │
   └────────────────────┘
            │
            ▼
   ┌────────────────────┐
   │  Stage 2:          │
   │  SUPERVISED        │   →  Instruction-Tuned Model
   │  FINE-TUNING (SFT) │      (helpful assistant)
   │  (10K-100K pairs)  │
   └────────────────────┘
            │
            ▼
   ┌────────────────────┐
   │  Stage 3:          │
   │  PREFERENCE TUNING │   →  Polished Chat Model
   │  (RLHF / DPO)      │      (what you ship)
   │  (ranked pairs)    │
   └────────────────────┘
```

---

### Stage 1: Pretraining → the "base model"

Train a transformer on trillions of tokens of text — the internet, books, code, academic papers. The only job during training: **predict the next token**.

After this stage, the model is a very capable text autocomplete. It has no concept of "user" or "assistant." It doesn't know questions are meant to be answered. It just continues whatever text you give it.

**Example: asking a base model a question**

```
Input:   "What is the capital of France?"
Output:  '"What is the capital of France?" is a common trivia
          question often asked in elementary schools...'
```

The base model isn't being unhelpful — it's just doing what it was trained to do: continue the text. Your question, to it, is just the start of an essay.

Base models are useful as *starting points* for further training. They are rarely what you ship to end users.

---

### Stage 2: Supervised Fine-Tuning (SFT) → "instruction-tuned"

Take the base model and keep training it, but this time on curated examples of good behavior — typically 10,000 to 100,000 hand-written `(instruction, response)` pairs:

```
User:       What's the capital of France?
Assistant:  The capital of France is Paris.
```

```
User:       Summarize this email in two sentences.
            [email text...]
Assistant:  The customer is reporting an issue with billing.
            They want a callback by end of day.
```

After SFT, the model has learned a critical new skill: **when I see this shape of input, I should produce a helpful response**. This is the step that turns a text autocomplete into something that behaves like an assistant.

Ollama model tags default to the instruction-tuned variant. For example, when you run `ollama pull gemma4:e2b`, you're getting an instruction-tuned Gemma 4. The base versions usually have a `:text` or `:base` suffix.

---

### Stage 3: Preference Tuning (RLHF / DPO) → polished

The SFT model is helpful but often rough — it may be verbose, off-tone, or willing to do things you don't want it to do. Preference tuning fixes this.

The mechanism: humans look at pairs of model responses to the same question and rank them ("A is better than B"). The model is then trained to prefer the winning pattern.

Two common algorithms:

- **RLHF (Reinforcement Learning from Human Feedback)** — the original method. Train a reward model on the preferences, then use reinforcement learning (usually PPO) to optimize against it.
- **DPO (Direct Preference Optimization)** — a simpler, more recent method that skips the reward model and trains directly on preference pairs. Cheaper, more stable, now the default in most shops.

This stage is where models acquire:
- A consistent tone
- Refusals for unsafe requests
- The instinct to ask for clarification on ambiguous prompts
- Most "personality" traits

Modern models often get an additional **reasoning-tuning** pass here — this is how GPT-5.x "thinking" mode and similar reasoning features are built.

---

### The Key Technical Detail: Chat Templates

During SFT, the instruction pairs aren't shown to the model as plain text. They're wrapped in **special tokens** the model learns to recognize:

```
<|im_start|>user
What's the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

Different model families use different syntax:

| Model Family | Template Syntax (simplified) |
|---|---|
| OpenAI GPT / ChatML | `<|im_start|>role ... <|im_end|>` |
| Llama 2 / 3 | `[INST] ... [/INST]` |
| Gemma | `<start_of_turn>role ... <end_of_turn>` |
| Claude | Human: / Assistant: (historical), plus structured tags |

The specifics don't matter. What matters:

**The model was trained to expect these markers and produce output between them. Talk to it outside those markers and behavior degrades in strange ways — empty responses, rambling, template tokens leaking into the output.**

This is why calling the wrong API endpoint can silently produce broken results. It's not a bug — it's a template mismatch.

---

## Why This Matters for AI (Real-World Examples)

### Case study: the "empty response" bug

On a recent install of Ollama with `gemma4:e2b`, the same prompt produced completely different results depending on which endpoint was used:

```bash
# /api/generate — raw completion mode (no chat template applied)
# Result: 80 tokens generated, response field EMPTY

# /api/chat — chat mode (template applied automatically)
# Result: 284 tokens generated, coherent answer at 108 tok/sec
```

Same model, same prompt, same hardware. The only difference was whether the endpoint wrapped the input in the model's chat template. An engineer who doesn't know about templates might spend an afternoon convinced the model is broken. An engineer who does knows it's a one-line fix.

### Case study: "the model got worse after the upgrade"

A team upgrades from model version N to N+1. Suddenly, responses are terse, weird, or include literal `<end_of_turn>` tokens in the output. Someone files a bug: "regression in new model."

The actual problem: the new version's chat template changed, and the integration code still uses the old one. Five-minute fix if you know to look for it; two-week goose chase if you don't.

---

## Decision Framework: When Do You Need to Tune?

When a stakeholder says "we need to fine-tune our own model," run through this checklist **before** agreeing:

| Problem | Try First | Fine-tune? |
|---|---|---|
| Model doesn't know your domain facts | RAG (retrieval-augmented generation) | Rarely — RAG is usually better |
| Model's tone doesn't match your brand | System prompt + few-shot examples | Maybe, if prompts aren't enough |
| Model refuses reasonable requests | Use a less-aligned base or switch providers | Sometimes |
| Model is slow/expensive at scale | Quantize, use a smaller model, or distill | Yes — distillation is a fine-tune |
| Model doesn't follow your output format | Structured outputs / JSON mode | Rarely needed |
| Model needs to behave like a specific character | Prompt engineering | Maybe |
| You need to ship a specialized small model | Yes — SFT + DPO on a small base | **Yes, this is the real use case** |

**Heuristic:** if you can get 80% of the way with a good system prompt, fine-tuning probably isn't worth the operational burden. Fine-tunes add training pipeline cost, ongoing retraining as base models improve, and evaluation complexity.

---

## Hands-On Lab (30-45 minutes)

**Goal:** See the difference between a base model and an instruction-tuned model yourself.

### Setup

Install Ollama if you haven't already (see [ollama.com](https://ollama.com) or `brew install ollama` on Mac).

```bash
ollama pull gemma3:1b          # ~815 MB, instruction-tuned
ollama pull llama3.2:1b        # ~1.3 GB, instruction-tuned
ollama pull llama3.2:1b-text-q4_K_M   # ~770 MB, BASE model (no instruction tuning)
```

The `-text` suffix on the Llama tag is Ollama's convention for base models. This is what you'll compare against.

### Exercise 1 — Instruction-tuned behavior

```bash
ollama run llama3.2:1b "What is the capital of France?"
```

Expected: a direct answer like "The capital of France is Paris."

### Exercise 2 — Base model behavior

```bash
ollama run llama3.2:1b-text-q4_K_M "What is the capital of France?"
```

Expected: something weird. The base model will probably continue your text as if your question were the start of a document — maybe a list of more trivia questions, maybe an essay intro, maybe just rambling.

**Reflection:** the base model isn't worse. It's just not trained to answer questions. Given the same weights, SFT is what bridges the gap.

### Exercise 3 — Watch the chat template in action

Compare Ollama's two endpoints directly:

```bash
# Chat endpoint — template applied
curl -s http://localhost:11434/api/chat -d '{
  "model": "gemma3:1b",
  "messages": [{"role": "user", "content": "Name three colors."}],
  "stream": false
}' | python3 -c "import json,sys; print(json.load(sys.stdin)['message']['content'])"

# Generate endpoint — raw, no template
curl -s http://localhost:11434/api/generate -d '{
  "model": "gemma3:1b",
  "prompt": "Name three colors.",
  "stream": false
}' | python3 -c "import json,sys; print(repr(json.load(sys.stdin)['response']))"
```

You may see dramatically different quality between the two. The underlying model is identical — the only difference is whether the input was wrapped in the model's trained chat template.

### Deliverable

Write a one-paragraph summary describing what happened in each exercise, in your own words, and what you'd explain to a colleague who asked "why does this model sometimes return nothing?"

---

## Knowledge Check

Answer these before moving on. No peeking.

1. What's the difference between a base model and an instruction-tuned model?
2. What does SFT stand for, and what does it do?
3. What does preference tuning (RLHF/DPO) add on top of SFT?
4. Why does the same model produce different results on `/api/generate` vs `/api/chat`?
5. When should you push back on a proposal to fine-tune, and what should you suggest first?
6. A teammate says: "The model is broken — I'm getting empty responses." What's your first debugging question?

---

## Bridge to Week 1

You now have the shape of the pipeline. Next week we zoom in on **Stage 1: Pretraining**. How do trillions of tokens become a base model? What's a tokenizer? What are scaling laws and why does every ML blog post talk about them? By the end of Week 1, you'll have tokenized a dataset yourself and estimated what it would cost to pretrain a small model from scratch.

---

## Further Reading

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar's classic explainer on how the underlying architecture works
- [Hugging Face — Chat Templates](https://huggingface.co/docs/transformers/main/chat_templating) — the canonical reference on templates
- [Chip Huyen — RLHF](https://huyenchip.com/2023/05/02/rlhf.html) — deep dive on preference tuning
- [Ollama model library](https://ollama.com/library) — browse base vs. instruction-tuned variants
- [InstructGPT paper (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) — the paper that introduced the now-standard pipeline
