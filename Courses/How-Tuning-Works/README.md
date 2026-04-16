# How Tuning Works

## A Practical Course on What Turns a Raw LLM Into Something Useful

Large language models don't come out of the training run ready to chat. The "model" you call via an API is the end product of a multi-stage pipeline — pretraining, fine-tuning, preference tuning, quantization, deployment — and each stage shapes how the model behaves, what it costs, and where it fails.

This course walks through that pipeline end to end, with enough hands-on work that by the time you finish you can explain (and reproduce at small scale) every step between a raw transformer and a production-ready assistant.

---

> **Unsure whether this course or a different one fits you?** Start with [Week 0: AI & LLMs — A Working Survey](../Week00/README.md) in the parent `Courses/` directory. It's a standalone 2-3 hour tour that covers the whole AI stack at ankle-to-knee depth and ends with a decision rubric pointing to this course, to [Data Science for Fusion Engineers](../DataScienceForFusionEngineers.md), or to neither.

---

## Who This Course Is For

- **Engineering managers** who need to evaluate proposals from AI teams without taking everything on faith
- **Product managers** deciding whether to buy, build, or fine-tune
- **Engineers and data scientists** crossing over from classical ML into the LLM world
- **Support leaders and ops folks** who need to understand why their AI tools behave the way they do

You do **not** need deep ML background for Week 0. Later weeks assume comfort with Python and command-line tools.

---

## Course Philosophy

### Understand the pipeline, not just the endpoints

Every behavior you see in ChatGPT, Claude, or your local Ollama install is the result of specific choices made during training. Knowing which stage introduced which behavior is the difference between "the model is weird" and "this is a chat-template mismatch — we're hitting the wrong endpoint."

### Reproduce at small scale what the labs do at large scale

You can't train GPT-5 at home. But you can fine-tune a 1B model on a laptop in under an hour, and the mechanics are the same. Small-scale reproduction builds real intuition.

### Every week ends with something you can show your team

Concept → lab → deliverable. No weeks of pure theory.

---

## Prerequisites

### Week 0 (Manager / Onboarding Week)

- Willingness to spend 1-2 hours on concepts and a short hands-on lab
- A computer that can run [Ollama](https://ollama.com) (Mac, Linux, or Windows with 8GB+ RAM)
- No programming required

### Weeks 1+

- Comfortable with Python basics (functions, dicts, virtualenvs)
- Terminal / command-line comfortable
- A GPU helps but isn't required — cloud options covered in Week 1

---

## Course Overview

| Week | Theme | What You'll Build |
|------|-------|-------------------|
| [0](Week00/README.md) | **Why Tuning Matters** — base models, SFT, RLHF, chat templates | Compare a base model to an instruction-tuned model and see the difference yourself |
| 1 | Pretraining & Base Models — tokenization, scaling laws, data curation | Tokenize a dataset and estimate a scaling-law budget |
| 2 | Supervised Fine-Tuning (SFT) — datasets, training loops, LoRA | Fine-tune a 1B model on a custom instruction dataset |
| 3 | Chat Templates & Tool Use — ChatML, Gemma, Llama formats, function calling | Write a custom template and teach a model to emit tool calls |
| 4 | Preference Tuning — RLHF, DPO, reward models, constitutional AI | Run DPO on your fine-tune using a small preference dataset |
| 5 | Reasoning Models — thinking tokens, chain-of-thought, test-time compute | Add reasoning behavior to a small model |
| 6 | Evaluation — benchmarks, vibes, LLM-as-judge, regression testing | Build an eval harness for your fine-tune |
| 7 | Deployment & Quantization — GGUF, MXFP4, serving frameworks | Quantize your model and serve it via Ollama |
| 8 | Capstone — end-to-end fine-tune and deploy | Ship a specialized model solving a real problem |

Weeks 1-8 are stubbed; Week 0 is the first milestone and is fully developed.

---

## Tools Used

- **[Ollama](https://ollama.com)** — local model runner for labs
- **[Hugging Face](https://huggingface.co)** — model hub and `transformers` library
- **[TRL](https://github.com/huggingface/trl)** — fine-tuning toolkit (SFT, DPO, PPO)
- **[Unsloth](https://github.com/unslothai/unsloth)** — fast fine-tuning on consumer GPUs
- **Jupyter / VS Code** for notebooks and exercises

---

## How to Use This Course

1. Start with [Week 0](Week00/README.md). It's the "manager's week" — concepts, no code.
2. After Week 0, decide whether you want to go deeper as an operator or hand off to a hands-on engineer.
3. Weeks 1-8 each take ~3-5 hours if you do the labs.

---

## License

Same as the parent repository.
