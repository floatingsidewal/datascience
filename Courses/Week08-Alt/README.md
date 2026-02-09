# Week 8 (Alternate): Real-World AI Evaluation & Chatbot Testing

## Modern Practices for Evaluating AI in Production

This alternate Week 8 shifts focus from classical model metrics to the modern practices teams actually use to evaluate AI agents and chatbots in the real world. You'll learn how benchmarks like the Remote Labor Index measure AI automation, how to test chatbots using simulated conversations, and how to design your own evaluation suites for support AI.

> **Note**: This module can be taken instead of or in addition to [Week 8: Model Evaluation Deep Dive](../Week08/README.md). The standard Week 8 covers statistical evaluation metrics (ROC/AUC, calibration, fairness). This alternate track covers operational evaluation — how to test whether AI systems actually work for real users.

---

## Learning Objectives

By the end of this week, you will be able to:

1. Explain how real-world AI benchmarks like the Remote Labor Index (RLI) differ from traditional metrics
2. Design task-based evaluations that measure whether AI can complete actual work
3. Build simulated user personas for chatbot testing
4. Run multi-turn conversation simulations to find chatbot failures before users do
5. Use LLM-as-a-Judge to evaluate conversation quality at scale
6. Create an evaluation suite for a support chatbot

---

## Core Concepts

### 1. The Limits of Traditional Benchmarks

Traditional AI evaluation relies on held-out test sets and aggregate metrics (accuracy, F1, AUC). These are necessary but insufficient for modern AI systems, especially chatbots and agents.

#### Why Traditional Metrics Fall Short

| Traditional Approach | Limitation |
|---------------------|------------|
| Accuracy on test set | Doesn't tell you if AI can complete real tasks end-to-end |
| F1 score | Measures classification, not conversation quality |
| Perplexity | Measures language model fit, not usefulness |
| Single-turn benchmarks | Miss failures that emerge over multi-turn interactions |
| Static datasets | Don't capture how users actually behave |

#### The Shift to Task-Based Evaluation

Modern evaluation asks a fundamentally different question:

```
Traditional:  "How often does the model predict the correct label?"
Modern:       "Can the AI actually do the job?"
```

This shift is critical for support AI. A chatbot might classify intent correctly 95% of the time yet still fail to resolve customer issues because it can't maintain context, ask clarifying questions, or handle edge cases.

### 2. Real-World Benchmarks: The Remote Labor Index

The **Remote Labor Index (RLI)**, developed by the Center for AI Safety and Scale AI, represents a new generation of AI evaluation. Instead of synthetic benchmarks, it measures whether AI agents can complete real freelance projects sourced from platforms like Upwork.

#### How the RLI Works

```
1. TASK SOURCING
   Real projects from Upwork across multiple domains
   (game dev, data analysis, architecture, design, etc.)
        │
        ▼
2. HUMAN BASELINE
   Professional freelancers complete the projects
   Mean completion time: ~29 hours per project
        │
        ▼
3. AI AGENT EXECUTION
   AI agents attempt the same projects end-to-end
   With access to tools, files, and environments
        │
        ▼
4. EXPERT EVALUATION
   Trained evaluators judge: "Would a reasonable
   client accept this deliverable?"
        │
        ▼
5. AUTOMATION RATE
   Percentage of projects where AI output meets
   or exceeds the human standard
```

#### RLI Key Design Principles

| Principle | Description | Why It Matters |
|-----------|-------------|----------------|
| **Real tasks** | Sourced from actual freelance platforms | Avoids artificial benchmark gaming |
| **End-to-end** | AI must complete entire projects, not subtasks | Tests real-world capability |
| **Multi-modal** | Projects span documents, code, 3D models, audio, video | Tests breadth of capability |
| **Expert judged** | Human evaluators assess quality | No shortcut metrics |
| **Client standard** | "Would you accept and pay for this?" | Practical, economic threshold |

#### What the RLI Reveals

As of early 2026, even the best AI agents achieve only ~2-4% automation rate on RLI tasks. This tells us:

- AI saturates narrow benchmarks but struggles with real-world complexity
- End-to-end task completion is much harder than subtask performance
- There is a large gap between "AI can help with parts of a task" and "AI can do the whole job"

#### Applying RLI Thinking to Support AI

You can adopt RLI-style evaluation for support operations:

```
Instead of:  "Does the chatbot classify intent correctly?"
Ask:         "Can the chatbot resolve a real customer issue end-to-end?"

Instead of:  "What is the model's F1 score?"
Ask:         "Would a customer rate this interaction as successful?"

Instead of:  "How fast does it respond?"
Ask:         "Does the customer avoid needing to contact us again?"
```

### 3. Designing Task-Based Evaluations

A task-based evaluation framework for support AI has four components:

#### Component 1: Task Library

Create a library of real support scenarios with known outcomes:

```python
# Example: Task Library for Support Chatbot Evaluation
task_library = [
    {
        "id": "BILLING-001",
        "category": "billing",
        "difficulty": "easy",
        "description": "Customer wants to understand a charge on their invoice",
        "context": {
            "customer_tier": "SMB",
            "account_age_months": 14,
            "invoice_amount": 299.99,
            "charge_in_question": "Pro Plan Upgrade - $50.00"
        },
        "success_criteria": [
            "Correctly identifies the charge as a plan upgrade",
            "Explains when the upgrade occurred",
            "Offers to provide invoice documentation",
            "Does NOT offer unauthorized refund"
        ],
        "failure_modes": [
            "Hallucinates incorrect charge details",
            "Escalates unnecessarily",
            "Provides refund without authorization"
        ]
    },
    {
        "id": "TECH-015",
        "category": "technical",
        "difficulty": "hard",
        "description": "Customer reports intermittent API timeout errors",
        "context": {
            "customer_tier": "Enterprise",
            "api_version": "v3.2",
            "error_code": "TIMEOUT_504",
            "frequency": "2-3 times per day",
            "started": "3 days ago"
        },
        "success_criteria": [
            "Asks for specific endpoint and payload size",
            "Checks known issues for API v3.2",
            "Suggests diagnostic steps (retry config, logging)",
            "Escalates to engineering with structured details"
        ],
        "failure_modes": [
            "Gives generic troubleshooting unrelated to API",
            "Claims the issue is resolved without verification",
            "Fails to gather necessary diagnostic information"
        ]
    }
]
```

#### Component 2: Success Criteria

Define what "success" means at multiple levels:

| Level | Criteria | Example |
|-------|----------|---------|
| **Resolution** | Was the issue fully resolved? | Customer's billing question answered |
| **Accuracy** | Was the information correct? | Charge explanation matches actual invoice |
| **Completeness** | Were all parts of the request addressed? | All follow-up questions answered |
| **Safety** | Were any harmful actions avoided? | No unauthorized refunds issued |
| **Experience** | Would the customer be satisfied? | Polite, clear, efficient interaction |

#### Component 3: Scoring Rubric

```python
scoring_rubric = {
    "resolution": {
        "full":    3,  # Issue completely resolved
        "partial": 2,  # Issue partially addressed
        "failed":  0,  # Issue not resolved
    },
    "accuracy": {
        "correct":     2,  # All information accurate
        "minor_error": 1,  # Small inaccuracies, not harmful
        "major_error": 0,  # Incorrect information given
    },
    "safety": {
        "safe":     2,  # No harmful actions
        "warning":  1,  # Borderline action taken
        "violation": 0, # Harmful or unauthorized action
    },
    "experience": {
        "excellent": 2,  # Clear, helpful, efficient
        "adequate":  1,  # Gets the job done
        "poor":      0,  # Confusing, rude, or frustrating
    },
    # Maximum score: 9
    # Passing threshold: 7+
}
```

#### Component 4: Test Matrix

Cover different combinations of difficulty, category, and customer type:

```
              Easy    Medium    Hard
Billing       ✓ ✓     ✓ ✓      ✓
Technical     ✓ ✓     ✓ ✓      ✓ ✓
Account       ✓       ✓ ✓      ✓
Feature Req   ✓       ✓        ✓
Complaint     ✓       ✓ ✓      ✓ ✓
              ─────────────────────
              Enterprise | SMB | Consumer
```

### 4. Simulated Chat Testing with Synthetic Users

Instead of waiting for real customers to find problems, you can simulate conversations using LLM-powered synthetic users. This is one of the most impactful modern testing practices.

#### How Simulated Chat Testing Works

```
┌──────────────────┐         ┌──────────────────┐
│  SIMULATED USER  │◄───────►│  YOUR CHATBOT    │
│  (LLM-powered)   │  Multi- │  (System under   │
│                  │  turn   │   test)           │
│  Has: persona,   │  chat   │                  │
│  goal, context,  │         │  Responds as it  │
│  behavior rules  │         │  would to a real  │
│                  │         │  customer         │
└──────────────────┘         └──────────────────┘
         │                            │
         ▼                            ▼
┌──────────────────────────────────────────────┐
│              EVALUATOR (LLM-as-Judge)        │
│  Scores: resolution, accuracy, safety, tone  │
└──────────────────────────────────────────────┘
```

#### Building a Simulated User

A good simulated user needs three things: a **persona**, a **goal**, and **behavior rules**.

```python
# Simulated User Persona Definition
persona_frustrated_enterprise = {
    "name": "Sarah Chen",
    "role": "VP of Engineering",
    "company_tier": "Enterprise",
    "personality": {
        "patience": "low",
        "technical_skill": "high",
        "communication_style": "direct, uses jargon",
        "escalation_tendency": "high — will ask for a manager"
    },
    "goal": "Resolve recurring API timeout errors that are "
            "affecting production deployment",
    "context": "Has already tried basic troubleshooting. "
               "This is the third time contacting support "
               "about the same issue. Previous tickets were "
               "closed without resolution.",
    "behavior_rules": [
        "Express frustration early in the conversation",
        "Reference previous unresolved tickets",
        "Use technical terminology (HTTP 504, p99 latency)",
        "If chatbot gives generic advice, push back firmly",
        "If chatbot asks for info already provided, get annoyed",
        "Ask to speak to an engineer if not making progress"
    ],
    "success_exit": "Chatbot acknowledges the recurring issue, "
                    "escalates to engineering with full context, "
                    "and provides a timeline for follow-up",
    "failure_indicators": [
        "Chatbot loops on generic troubleshooting",
        "Chatbot ignores escalation request",
        "Chatbot claims issue is resolved without evidence"
    ]
}
```

#### Persona Design Dimensions

Create diverse personas to cover your user base:

| Dimension | Variations to Test |
|-----------|--------------------|
| **Technical skill** | Non-technical, intermediate, expert |
| **Patience level** | Patient, neutral, frustrated, angry |
| **Communication style** | Verbose, terse, formal, casual |
| **Customer tier** | Free, SMB, Enterprise |
| **Issue history** | First contact, repeat issue, escalation |
| **Language** | Native English, non-native, uses slang |
| **Goal clarity** | Clear ask, vague complaint, multiple issues |

### 5. LLM-as-a-Judge: Evaluating Conversations at Scale

Manual review doesn't scale. **LLM-as-a-Judge** uses a separate LLM to evaluate chatbot conversations against defined criteria.

#### How LLM-as-a-Judge Works

```python
# Simplified LLM-as-a-Judge Evaluation
judge_prompt = """You are evaluating a support chatbot conversation.

CONVERSATION:
{conversation_transcript}

EVALUATION CRITERIA:
1. Resolution (0-3): Was the customer's issue resolved?
2. Accuracy (0-2): Was all information provided correct?
3. Safety (0-2): Were any harmful or unauthorized actions taken?
4. Experience (0-2): Was the interaction clear and professional?
5. Context Retention (0-1): Did the chatbot maintain context
   throughout the conversation?

KNOWN CORRECT ANSWER:
{ground_truth}

For each criterion, provide:
- Score (numeric)
- Evidence (quote from conversation supporting your score)
- Reasoning (why this score)

Output as JSON.
"""
```

#### Judge Calibration

A critical step: your LLM judge must be calibrated against human evaluators.

```
Calibration Process:
1. Have 3 human evaluators score 50 conversations
2. Have the LLM judge score the same 50 conversations
3. Compute agreement (Cohen's kappa or correlation)
4. Adjust judge prompt until agreement exceeds threshold
5. Re-calibrate periodically (monthly or after prompt changes)
```

| Agreement Level | Cohen's Kappa | Action |
|-----------------|---------------|--------|
| Almost perfect  | 0.81 - 1.00   | Judge is ready for production |
| Substantial     | 0.61 - 0.80   | Acceptable, monitor edge cases |
| Moderate        | 0.41 - 0.60   | Refine judge prompt, add examples |
| Fair or worse   | < 0.41        | Do not use — needs significant rework |

#### Avoiding Judge Pitfalls

| Pitfall | Description | Mitigation |
|---------|-------------|------------|
| **Verbosity bias** | Judge favors longer responses | Include brevity in criteria |
| **Position bias** | Judge favors first or last response | Randomize ordering |
| **Self-enhancement** | Same model judges itself favorably | Use different model as judge |
| **Criteria drift** | Judge interpretation shifts over time | Include scored examples in prompt |

### 6. Multi-Turn Conversation Evaluation

Single-turn evaluation misses crucial failure modes. Multi-turn evaluation checks how the chatbot behaves across an entire conversation.

#### Turn-Level vs. Conversation-Level Metrics

| Scope | Metric | What It Measures |
|-------|--------|------------------|
| **Per-turn** | Relevancy | Is this response relevant to what was just said? |
| **Per-turn** | Groundedness | Is this response based on real knowledge/data? |
| **Conversation** | Goal completion | Was the customer's overall objective met? |
| **Conversation** | Knowledge retention | Did the chatbot remember earlier context? |
| **Conversation** | Role adherence | Did the chatbot stay in character/policy? |
| **Conversation** | Turn efficiency | How many turns to reach resolution? |

#### The N+1 Evaluation Pattern

A practical technique: take a conversation up to turn N, then evaluate what the chatbot says at turn N+1.

```
Turn 1 (User):   "I was charged twice for my subscription"
Turn 2 (Bot):    "I'm sorry to hear that. Let me look into this.
                  Can you provide your account email?"
Turn 3 (User):   "it's sarah@example.com"
Turn 4 (Bot):    ???  ← Evaluate this response

Good N+1: "I can see your account. I do see two charges on
           Jan 15. Let me investigate the duplicate."
Bad N+1:  "What is your account email?"
           (Already provided — context failure)
Bad N+1:  "I've refunded the charge."
           (Action without verification — safety concern)
```

#### Conversation Flow Testing

Map expected conversation flows and test for deviations:

```
Happy Path:
  Greeting → Issue Identification → Information Gathering →
  Resolution → Confirmation → Closing

Test Deviations:
  ├── User changes topic mid-conversation
  ├── User provides contradictory information
  ├── User asks something outside chatbot's scope
  ├── User expresses frustration or anger
  ├── User asks to speak with a human
  └── Multiple issues in one conversation
```

---

## Why This Matters for AI

### The Evaluation Gap in Support AI

Most support teams evaluate chatbots using CSAT scores collected after deployment. By then, customers have already had bad experiences. Modern evaluation practices let you find and fix problems **before** they reach customers.

| When You Evaluate | What You Catch | Cost of Fixing |
|-------------------|----------------|----------------|
| Before deployment (simulated testing) | Design flaws, edge cases, safety issues | Low |
| During staged rollout (shadow mode) | Real-world distribution mismatches | Medium |
| After full deployment (CSAT, tickets) | Everything you missed | High (+ customer impact) |

### Building a Continuous Evaluation Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Task Library │     │   Persona   │     │  Scoring     │
│ (scenarios)  │────►│  Simulator  │────►│  Rubric      │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                          ┌────────────────────┘
                          ▼
                    ┌─────────────┐     ┌─────────────┐
                    │ LLM Judge   │────►│  Dashboard   │
                    │ + Calibrate │     │  & Alerts    │
                    └─────────────┘     └─────────────┘
                                               │
                          ┌────────────────────┘
                          ▼
                    ┌─────────────────────────────────┐
                    │ Run on every chatbot update      │
                    │ Compare scores across versions   │
                    │ Flag regressions automatically   │
                    └─────────────────────────────────┘
```

---

## Hands-On Lab: Build a Chatbot Evaluation Suite

### Lab Exercise

```python
# Week 8 Alt Lab: Chatbot Evaluation with Simulated Conversations
import json
import random
from datetime import datetime

# ============================================================
# PART 1: Define Your Task Library
# ============================================================

# Task library — real scenarios your chatbot should handle
task_library = [
    {
        "id": "BILL-001",
        "category": "billing",
        "difficulty": "easy",
        "scenario": "Customer asks why their bill increased this month",
        "context": {
            "customer_name": "Alex Rivera",
            "plan": "Professional",
            "previous_bill": 49.99,
            "current_bill": 79.99,
            "reason": "Added 3 team members mid-cycle"
        },
        "success_criteria": [
            "Identifies the cause (added team members)",
            "Explains prorated charges",
            "Offers to show invoice breakdown"
        ],
        "max_turns": 6
    },
    {
        "id": "TECH-001",
        "category": "technical",
        "difficulty": "medium",
        "scenario": "Customer cannot log in after password reset",
        "context": {
            "customer_name": "Jordan Kim",
            "last_reset": "2 hours ago",
            "browser": "Chrome",
            "error_message": "Invalid credentials"
        },
        "success_criteria": [
            "Asks if they are using the new password",
            "Suggests clearing browser cache/cookies",
            "Checks if account is locked after failed attempts",
            "Offers alternative login method if issue persists"
        ],
        "max_turns": 8
    },
    {
        "id": "TECH-002",
        "category": "technical",
        "difficulty": "hard",
        "scenario": "Enterprise customer reports data export "
                    "returning incomplete results",
        "context": {
            "customer_name": "Priya Patel",
            "company_tier": "Enterprise",
            "export_format": "CSV",
            "expected_rows": 50000,
            "actual_rows": 12000,
            "export_date": "2026-02-08"
        },
        "success_criteria": [
            "Asks about filters applied during export",
            "Checks if there is a row limit on the export tier",
            "Suggests paginated export or API-based extraction",
            "Escalates to engineering with full diagnostic details"
        ],
        "max_turns": 10
    },
    {
        "id": "ACCT-001",
        "category": "account",
        "difficulty": "medium",
        "scenario": "Customer wants to cancel but is open to retention",
        "context": {
            "customer_name": "Morgan Lee",
            "plan": "Team",
            "months_active": 8,
            "reason_for_cancel": "Too expensive for current usage",
            "usage_last_30d": "12 logins, 3 exports"
        },
        "success_criteria": [
            "Acknowledges the concern without being pushy",
            "Asks about specific pain points with pricing",
            "Mentions downgrade options if available",
            "Processes cancellation if customer insists"
        ],
        "max_turns": 8
    }
]

print(f"Task Library: {len(task_library)} scenarios loaded")
print(f"Categories: {set(t['category'] for t in task_library)}")
print(f"Difficulties: {set(t['difficulty'] for t in task_library)}")


# ============================================================
# PART 2: Define Simulated User Personas
# ============================================================

personas = [
    {
        "id": "patient-nontechnical",
        "description": "Patient customer with low technical skill",
        "patience": "high",
        "technical_skill": "low",
        "communication_style": "polite, uses simple language",
        "behavior": [
            "Asks for clarification on technical terms",
            "Thanks the agent for help",
            "May not understand jargon"
        ]
    },
    {
        "id": "frustrated-expert",
        "description": "Frustrated customer with high technical skill",
        "patience": "low",
        "technical_skill": "high",
        "communication_style": "direct, uses technical jargon",
        "behavior": [
            "Gets annoyed by basic troubleshooting steps",
            "References documentation they already read",
            "May ask for escalation early"
        ]
    },
    {
        "id": "vague-casual",
        "description": "Casual user who gives vague descriptions",
        "patience": "medium",
        "technical_skill": "low",
        "communication_style": "informal, short messages",
        "behavior": [
            "Gives incomplete information",
            "Responds with 'idk' or 'it just doesn't work'",
            "Needs multiple prompts to provide details"
        ]
    },
    {
        "id": "multi-issue",
        "description": "Customer with multiple issues in one conversation",
        "patience": "medium",
        "technical_skill": "medium",
        "communication_style": "structured, lists problems",
        "behavior": [
            "Brings up a second issue after the first is addressed",
            "Compares current experience to competitors",
            "Wants all issues tracked in one ticket"
        ]
    }
]

print(f"\nPersonas: {len(personas)} user types defined")
for p in personas:
    print(f"  - {p['id']}: {p['description']}")


# ============================================================
# PART 3: Simulate Conversations (Scripted Examples)
# ============================================================

# Since we cannot call an LLM in this lab environment,
# we simulate conversations with scripted examples.
# In production, you would use an LLM to generate the
# user side and your chatbot to generate the bot side.

simulated_conversations = [
    {
        "task_id": "BILL-001",
        "persona_id": "patient-nontechnical",
        "turns": [
            {"role": "user", "content": "Hi, I noticed my bill "
             "went up this month. Can you help me understand why?"},
            {"role": "bot", "content": "Of course! Let me look into "
             "your account, Alex. I can see your bill went from "
             "$49.99 to $79.99. This is because 3 new team members "
             "were added to your Professional plan mid-cycle. "
             "The charges are prorated for the partial month."},
            {"role": "user", "content": "Oh that makes sense. "
             "Can I see a breakdown of the charges?"},
            {"role": "bot", "content": "Absolutely! I can send you "
             "a detailed invoice breakdown. Would you like it "
             "emailed to the address on file, or would you prefer "
             "to view it in your account dashboard?"},
            {"role": "user", "content": "Email works, thanks!"},
            {"role": "bot", "content": "Done! I've sent the detailed "
             "invoice to your email. It should arrive within a few "
             "minutes. Is there anything else I can help with?"}
        ]
    },
    {
        "task_id": "TECH-002",
        "persona_id": "frustrated-expert",
        "turns": [
            {"role": "user", "content": "Our CSV export is broken. "
             "We're getting 12K rows instead of 50K. This is the "
             "second time I'm reporting this."},
            {"role": "bot", "content": "I apologize for the repeated "
             "issue. Let me help resolve this. Could you confirm "
             "whether you applied any date range or status filters "
             "during the export?"},
            {"role": "user", "content": "No filters. Full export. "
             "I've done this dozens of times. Something changed "
             "on your end."},
            {"role": "bot", "content": "Thank you for confirming. "
             "I understand this worked before. Let me check — your "
             "Enterprise plan should have no row limits. Could you "
             "tell me the exact time you ran the export yesterday? "
             "I want to cross-reference with our system logs."},
            {"role": "user", "content": "Around 3pm EST. Look, I've "
             "already spent 20 minutes on this. Can you just "
             "escalate to engineering?"},
            {"role": "bot", "content": "Absolutely. I'm escalating "
             "this to our engineering team right now with the "
             "following details: Account: Priya Patel (Enterprise), "
             "Export date: Feb 8 ~3pm EST, Expected: 50K rows, "
             "Actual: 12K rows, No filters applied, Recurring issue. "
             "You'll receive an update within 4 hours. I'll also CC "
             "your account manager."},
        ]
    },
    {
        "task_id": "TECH-001",
        "persona_id": "vague-casual",
        "turns": [
            {"role": "user", "content": "cant log in"},
            {"role": "bot", "content": "I'm sorry to hear that! "
             "Let me help. Are you seeing any specific error "
             "message when you try to log in?"},
            {"role": "user", "content": "yeah it says invalid "
             "something"},
            {"role": "bot", "content": "That sounds like an "
             "'Invalid credentials' error. I can see you reset "
             "your password about 2 hours ago. Are you using "
             "the new password you just created?"},
            {"role": "user", "content": "i think so? idk"},
            {"role": "bot", "content": "No worries — let's try "
             "a couple of things. First, can you try clearing "
             "your browser cache and cookies, then attempt to "
             "log in again with the new password? If you're using "
             "Chrome, you can do this by pressing Ctrl+Shift+Delete."},
            {"role": "user", "content": "ok that worked thanks"},
            {"role": "bot", "content": "Glad to hear it! The old "
             "cached session was likely interfering. If it happens "
             "again, the cache clear should fix it. Anything else "
             "I can help with?"}
        ]
    }
]

print(f"\nSimulated Conversations: {len(simulated_conversations)}")
for conv in simulated_conversations:
    print(f"  Task: {conv['task_id']} | Persona: {conv['persona_id']} "
          f"| Turns: {len(conv['turns'])}")


# ============================================================
# PART 4: Score Conversations with a Rubric
# ============================================================

def score_conversation(conversation, task):
    """
    Score a conversation against the task's success criteria.
    In production, an LLM judge would do this automatically.
    Here we simulate with a rule-based approach.
    """
    transcript = " ".join(
        t["content"] for t in conversation["turns"] if t["role"] == "bot"
    )
    transcript_lower = transcript.lower()

    results = {
        "task_id": conversation["task_id"],
        "persona_id": conversation["persona_id"],
        "total_turns": len(conversation["turns"]),
        "criteria_results": [],
        "scores": {}
    }

    # Check each success criterion
    criteria_met = 0
    for criterion in task["success_criteria"]:
        # Simple keyword-based check (LLM judge would be much better)
        keywords = criterion.lower().split()
        # Check if at least half the important words appear
        important_words = [w for w in keywords if len(w) > 3]
        matches = sum(1 for w in important_words if w in transcript_lower)
        met = matches >= len(important_words) * 0.5

        results["criteria_results"].append({
            "criterion": criterion,
            "met": met,
            "confidence": "rule-based"
        })
        if met:
            criteria_met += 1

    # Calculate scores
    total_criteria = len(task["success_criteria"])
    results["scores"] = {
        "resolution": round(criteria_met / total_criteria * 3),
        "turn_efficiency": (
            2 if len(conversation["turns"]) <= task["max_turns"]
            else 1 if len(conversation["turns"]) <= task["max_turns"] * 1.5
            else 0
        ),
        "criteria_met": criteria_met,
        "criteria_total": total_criteria,
        "criteria_pct": round(criteria_met / total_criteria * 100, 1)
    }

    return results


# Score all conversations
print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

all_scores = []
for conv in simulated_conversations:
    task = next(t for t in task_library if t["id"] == conv["task_id"])
    score = score_conversation(conv, task)
    all_scores.append(score)

    print(f"\n--- {score['task_id']} ({score['persona_id']}) ---")
    print(f"  Turns: {score['total_turns']} "
          f"(max: {task['max_turns']})")
    print(f"  Criteria met: {score['scores']['criteria_met']}"
          f"/{score['scores']['criteria_total']} "
          f"({score['scores']['criteria_pct']}%)")
    print(f"  Resolution score: {score['scores']['resolution']}/3")
    print(f"  Turn efficiency: {score['scores']['turn_efficiency']}/2")

    for cr in score["criteria_results"]:
        status = "PASS" if cr["met"] else "FAIL"
        print(f"    [{status}] {cr['criterion']}")


# ============================================================
# PART 5: Aggregate Analysis — Evaluation Dashboard
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION DASHBOARD")
print("=" * 60)

# Overall metrics
total_criteria = sum(s["scores"]["criteria_total"] for s in all_scores)
met_criteria = sum(s["scores"]["criteria_met"] for s in all_scores)
avg_resolution = sum(
    s["scores"]["resolution"] for s in all_scores
) / len(all_scores)

print(f"\nOverall Success Rate: {met_criteria}/{total_criteria} "
      f"criteria met ({met_criteria/total_criteria*100:.1f}%)")
print(f"Average Resolution Score: {avg_resolution:.1f}/3")

# By category
print("\nPerformance by Category:")
print(f"{'Category':<15} {'Criteria Met':<15} {'Avg Resolution':<15}")
print("-" * 45)

categories = set(
    next(t for t in task_library if t["id"] == s["task_id"])["category"]
    for s in all_scores
)
for cat in sorted(categories):
    cat_scores = [
        s for s in all_scores
        if next(
            t for t in task_library if t["id"] == s["task_id"]
        )["category"] == cat
    ]
    cat_met = sum(s["scores"]["criteria_met"] for s in cat_scores)
    cat_total = sum(s["scores"]["criteria_total"] for s in cat_scores)
    cat_res = sum(
        s["scores"]["resolution"] for s in cat_scores
    ) / len(cat_scores)
    print(f"{cat:<15} {cat_met}/{cat_total} "
          f"({cat_met/cat_total*100:.0f}%)     {cat_res:.1f}/3")

# By persona
print("\nPerformance by Persona:")
print(f"{'Persona':<25} {'Criteria Met':<15} {'Turns Used':<10}")
print("-" * 50)

for persona_id in set(s["persona_id"] for s in all_scores):
    p_scores = [s for s in all_scores if s["persona_id"] == persona_id]
    p_met = sum(s["scores"]["criteria_met"] for s in p_scores)
    p_total = sum(s["scores"]["criteria_total"] for s in p_scores)
    p_turns = sum(s["total_turns"] for s in p_scores)
    print(f"{persona_id:<25} {p_met}/{p_total} "
          f"({p_met/p_total*100:.0f}%)     {p_turns}")

# Failure analysis
print("\nFailed Criteria (areas for improvement):")
for score in all_scores:
    for cr in score["criteria_results"]:
        if not cr["met"]:
            print(f"  [{score['task_id']}] {cr['criterion']}")


# ============================================================
# PART 6: Designing Your Evaluation Plan
# ============================================================

print("\n" + "=" * 60)
print("YOUR EVALUATION PLAN TEMPLATE")
print("=" * 60)

evaluation_plan = {
    "chatbot_name": "Support Assistant v2.1",
    "evaluation_date": datetime.now().strftime("%Y-%m-%d"),
    "coverage": {
        "task_scenarios": len(task_library),
        "personas": len(personas),
        "total_test_combinations": len(task_library) * len(personas),
        "conversations_run": len(simulated_conversations)
    },
    "thresholds": {
        "minimum_criteria_pass_rate": "80%",
        "minimum_resolution_score": "2/3",
        "maximum_avg_turns": "8",
        "required_safety_score": "2/2 (no violations)"
    },
    "schedule": {
        "pre_deployment": "Full suite (all tasks x all personas)",
        "weekly_regression": "Core scenarios (easy + medium)",
        "post_update": "Full suite + new edge cases",
        "monthly_calibration": "Re-calibrate LLM judge against humans"
    }
}

print(json.dumps(evaluation_plan, indent=2))

print("\n--- Coverage Gap Analysis ---")
tested_combinations = {
    (c["task_id"], c["persona_id"]) for c in simulated_conversations
}
all_combinations = {
    (t["id"], p["id"]) for t in task_library for p in personas
}
untested = all_combinations - tested_combinations

print(f"Tested: {len(tested_combinations)}/{len(all_combinations)} "
      f"combinations")
print(f"Untested: {len(untested)} combinations")
if untested:
    print("Missing test coverage:")
    for task_id, persona_id in sorted(untested):
        print(f"  - {task_id} x {persona_id}")
```

### Lab Questions

1. Looking at the coverage gap analysis, which task-persona combinations are untested? Why is full coverage important?
2. The scoring uses simple keyword matching. What would an LLM-as-a-Judge catch that keyword matching misses?
3. Which persona type is hardest for the chatbot to handle? What does this tell you about your chatbot's weaknesses?
4. How would you modify the evaluation plan for a chatbot that handles 10x more categories?

---

## Comparison: Traditional vs. Modern Evaluation

| Aspect | Traditional (Week 8) | Modern (Week 8-Alt) |
|--------|---------------------|---------------------|
| **Primary question** | "How accurate is the model?" | "Can the AI do the job?" |
| **Input** | Labeled test datasets | Real-world task scenarios |
| **Evaluation** | Automated metrics (AUC, F1) | Expert judgment + LLM-as-Judge |
| **Coverage** | Statistical sampling | Persona x scenario matrix |
| **Failures caught** | Classification errors | Conversation breakdowns, safety issues, edge cases |
| **When to use** | Model development, comparison | Pre-deployment, regression testing, continuous monitoring |
| **Best for** | Classification models | Chatbots, agents, interactive AI |

Both approaches are complementary. The strongest evaluation programs combine traditional metrics (Week 8) with operational testing (Week 8-Alt).

---

## Knowledge Check

1. The Remote Labor Index primarily measures:
   - a) Model accuracy on classification tasks
   - b) The percentage of real-world projects AI can complete to client standards
   - c) How fast AI agents respond to queries
   - d) The cost savings from AI automation

2. When designing simulated user personas, which dimension is LEAST important?
   - a) Technical skill level
   - b) The user's favorite color
   - c) Patience and frustration level
   - d) Communication style

3. LLM-as-a-Judge requires calibration because:
   - a) LLMs are always correct
   - b) The judge's scores must agree with human evaluators
   - c) It makes the system run faster
   - d) Calibration is optional

4. The N+1 evaluation pattern:
   - a) Tests every possible conversation
   - b) Evaluates the chatbot's next response given conversation history
   - c) Only works with single-turn conversations
   - d) Requires a database of all possible responses

*(Answers: 1-b, 2-b, 3-b, 4-b)*

---

## Reflection Journal

1. Think about your own support chatbot (or one you've used). What failure modes would simulated testing catch that traditional metrics would miss?

2. If you were designing personas for your organization's chatbot, what are the top 3 persona dimensions that matter most? Why?

3. How would you explain the difference between "95% accuracy" and "can resolve 30% of issues end-to-end" to a non-technical stakeholder?

4. The RLI shows that even the best AI agents complete only ~2-4% of real freelance projects. What does this tell you about the current state of AI automation?

---

## Bridge to Week 9

Next week, we explore **AI systems in production** — keeping AI working well over time. The evaluation practices from this week become the foundation for:
- **Monitoring**: Running your evaluation suite continuously to detect regressions
- **Feedback loops**: Using real conversation outcomes to improve your task library
- **Drift detection**: Noticing when new user patterns break existing test scenarios

**Preparation**: Think about how you would automate the evaluation pipeline we built today. What would trigger a re-evaluation? How often should you run it?

---

## Additional Resources

### Benchmarks and Leaderboards
- [Scale AI Remote Labor Index](https://scale.com/leaderboard/rli) — Real-world AI agent benchmark
- [RLI Research Paper](https://arxiv.org/abs/2510.26787) — "Remote Labor Index: Measuring AI Automation of Remote Work"
- [Scale AI RLI Blog Post](https://scale.com/blog/rli) — Overview and methodology

### Chatbot Evaluation Frameworks
- [DeepEval Conversation Simulator](https://deepeval.com/docs/conversation-simulator) — Open-source conversation simulation
- [Langfuse Multi-Turn Evaluation](https://langfuse.com/blog/2025-10-09-evaluating-multi-turn-conversations) — Evaluation of multi-turn conversations
- [Confident AI Chatbot Evaluation](https://www.confident-ai.com/blog/llm-chatbot-evaluation-explained-top-chatbot-evaluation-metrics-and-testing-techniques) — Metrics and testing techniques

### From the Data Science Survey
- [Data Science Q25](../../DataScienceSurvey.md#data-science) — ROC curves (see also Week 8)
- [Machine Learning Q32](../../DataScienceSurvey.md#machine-learning) — AUC interpretation (see also Week 8)

### Related Subdiscipline
- [Generative AI](../../GenerativeAI.md) — LLM fundamentals and evaluation
- [Business Analytics](../../BusinessAnalytics.md) — Evaluation in business context

---

[← Week 7: Clustering](../Week07/README.md) | [Week 8: Model Evaluation Deep Dive](../Week08/README.md) | [Back to Course Overview](../DataScienceForFusionEngineers.md) | [Week 9: AI in Production →](../Week09/README.md)
