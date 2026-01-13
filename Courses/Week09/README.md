# Week 9: AI Systems in Production

## Keeping AI Working Well Over Time

This week we explore MLOps—the practices for maintaining AI quality in production. You'll learn to monitor performance, detect degradation, and establish feedback loops for continuous improvement.

---

## Learning Objectives

By the end of this week, you will be able to:

1. Explain core MLOps concepts and why they matter
2. Design monitoring dashboards for AI systems
3. Detect data drift and concept drift
4. Implement feedback loops for model improvement
5. Design human-in-the-loop escalation workflows
6. Create maintenance plans for production AI

---

## Core Concepts

### 1. Why MLOps Matters

AI models degrade over time. Unlike traditional software, ML systems can silently fail.

| Traditional Software | ML Systems |
|---------------------|------------|
| Fails visibly (crashes, errors) | Fails silently (wrong predictions) |
| Bug = code problem | Bug = code OR data OR model problem |
| Fix once, stays fixed | Requires ongoing maintenance |
| Testing verifies correctness | Testing provides estimates |

### 2. The MLOps Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Data ──> Training ──> Deployment ──> Monitoring          │
│     ↑                                       │               │
│     │                                       │               │
│     └───────── Feedback Loop ───────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3. Types of Drift

#### Data Drift (Covariate Shift)
The distribution of input data changes.

**Example**: Support tickets during a holiday season look different from normal times.

```
Training Period:          Production:
- 40% Technical           - 20% Technical
- 30% Billing             - 50% Billing  ← Shift!
- 30% Account             - 30% Account
```

#### Concept Drift
The relationship between inputs and outputs changes.

**Example**: What constitutes an "urgent" ticket changes after policy update.

```
Before Policy Change:     After Policy Change:
"Response time issue"     "Response time issue"
  → Not Urgent             → URGENT ← Relationship changed!
```

#### Label Drift
The distribution of labels/outcomes changes.

**Example**: Escalation rates change due to new product launch.

### 4. Monitoring Strategies

#### What to Monitor

| Category | Metrics | Alert Threshold |
|----------|---------|-----------------|
| **Volume** | Predictions/hour, unique users | ±50% from baseline |
| **Latency** | Response time (p50, p95, p99) | p95 > 500ms |
| **Accuracy** | Precision, Recall, F1 (if labels available) | Drop >5% |
| **Confidence** | Mean confidence, % low-confidence | Confidence drop >10% |
| **Distribution** | Feature distributions, prediction distribution | Statistical tests fail |
| **Errors** | Error rate, error types | Error rate >1% |

#### Monitoring Dashboard Design

```python
# Pseudocode for monitoring metrics collection
class AIMonitor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.metrics_store = MetricsDB()

    def log_prediction(self, features, prediction, confidence, latency):
        """Log every prediction for monitoring"""
        self.metrics_store.insert({
            'timestamp': datetime.now(),
            'model': self.model_name,
            'prediction': prediction,
            'confidence': confidence,
            'latency_ms': latency,
            'feature_hash': hash(features)  # For distribution tracking
        })

    def log_feedback(self, prediction_id, actual_label):
        """Log feedback when ground truth is available"""
        self.metrics_store.update_feedback(prediction_id, actual_label)

    def compute_daily_metrics(self, date):
        """Aggregate daily metrics for dashboards"""
        predictions = self.metrics_store.get_predictions(date)
        return {
            'volume': len(predictions),
            'mean_confidence': predictions['confidence'].mean(),
            'low_confidence_rate': (predictions['confidence'] < 0.6).mean(),
            'latency_p95': predictions['latency_ms'].quantile(0.95),
            # Accuracy metrics if feedback available
        }
```

### 5. Feedback Loops

#### Types of Feedback

| Type | Source | Delay | Reliability |
|------|--------|-------|-------------|
| **Explicit** | User corrections, ratings | Immediate | High |
| **Implicit** | User behavior (click, ignore) | Minutes | Medium |
| **Outcome** | Business metrics (CSAT, resolution) | Hours/Days | High |
| **Expert Review** | QA sampling | Days/Weeks | Very High |

#### Feedback Collection Strategies

```python
class FeedbackCollector:
    """Collect feedback for AI predictions"""

    def collect_explicit_feedback(self, prediction_id, agent_correction):
        """Agent explicitly corrects AI prediction"""
        return {
            'type': 'explicit',
            'prediction_id': prediction_id,
            'original': self.get_prediction(prediction_id),
            'corrected': agent_correction,
            'timestamp': datetime.now()
        }

    def collect_implicit_feedback(self, prediction_id, user_action):
        """Track user actions as implicit feedback"""
        # e.g., Did user accept suggested response or modify it?
        return {
            'type': 'implicit',
            'prediction_id': prediction_id,
            'action': user_action,  # 'accepted', 'modified', 'rejected'
            'timestamp': datetime.now()
        }

    def collect_outcome_feedback(self, ticket_id):
        """Collect business outcomes (delayed feedback)"""
        # e.g., Did the ticket escalate? What was CSAT?
        ticket = self.get_ticket_outcome(ticket_id)
        return {
            'type': 'outcome',
            'ticket_id': ticket_id,
            'escalated': ticket['escalated'],
            'csat_score': ticket['csat'],
            'resolution_time': ticket['resolution_hours']
        }
```

### 6. Human-in-the-Loop Systems

Combining AI efficiency with human judgment for optimal results.

#### Escalation Patterns

```
┌──────────────┐     High Confidence      ┌──────────────┐
│   AI Model   │ ────────────────────────>│ Auto-Process │
└──────────────┘                          └──────────────┘
       │
       │ Low Confidence / High Risk
       │
       ▼
┌──────────────┐     Human Decision       ┌──────────────┐
│ Human Review │ ────────────────────────>│    Output    │
└──────────────┘                          └──────────────┘
       │
       │ Feedback
       │
       ▼
┌──────────────┐
│   Retraining │
└──────────────┘
```

#### Threshold Strategies

| Strategy | When to Route to Human |
|----------|----------------------|
| **Confidence** | Confidence < 0.7 |
| **Risk** | High-value customer OR sensitive topic |
| **Disagreement** | Multiple models disagree |
| **Novelty** | Input is out of distribution |
| **Audit Sample** | Random 5% for quality assurance |

---

## Why This Matters for Fusion Developers

### Your Role in MLOps

As a Fusion Developer, you bridge AI and operations:

1. **Monitor**: Set up and review dashboards
2. **Triage**: Investigate alerts and anomalies
3. **Curate**: Manage feedback data for retraining
4. **Advocate**: Champion AI quality with stakeholders
5. **Iterate**: Propose improvements based on patterns

### Key Questions to Ask

- How would we know if the AI started failing?
- What's our plan when accuracy drops?
- How do we collect feedback systematically?
- What's the retraining cadence?
- Who owns AI quality long-term?

---

## Hands-On Lab: Designing a Monitoring Plan

### Lab Exercise

```python
# Week 9 Lab: AI Monitoring and Drift Detection
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Simulate production data over time
np.random.seed(42)

# Training period distribution
training_data = {
    'confidence': np.random.beta(8, 2, 1000),
    'category_technical': np.random.choice([0, 1], 1000, p=[0.6, 0.4]),
    'resolution_hours': np.random.exponential(4, 1000)
}
training_df = pd.DataFrame(training_data)

# Production period - with drift introduced
production_weeks = []

for week in range(8):
    n = 200

    if week < 4:
        # Normal operation
        data = {
            'week': week + 1,
            'confidence': np.random.beta(8, 2, n),
            'category_technical': np.random.choice([0, 1], n, p=[0.6, 0.4]),
            'accuracy': np.random.choice([0, 1], n, p=[0.12, 0.88])
        }
    else:
        # Drift occurs (new product launch?)
        data = {
            'week': week + 1,
            'confidence': np.random.beta(6, 3, n),  # Lower confidence
            'category_technical': np.random.choice([0, 1], n, p=[0.3, 0.7]),  # More technical
            'accuracy': np.random.choice([0, 1], n, p=[0.20, 0.80])  # Lower accuracy
        }

    production_weeks.append(pd.DataFrame(data))

production_df = pd.concat(production_weeks, ignore_index=True)

# Task 1: Monitor Accuracy Over Time
weekly_metrics = production_df.groupby('week').agg({
    'accuracy': 'mean',
    'confidence': 'mean',
    'category_technical': 'mean'
}).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Accuracy trend
axes[0].plot(weekly_metrics['week'], weekly_metrics['accuracy'], 'bo-', markersize=8)
axes[0].axhline(y=0.85, color='r', linestyle='--', label='Threshold (85%)')
axes[0].axhline(y=training_df['category_technical'].mean() * 0 + 0.88, color='g',
                linestyle='--', label='Training baseline')
axes[0].fill_between(weekly_metrics['week'], 0.80, 0.90, alpha=0.2, color='green')
axes[0].set_xlabel('Week')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy Over Time')
axes[0].legend()

# Confidence trend
axes[1].plot(weekly_metrics['week'], weekly_metrics['confidence'], 'go-', markersize=8)
axes[1].set_xlabel('Week')
axes[1].set_ylabel('Mean Confidence')
axes[1].set_title('Model Confidence Over Time')

# Category distribution (data drift indicator)
axes[2].plot(weekly_metrics['week'], weekly_metrics['category_technical'], 'ro-', markersize=8)
axes[2].axhline(y=0.4, color='blue', linestyle='--', label='Training baseline')
axes[2].set_xlabel('Week')
axes[2].set_ylabel('% Technical Tickets')
axes[2].set_title('Input Distribution (Data Drift)')
axes[2].legend()

plt.tight_layout()
plt.show()

# Task 2: Statistical Drift Detection
print("=== Drift Detection Tests ===\n")

# Compare training vs recent production
recent_production = production_df[production_df['week'] >= 5]

# Test 1: Confidence distribution (KS test)
ks_stat, ks_pval = stats.ks_2samp(training_df['confidence'],
                                  recent_production['confidence'])
print(f"Confidence Distribution (KS Test):")
print(f"  Statistic: {ks_stat:.4f}, p-value: {ks_pval:.4f}")
print(f"  Drift detected: {'YES ⚠️' if ks_pval < 0.05 else 'No'}\n")

# Test 2: Category distribution (Chi-square)
training_tech_rate = training_df['category_technical'].mean()
production_tech_rate = recent_production['category_technical'].mean()
print(f"Technical Ticket Rate:")
print(f"  Training: {training_tech_rate:.1%}")
print(f"  Recent Production: {production_tech_rate:.1%}")
print(f"  Change: {(production_tech_rate - training_tech_rate) * 100:+.1f} percentage points")
print(f"  Drift detected: {'YES ⚠️' if abs(production_tech_rate - training_tech_rate) > 0.1 else 'No'}\n")

# Task 3: Alert Configuration
print("=== Alert Configuration ===\n")

class AlertConfig:
    """Define monitoring alerts"""

    alerts = {
        'accuracy_drop': {
            'metric': 'weekly_accuracy',
            'condition': 'value < 0.85',
            'severity': 'HIGH',
            'action': 'Page on-call, investigate immediately'
        },
        'confidence_drop': {
            'metric': 'mean_confidence',
            'condition': 'value < 0.70',
            'severity': 'MEDIUM',
            'action': 'Review in next standup, check for data drift'
        },
        'distribution_shift': {
            'metric': 'category_technical_rate',
            'condition': 'abs(value - baseline) > 0.10',
            'severity': 'MEDIUM',
            'action': 'Investigate cause, consider retraining'
        },
        'volume_anomaly': {
            'metric': 'daily_predictions',
            'condition': 'value < 0.5 * baseline OR value > 2 * baseline',
            'severity': 'LOW',
            'action': 'Check for upstream issues'
        }
    }

    @classmethod
    def check_alerts(cls, metrics, baselines):
        triggered = []
        for alert_name, config in cls.alerts.items():
            # Simplified check
            triggered.append({
                'alert': alert_name,
                'severity': config['severity'],
                'action': config['action']
            })
        return triggered

for alert_name, config in AlertConfig.alerts.items():
    print(f"Alert: {alert_name}")
    print(f"  Severity: {config['severity']}")
    print(f"  Condition: {config['condition']}")
    print(f"  Action: {config['action']}")
    print()

# Task 4: Design a Feedback Loop
print("=== Feedback Loop Design ===\n")

feedback_plan = """
FEEDBACK COLLECTION PLAN

1. EXPLICIT FEEDBACK (Real-time)
   - Agent correction interface for wrong predictions
   - Thumbs up/down on AI suggestions
   - Collection rate: ~15% of predictions

2. IMPLICIT FEEDBACK (Automated)
   - Track if agents modify AI-suggested responses
   - Track if AI routing is overridden
   - Collection rate: 100% of predictions

3. OUTCOME FEEDBACK (Delayed)
   - Link predictions to ticket resolution metrics
   - Track CSAT for AI-handled vs human-handled
   - Collection rate: 100% (with 24-48hr delay)

4. EXPERT REVIEW (Sampled)
   - Weekly QA review of 50 random predictions
   - Monthly deep-dive on all errors
   - Focus on high-confidence errors

RETRAINING TRIGGERS:
- Accuracy drops below 85% for 2 consecutive weeks
- Data drift detected (KS p-value < 0.01)
- Significant product/process change
- Quarterly scheduled retrain regardless
"""

print(feedback_plan)
```

### Lab Questions

1. At what week does performance start degrading? What metrics indicate this?
2. What type of drift is occurring (data drift, concept drift, or both)?
3. Which alert would trigger first based on this data?
4. How would you use the feedback loop to improve the model?

---

## Creating a Maintenance Plan

### Template: AI System Maintenance Plan

```markdown
# AI System Maintenance Plan
## [System Name]

### Monitoring
| Metric | Frequency | Threshold | Owner |
|--------|-----------|-----------|-------|
| Accuracy | Daily | >85% | [Name] |
| Latency p95 | Hourly | <500ms | [Name] |
| Volume | Hourly | ±50% baseline | [Name] |

### Alerting
| Alert | Severity | Response SLA | Runbook |
|-------|----------|--------------|---------|
| Accuracy drop | P1 | 1 hour | link |
| Drift detected | P2 | 24 hours | link |
| Volume anomaly | P3 | Next business day | link |

### Feedback Collection
- Explicit: Agent corrections via [tool]
- Implicit: Usage analytics via [platform]
- Outcome: CSAT/escalation via [system]

### Retraining Schedule
- Trigger-based: When accuracy < 85% for 2 weeks
- Scheduled: Quarterly (Q1, Q2, Q3, Q4)
- Emergency: Major product/process change

### Escalation Path
1. First responder: [Team/Person]
2. Escalation: [Manager/Lead]
3. Executive: [Director/VP]
```

---

## Knowledge Check

1. Data drift refers to:
   - a) The model becoming slower
   - b) Changes in the distribution of input data
   - c) Changes in model architecture
   - d) Bugs in the code

2. A human-in-the-loop system typically routes to humans when:
   - a) The system is running slowly
   - b) Confidence is low or risk is high
   - c) The data is too large
   - d) The model is being retrained

3. Which feedback type provides the most reliable ground truth?
   - a) Implicit feedback from user clicks
   - b) Explicit expert review
   - c) Automated logging
   - d) User satisfaction surveys

4. How often should you typically retrain a production model?
   - a) Never - once deployed, it's done
   - b) Daily
   - c) When triggered by metrics OR on a regular schedule
   - d) Only when users complain

*(Answers: 1-b, 2-b, 3-b, 4-c)*

---

## Reflection Journal

1. If your AI started performing worse, how quickly would you know? What would you do?

2. What feedback loops exist (or should exist) for AI in your organization?

3. Who owns AI quality in your team? Is that clear?

---

## Bridge to Week 10

Next week is our **Capstone**—putting everything together in a comprehensive project. You'll:
- Conduct an end-to-end AI evaluation
- Create a professional evaluation report
- Build your continued learning roadmap

**Preparation**: Identify an AI system you'd like to evaluate for your capstone project.

---

## Additional Resources

### Related Subdiscipline
- [MLOps](../../MLOps.md) — Deep dive into ML operations

### From the Data Science Survey
- [Miscellaneous Q9-Q11](../../DataScienceSurvey.md#miscellaneous) — Model maintenance
- [Miscellaneous Q24-Q25](../../DataScienceSurvey.md#miscellaneous) — Deployment strategies

### External Resources
- [Azure Machine Learning MLOps](https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-management-and-deployment)
- [Azure ML Monitoring](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-monitor-online-endpoints)
- [Evidently AI (Open Source Monitoring)](https://www.evidentlyai.com/)

---

[← Week 8: Model Evaluation](../Week08/README.md) | [Back to Course Overview](../DataScienceForFusionEngineers.md) | [Week 10: Capstone →](../Week10/README.md)
