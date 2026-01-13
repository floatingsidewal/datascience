# Week 4: Classification & Confusion

## Understanding How AI Makes (and Misses) Predictions

This week we dive into classification—the backbone of most AI systems in support. You'll learn to read confusion matrices, understand when different metrics matter, and set appropriate thresholds for AI decisions.

---

## Learning Objectives

By the end of this week, you will be able to:

1. Explain binary and multi-class classification
2. Construct and interpret a confusion matrix
3. Calculate precision, recall, F1-score, and accuracy
4. Determine which metric matters most for different scenarios
5. Adjust classification thresholds to optimize outcomes
6. Evaluate chatbot and routing AI performance

---

## Core Concepts

### 1. What is Classification?

Classification is predicting a category or label from input data.

**Binary Classification**: Two possible outcomes
- Spam / Not Spam
- Escalate / Don't Escalate
- Urgent / Not Urgent

**Multi-class Classification**: Multiple possible outcomes
- Ticket Category: [Technical, Billing, Account, Feature Request]
- Sentiment: [Positive, Neutral, Negative]
- Intent: [Question, Complaint, Request, Feedback]

### 2. The Confusion Matrix

A table that describes the performance of a classification model.

#### Binary Classification Matrix

|  | Predicted: Positive | Predicted: Negative |
|--|---------------------|---------------------|
| **Actual: Positive** | True Positive (TP) | False Negative (FN) |
| **Actual: Negative** | False Positive (FP) | True Negative (TN) |

#### Support Example: Escalation Prediction

|  | Predicted: Escalate | Predicted: Don't Escalate |
|--|---------------------|---------------------------|
| **Actually Escalated** | TP: 45 | FN: 15 |
| **Didn't Escalate** | FP: 10 | TN: 130 |

**Reading this matrix**:
- 45 tickets correctly predicted to escalate
- 15 escalations the AI missed (dangerous!)
- 10 tickets flagged for escalation that didn't need it (annoying but safer)
- 130 correctly predicted not to escalate

### 3. Key Metrics

| Metric | Formula | Question It Answers |
|--------|---------|---------------------|
| **Accuracy** | (TP + TN) / Total | What % of predictions were correct? |
| **Precision** | TP / (TP + FP) | When AI says "yes", how often is it right? |
| **Recall** | TP / (TP + FN) | Of actual positives, how many did AI catch? |
| **F1-Score** | 2 × (P × R) / (P + R) | Balanced measure of precision and recall |
| **Specificity** | TN / (TN + FP) | Of actual negatives, how many did AI correctly identify? |

#### Calculating from Our Example

```
Total = 45 + 15 + 10 + 130 = 200

Accuracy = (45 + 130) / 200 = 87.5%
Precision = 45 / (45 + 10) = 81.8%
Recall = 45 / (45 + 15) = 75.0%
F1-Score = 2 × (0.818 × 0.75) / (0.818 + 0.75) = 78.3%
```

### 4. When Each Metric Matters

| Scenario | Priority Metric | Reasoning |
|----------|-----------------|-----------|
| **Fraud Detection** | Recall | Missing fraud is costly; false alarms are acceptable |
| **Spam Filtering** | Precision | False positives (blocking real email) are very costly |
| **Medical Diagnosis** | Recall | Missing a disease could be fatal |
| **Support Escalation** | Recall | Missing a true escalation damages customer trust |
| **Auto-Response Eligibility** | Precision | Sending wrong auto-response is worse than not sending |

### 5. The Precision-Recall Trade-off

Most classifiers output a **confidence score** (0-1), and you choose a **threshold**.

- **Lower threshold** → More positives → Higher recall, lower precision
- **Higher threshold** → Fewer positives → Higher precision, lower recall

```
Example: "Should this ticket get auto-response?"

Threshold = 0.5:
  - AI responds to more tickets (higher recall)
  - But makes more mistakes (lower precision)

Threshold = 0.9:
  - AI only responds when very confident (higher precision)
  - But misses many opportunities (lower recall)
```

---

## Why This Matters for AI

### Evaluating Chatbot Performance

When your AI chatbot handles customer inquiries, classification happens at multiple levels:

1. **Intent Classification**: What does the customer want?
2. **Entity Extraction**: What specific things are mentioned?
3. **Routing Decision**: Which team/queue should handle this?
4. **Auto-resolution Eligibility**: Can AI fully resolve this?

Each level needs appropriate metrics based on the cost of errors.

### Real Scenario: Intent Classification

Your chatbot classifies intents into: [Password Reset, Billing Question, Technical Issue, Cancel Account]

**Confusion Matrix for "Cancel Account" Intent**:

|  | Predicted: Cancel | Predicted: Other |
|--|-------------------|------------------|
| **Actual: Cancel** | 18 | 7 |
| **Actual: Other** | 3 | 472 |

**Analysis**:
- Recall = 18/25 = 72% — AI misses 28% of cancellation requests!
- Precision = 18/21 = 86% — When AI flags cancellation, it's usually right

**Business Impact**: Missing cancellation requests means customers who want to cancel go through the wrong flow, likely increasing frustration and churn.

**Recommendation**: Lower the threshold for "Cancel Account" to improve recall, even if it means some false positives (which just route to a human faster).

---

## Hands-On Lab: Building a Confusion Matrix

### Lab Exercise

```python
# Week 4 Lab: Classification Metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve

# Simulate ticket classification data
np.random.seed(42)

# Actual labels (ground truth)
n_samples = 500
actual_urgent = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])  # 15% are actually urgent

# AI predictions (with some errors)
# The AI has 80% accuracy overall but makes specific kinds of mistakes
predicted_scores = np.where(
    actual_urgent == 1,
    np.random.beta(7, 3, n_samples),  # High scores for actual urgent
    np.random.beta(2, 8, n_samples)   # Low scores for non-urgent
)

# Task 1: Apply different thresholds
thresholds = [0.3, 0.5, 0.7]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, threshold in enumerate(thresholds):
    predicted_urgent = (predicted_scores >= threshold).astype(int)
    cm = confusion_matrix(actual_urgent, predicted_urgent)

    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Not Urgent', 'Urgent'],
                yticklabels=['Not Urgent', 'Urgent'])
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_title(f'Threshold: {threshold}\nP={precision:.2f}, R={recall:.2f}, F1={f1:.2f}')

plt.tight_layout()
plt.show()

# Task 2: Precision-Recall Curve
precision_curve, recall_curve, thresholds_curve = precision_recall_curve(actual_urgent, predicted_scores)

plt.figure(figsize=(8, 5))
plt.plot(recall_curve, precision_curve, 'b-', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve: Urgency Detection')
plt.grid(True, alpha=0.3)

# Mark our three thresholds
for threshold in thresholds:
    predicted = (predicted_scores >= threshold).astype(int)
    cm = confusion_matrix(actual_urgent, predicted)
    tn, fp, fn, tp = cm.ravel()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    plt.scatter(r, p, s=100, zorder=5, label=f'Threshold {threshold}')

plt.legend()
plt.show()

# Task 3: Multi-class Classification Report
categories = ['Technical', 'Billing', 'Account', 'Feature Request']
n_multi = 300

# Simulate multi-class data
actual_category = np.random.choice(categories, n_multi, p=[0.4, 0.25, 0.2, 0.15])
predicted_category = actual_category.copy()

# Introduce some errors
error_mask = np.random.random(n_multi) < 0.15  # 15% error rate
predicted_category[error_mask] = np.random.choice(categories, error_mask.sum())

print("\n=== Multi-class Classification Report ===")
print(classification_report(actual_category, predicted_category, target_names=categories))

# Confusion matrix for multi-class
plt.figure(figsize=(8, 6))
cm_multi = confusion_matrix(actual_category, predicted_category, labels=categories)
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues',
            xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted Category')
plt.ylabel('Actual Category')
plt.title('Multi-class Confusion Matrix: Ticket Routing')
plt.tight_layout()
plt.show()
```

### Lab Questions

1. At threshold 0.3, what is the recall? What about at 0.7?
2. For an urgency detection system, which threshold would you recommend? Why?
3. In the multi-class matrix, which category has the most confusion with others?
4. If "Account" tickets often get misrouted, what could be done to improve this?

---

## Setting Thresholds in Production

### The Business Decision Framework

When deploying AI classification, ask:

1. **What is the cost of a false positive?**
   - Customer frustration, wasted agent time, over-escalation

2. **What is the cost of a false negative?**
   - Missed urgency, customer churn, SLA breach

3. **What is the volume at different thresholds?**
   - Too many positives overwhelms the system
   - Too few positives misses the value of AI

### Threshold Selection Process

```python
# Analyze business impact at different thresholds
def analyze_threshold(threshold, predicted_scores, actual_labels,
                      cost_false_positive=10, cost_false_negative=100):
    """
    Analyze business impact of a classification threshold
    """
    predictions = (predicted_scores >= threshold).astype(int)
    cm = confusion_matrix(actual_labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    total_cost = (fp * cost_false_positive) + (fn * cost_false_negative)

    return {
        'threshold': threshold,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'total_cost': total_cost
    }

# Find optimal threshold
results = []
for t in np.arange(0.1, 0.9, 0.05):
    results.append(analyze_threshold(t, predicted_scores, actual_urgent))

results_df = pd.DataFrame(results)
optimal_threshold = results_df.loc[results_df['total_cost'].idxmin(), 'threshold']
print(f"Optimal threshold based on costs: {optimal_threshold}")
```

---

## Knowledge Check

1. In a confusion matrix, a False Negative means:
   - a) AI predicted positive, but it was negative
   - b) AI predicted negative, but it was positive
   - c) AI correctly predicted negative
   - d) AI correctly predicted positive

2. If you want to ensure you catch all urgent tickets (even at the cost of some false alarms), you should prioritize:
   - a) Accuracy
   - b) Precision
   - c) Recall
   - d) Specificity

3. An F1-score of 0.80 indicates:
   - a) The model is 80% accurate
   - b) The model has balanced precision and recall around 80%
   - c) 80% of predictions are positive
   - d) The model catches 80% of positive cases

4. Lowering the classification threshold will generally:
   - a) Increase precision, decrease recall
   - b) Decrease precision, increase recall
   - c) Increase both precision and recall
   - d) Decrease both precision and recall

*(Answers: 1-b, 2-c, 3-b, 4-b)*

---

## Reflection Journal

1. Think of a classification task at your organization. What are the costs of false positives vs. false negatives?

2. If you were evaluating a vendor's chatbot, what metrics would you ask for? What thresholds would you consider acceptable?

3. How might you explain precision vs. recall to a non-technical stakeholder?

---

## Bridge to Week 5

Next week, we explore **text and language processing**—how AI understands human language. You'll learn:
- How text is converted to numbers for AI
- What TF-IDF and embeddings mean
- Why similar questions sometimes get different AI responses

**Preparation**: Collect 10 examples of customer messages that you think should be handled similarly. We'll analyze what makes them "similar" to an AI.

---

## Additional Resources

### From the Data Science Survey
- [Data Science Q4](../../DataScienceSurvey.md#data-science) — Confusion matrix deep dive
- [Data Science Q25](../../DataScienceSurvey.md#data-science) — ROC curves
- [Machine Learning Q24](../../DataScienceSurvey.md#machine-learning) — F1-Score

### External Resources
- [Scikit-learn Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Microsoft Learn - Classification](https://learn.microsoft.com/en-us/training/modules/understand-classification-machine-learning/)
- [Azure ML - Evaluate Models](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-understand-automated-ml)

---

[← Week 3: Statistical Foundations](../Week03/README.md) | [Back to Course Overview](../DataScienceForFusionEngineers.md) | [Week 5: Text & Language Basics →](../Week05/README.md)
