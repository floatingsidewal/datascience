# Week 8: Model Evaluation Deep Dive

## Beyond Accuracy—Comprehensive AI Assessment

This week we go beyond basic metrics to learn comprehensive evaluation techniques. You'll understand ROC curves, detect bias in AI systems, and build evaluation frameworks suitable for production AI.

---

## Learning Objectives

By the end of this week, you will be able to:

1. Interpret ROC curves and AUC scores
2. Understand Precision-Recall curves and when to use them
3. Evaluate model calibration and confidence scores
4. Detect bias and fairness issues in AI systems
5. Conduct systematic error analysis
6. Build comprehensive AI evaluation frameworks

---

## Core Concepts

### 1. ROC Curves and AUC

**ROC** = Receiver Operating Characteristic
**AUC** = Area Under the Curve

The ROC curve plots True Positive Rate vs. False Positive Rate at all possible classification thresholds.

```
TPR (Sensitivity)
    │
1.0 ├───────────────────▲
    │               ▲▲▲
    │           ▲▲▲▲
    │       ▲▲▲▲  ← Your model
    │   ▲▲▲▲
0.5 ├──────────────────
    │  ▲▲
    │▲▲ ← Random classifier (diagonal)
    │
0.0 └───────────────────
    0.0       0.5       1.0
              FPR (1-Specificity)
```

#### Interpreting AUC

| AUC Range | Interpretation |
|-----------|----------------|
| 0.90-1.00 | Excellent |
| 0.80-0.90 | Good |
| 0.70-0.80 | Fair |
| 0.60-0.70 | Poor |
| 0.50-0.60 | Fail (no better than random) |

#### Why AUC Matters

- **Threshold-independent**: Evaluates model across all thresholds
- **Comparable**: Can compare different models fairly
- **Interpretable**: Probability that model ranks a random positive higher than a random negative

### 2. Precision-Recall Curves

Better than ROC when classes are imbalanced (rare events like escalations).

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
avg_precision = average_precision_score(y_true, y_scores)

plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AP={avg_precision:.2f})')
```

#### When to Use Which

| Scenario | Preferred Metric |
|----------|------------------|
| Balanced classes | ROC AUC |
| Imbalanced classes | PR AUC |
| Cost of FP ≈ cost of FN | ROC AUC |
| Very different costs | PR AUC with threshold tuning |

### 3. Calibration: Can You Trust Confidence Scores?

A well-calibrated model's confidence scores reflect true probabilities.

**Calibrated**: When model says "80% confident," it's correct ~80% of the time
**Uncalibrated**: Confidence scores don't match actual accuracy

```python
from sklearn.calibration import calibration_curve

# Compute calibration
prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10)

# Plot reliability diagram
plt.plot(prob_pred, prob_true, 'o-', label='Model')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Plot')
plt.legend()
```

### 4. Bias and Fairness

#### Types of Bias in AI

| Type | Description | Example |
|------|-------------|---------|
| **Sampling Bias** | Training data not representative | Mostly English tickets, poor performance on Spanish |
| **Historical Bias** | Past decisions were biased | If past escalations favored enterprise, AI learns this |
| **Measurement Bias** | Features proxy for protected attributes | Zip code proxying for race/income |
| **Aggregation Bias** | Model works overall but fails for subgroups | Good average accuracy, poor for specific regions |

#### Fairness Metrics

| Metric | Definition | Question |
|--------|------------|----------|
| **Demographic Parity** | Equal positive rates across groups | Does AI escalate equally for all customer segments? |
| **Equal Opportunity** | Equal TPR across groups | When escalation is needed, is AI equally good at catching it? |
| **Calibration by Group** | Equal calibration across groups | When AI says 80% confident, is that accurate for all segments? |

### 5. Error Analysis

Systematic investigation of where and why the model fails.

#### Error Categories

1. **Random Errors**: Scattered, no pattern
2. **Systematic Errors**: Consistent patterns (e.g., always misses urgent billing issues)
3. **Edge Cases**: Unusual inputs the model hasn't seen

#### Error Analysis Process

```
1. Collect all predictions
2. Identify misclassifications
3. Categorize by:
   - True label
   - Predicted label
   - Confidence level
   - Customer segment
   - Ticket characteristics
4. Look for patterns
5. Prioritize fixes based on impact
```

---

## Why This Matters for AI

### The Full Evaluation Picture

A production AI system needs evaluation across multiple dimensions:

| Dimension | Questions to Answer |
|-----------|---------------------|
| **Accuracy** | How often is it right? |
| **Discrimination** | Can it separate classes? (AUC) |
| **Calibration** | Can we trust confidence scores? |
| **Fairness** | Does it work equally for all groups? |
| **Robustness** | Does it handle edge cases? |
| **Efficiency** | Is it fast enough for production? |

### Building an Evaluation Framework

```python
class AIEvaluationFramework:
    """Comprehensive evaluation for support AI"""

    def __init__(self, y_true, y_pred, y_scores, segments=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_scores = y_scores
        self.segments = segments

    def basic_metrics(self):
        """Accuracy, Precision, Recall, F1"""
        pass

    def discrimination_metrics(self):
        """ROC AUC, PR AUC"""
        pass

    def calibration_analysis(self):
        """Reliability diagram, expected calibration error"""
        pass

    def fairness_analysis(self):
        """Per-segment performance comparison"""
        pass

    def error_analysis(self):
        """Categorize and analyze errors"""
        pass

    def generate_report(self):
        """Comprehensive evaluation report"""
        pass
```

---

## Hands-On Lab: Comprehensive Model Evaluation

### Lab Exercise

```python
# Week 8 Lab: Comprehensive AI Evaluation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Generate sample data with customer segments
np.random.seed(42)
n_samples = 1000

# Features
X = np.random.randn(n_samples, 5)

# True labels (some tickets escalate)
y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5 > 1).astype(int)

# Customer segments (for fairness analysis)
segments = np.random.choice(['Enterprise', 'SMB', 'Consumer'],
                           n_samples, p=[0.2, 0.3, 0.5])

# Introduce some bias: Enterprise customers more likely to be labeled as escalation
y[segments == 'Enterprise'] = np.where(
    np.random.rand((segments == 'Enterprise').sum()) < 0.3,
    1, y[segments == 'Enterprise']
)

# Train/test split
X_train, X_test, y_train, y_test, seg_train, seg_test = train_test_split(
    X, y, segments, test_size=0.3, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Get predictions
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1]

# Task 1: Basic Metrics
print("=== Basic Metrics ===")
print(classification_report(y_test, y_pred))

# Task 2: ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = roc_auc_score(y_test, y_scores)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Task 3: Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_scores)
avg_precision = average_precision_score(y_test, y_scores)

plt.subplot(1, 3, 2)
plt.plot(recall, precision, 'g-', label=f'PR (AP = {avg_precision:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

# Task 4: Calibration Plot
prob_true, prob_pred = calibration_curve(y_test, y_scores, n_bins=10)

plt.subplot(1, 3, 3)
plt.plot(prob_pred, prob_true, 'o-', label='Model')
plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Plot')
plt.legend()

plt.tight_layout()
plt.show()

# Task 5: Fairness Analysis - Performance by Segment
print("\n=== Fairness Analysis ===")
print(f"{'Segment':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'AUC':<10} {'Count':<10}")
print("-" * 62)

for segment in ['Enterprise', 'SMB', 'Consumer']:
    mask = seg_test == segment

    if mask.sum() > 0:
        seg_accuracy = (y_pred[mask] == y_test[mask]).mean()
        seg_precision = (y_pred[mask] & y_test[mask]).sum() / max(y_pred[mask].sum(), 1)
        seg_recall = (y_pred[mask] & y_test[mask]).sum() / max(y_test[mask].sum(), 1)

        if len(np.unique(y_test[mask])) > 1:
            seg_auc = roc_auc_score(y_test[mask], y_scores[mask])
        else:
            seg_auc = np.nan

        print(f"{segment:<12} {seg_accuracy:<10.3f} {seg_precision:<10.3f} {seg_recall:<10.3f} {seg_auc:<10.3f} {mask.sum():<10}")

# Task 6: Error Analysis
print("\n=== Error Analysis ===")

# Create error dataframe
errors_df = pd.DataFrame({
    'true': y_test,
    'pred': y_pred,
    'score': y_scores,
    'segment': seg_test
})
errors_df['correct'] = errors_df['true'] == errors_df['pred']
errors_df['error_type'] = np.where(
    errors_df['correct'], 'Correct',
    np.where(errors_df['true'] == 1, 'False Negative', 'False Positive')
)

# Error breakdown
print("\nError Type Distribution:")
print(errors_df['error_type'].value_counts())

# Confidence analysis for errors
print("\nMean Confidence by Error Type:")
confidence_analysis = errors_df.groupby('error_type')['score'].agg(['mean', 'std', 'count'])
print(confidence_analysis)

# High-confidence errors (most concerning!)
high_conf_errors = errors_df[
    (~errors_df['correct']) &
    ((errors_df['score'] > 0.8) | (errors_df['score'] < 0.2))
]
print(f"\nHigh-confidence errors: {len(high_conf_errors)} ({len(high_conf_errors)/len(errors_df)*100:.1f}%)")

# Task 7: Fairness Comparison Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy by segment
segment_accuracy = errors_df.groupby('segment')['correct'].mean()
axes[0].bar(segment_accuracy.index, segment_accuracy.values)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy by Customer Segment')
axes[0].axhline(y=errors_df['correct'].mean(), color='r', linestyle='--', label='Overall')
axes[0].legend()

# False Negative rate by segment (missing escalations)
fn_by_segment = errors_df[errors_df['true'] == 1].groupby('segment').apply(
    lambda x: (x['error_type'] == 'False Negative').mean()
)
axes[1].bar(fn_by_segment.index, fn_by_segment.values, color='orange')
axes[1].set_ylabel('False Negative Rate')
axes[1].set_title('Missed Escalations by Segment')

plt.tight_layout()
plt.show()

print("\n=== Summary ===")
print(f"Overall AUC: {roc_auc:.3f}")
print(f"Highest segment AUC: {segment} - Enterprise? SMB? Consumer?")
print(f"Fairness concern: Does any segment have significantly worse performance?")
```

### Lab Questions

1. What does the ROC AUC score tell you about this model's discriminative ability?
2. Looking at the calibration plot, is the model well-calibrated?
3. Which customer segment has the worst performance? What might cause this?
4. What percentage of errors are "high-confidence errors"? Why are these particularly concerning?

---

## Building an Evaluation Report

### Report Template

```markdown
# AI Model Evaluation Report

## Executive Summary
- Model: [Name/Version]
- Purpose: [e.g., Escalation Prediction]
- Recommendation: [Ship / Needs Work / Reject]

## Performance Metrics
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Accuracy | X% | >85% | ✓/✗ |
| AUC | X.XX | >0.80 | ✓/✗ |
| Precision | X% | >75% | ✓/✗ |
| Recall | X% | >80% | ✓/✗ |

## Fairness Analysis
| Segment | AUC | Accuracy | Gap from Overall |
|---------|-----|----------|------------------|
| Enterprise | X.XX | X% | +/-X% |
| SMB | X.XX | X% | +/-X% |
| Consumer | X.XX | X% | +/-X% |

## Error Analysis
- Total errors: N (X%)
- High-confidence errors: N (concerning)
- Most common error patterns:
  1. [Pattern 1]
  2. [Pattern 2]

## Recommendations
1. [Action item 1]
2. [Action item 2]
```

---

## Knowledge Check

1. An AUC of 0.5 indicates:
   - a) Perfect model
   - b) Model performs no better than random
   - c) Model needs more training
   - d) Data is imbalanced

2. A well-calibrated model:
   - a) Always predicts the majority class
   - b) Has confidence scores that reflect true probabilities
   - c) Has the highest accuracy
   - d) Runs fastest

3. "Demographic parity" in fairness means:
   - a) Equal accuracy across groups
   - b) Equal positive prediction rates across groups
   - c) Equal sample sizes across groups
   - d) Equal features across groups

4. High-confidence errors are concerning because:
   - a) They happen frequently
   - b) They indicate the model is "confidently wrong"
   - c) They're easy to fix
   - d) They only affect small customers

*(Answers: 1-b, 2-b, 3-b, 4-b)*

---

## Reflection Journal

1. How would you evaluate fairness for your support AI? What segments matter most?

2. What would be acceptable AUC and accuracy thresholds for your use case?

3. How would you communicate evaluation results to non-technical stakeholders?

---

## Bridge to Week 9

Next week, we explore **AI systems in production**—keeping AI working well over time. You'll learn:
- MLOps basics for ongoing AI management
- Monitoring for performance degradation
- Setting up feedback loops for continuous improvement

**Preparation**: Think about how you would know if your AI started performing worse. What signals would you look for?

---

## Additional Resources

### From the Data Science Survey
- [Data Science Q25](../../DataScienceSurvey.md#data-science) — ROC curves
- [Machine Learning Q32](../../DataScienceSurvey.md#machine-learning) — AUC interpretation

### Related Subdiscipline
- [Business Analytics](../../BusinessAnalytics.md) — Evaluation in business context

### External Resources
- [Google's ML Fairness](https://developers.google.com/machine-learning/fairness-overview)
- [Scikit-learn Calibration](https://scikit-learn.org/stable/modules/calibration.html)

---

[← Week 7: Clustering](../Week07/README.md) | [Back to Course Overview](../DataScienceForFusionEngineers.md) | [Week 9: AI in Production →](../Week09/README.md)
