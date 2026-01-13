# Week 10: Capstone & Future Learning

## Putting It All Together

Welcome to your final week! You'll apply everything you've learned in a comprehensive capstone project and create a roadmap for continued growth as a Fusion Developer with data science expertise.

---

## Learning Objectives

By the end of this week, you will be able to:

1. Conduct an end-to-end AI system evaluation
2. Communicate findings effectively to stakeholders
3. Make data-driven recommendations for AI improvements
4. Identify personal learning goals and paths
5. Connect with resources for continued growth

---

## Capstone Project

### Overview

You will evaluate an AI system (real or simulated) and produce a professional evaluation report. This project synthesizes skills from all previous weeks.

### Project Requirements

Your capstone must demonstrate competency in:

| Week | Skill | How to Demonstrate |
|------|-------|-------------------|
| 1-2 | Data exploration | Explore and visualize the evaluation data |
| 3 | Statistics | Apply appropriate statistical tests |
| 4 | Classification metrics | Calculate and interpret precision, recall, F1, etc. |
| 5 | NLP understanding | Analyze text-based inputs or outputs |
| 6 | ML concepts | Discuss training/test methodology |
| 7 | Clustering | Group errors or discover patterns |
| 8 | Comprehensive evaluation | Use ROC/AUC, calibration, fairness analysis |
| 9 | Production considerations | Propose monitoring and maintenance plans |

### Deliverables

1. **Evaluation Report** (primary deliverable)
2. **Code/Notebook** with analysis
3. **Executive Summary** (1-page)
4. **Recommendations Presentation** (5 slides)

---

## Capstone Project Guide

### Option A: Evaluate Provided Simulation

Use the provided simulated AI system data.

```python
# Capstone Project: AI System Evaluation
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# PART 1: Load and Explore Data
# ============================================

np.random.seed(42)
n_samples = 2000

# Simulate AI predictions for ticket routing
data = {
    'ticket_id': range(1, n_samples + 1),
    'ticket_text': [f"Sample ticket text {i}" for i in range(n_samples)],
    'true_category': np.random.choice(
        ['Technical', 'Billing', 'Account', 'Feature Request'],
        n_samples, p=[0.35, 0.30, 0.25, 0.10]
    ),
    'customer_segment': np.random.choice(
        ['Enterprise', 'SMB', 'Consumer'],
        n_samples, p=[0.20, 0.35, 0.45]
    ),
    'ai_confidence': np.random.beta(7, 2, n_samples),
    'timestamp': pd.date_range('2025-01-01', periods=n_samples, freq='30min')
}

df = pd.DataFrame(data)

# Simulate AI predictions (mostly correct, some errors)
def simulate_prediction(row):
    if np.random.random() < 0.82:  # 82% overall accuracy
        return row['true_category']
    else:
        categories = ['Technical', 'Billing', 'Account', 'Feature Request']
        categories.remove(row['true_category'])
        return np.random.choice(categories)

df['ai_prediction'] = df.apply(simulate_prediction, axis=1)
df['correct'] = df['true_category'] == df['ai_prediction']

# Introduce systematic bias: worse performance for Consumer segment
consumer_mask = df['customer_segment'] == 'Consumer'
flip_mask = consumer_mask & (np.random.random(n_samples) < 0.10)
wrong_categories = ['Technical', 'Billing', 'Account', 'Feature Request']
df.loc[flip_mask, 'ai_prediction'] = df.loc[flip_mask, 'true_category'].apply(
    lambda x: np.random.choice([c for c in wrong_categories if c != x])
)
df['correct'] = df['true_category'] == df['ai_prediction']

print("=== Dataset Overview ===")
print(f"Total tickets: {len(df)}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"\nCategory distribution:")
print(df['true_category'].value_counts())
print(f"\nSegment distribution:")
print(df['customer_segment'].value_counts())

# ============================================
# PART 2: Performance Analysis
# ============================================

print("\n=== Overall Performance ===")
overall_accuracy = df['correct'].mean()
print(f"Overall Accuracy: {overall_accuracy:.1%}")

print("\n=== Classification Report ===")
print(classification_report(df['true_category'], df['ai_prediction']))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(df['true_category'], df['ai_prediction'],
                      labels=['Technical', 'Billing', 'Account', 'Feature Request'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Technical', 'Billing', 'Account', 'Feature Request'],
            yticklabels=['Technical', 'Billing', 'Account', 'Feature Request'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Ticket Routing AI')
plt.tight_layout()
plt.show()

# ============================================
# PART 3: Fairness Analysis
# ============================================

print("\n=== Fairness Analysis by Customer Segment ===")
fairness_results = df.groupby('customer_segment').agg({
    'correct': ['mean', 'count'],
    'ai_confidence': 'mean'
}).round(3)
fairness_results.columns = ['Accuracy', 'Count', 'Avg Confidence']
print(fairness_results)

# Visualize fairness
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

segment_accuracy = df.groupby('customer_segment')['correct'].mean()
colors = ['green' if acc > overall_accuracy * 0.95 else 'orange'
          if acc > overall_accuracy * 0.90 else 'red'
          for acc in segment_accuracy]
axes[0].bar(segment_accuracy.index, segment_accuracy.values, color=colors)
axes[0].axhline(y=overall_accuracy, color='blue', linestyle='--', label='Overall')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy by Customer Segment')
axes[0].legend()

# Error rate by segment
error_rate = df.groupby('customer_segment')['correct'].apply(lambda x: 1 - x.mean())
axes[1].bar(error_rate.index, error_rate.values, color='red', alpha=0.7)
axes[1].set_ylabel('Error Rate')
axes[1].set_title('Error Rate by Customer Segment')

plt.tight_layout()
plt.show()

# ============================================
# PART 4: Calibration Analysis
# ============================================

# For binary calibration, pick one category
binary_true = (df['true_category'] == 'Technical').astype(int)
binary_pred_correct = (df['ai_prediction'] == 'Technical').astype(int)

# Use confidence as proxy for probability
prob_true, prob_pred = calibration_curve(binary_true, df['ai_confidence'], n_bins=10)

plt.figure(figsize=(6, 5))
plt.plot(prob_pred, prob_true, 'o-', label='Model')
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
plt.xlabel('Mean Predicted Confidence')
plt.ylabel('Fraction Correct')
plt.title('Calibration Analysis')
plt.legend()
plt.show()

# ============================================
# PART 5: Error Analysis
# ============================================

errors = df[~df['correct']].copy()

print("\n=== Error Analysis ===")
print(f"Total errors: {len(errors)} ({len(errors)/len(df)*100:.1f}%)")

print("\nMost common misclassifications:")
error_pairs = errors.groupby(['true_category', 'ai_prediction']).size().sort_values(ascending=False)
print(error_pairs.head(10))

# High confidence errors
high_conf_errors = errors[errors['ai_confidence'] > 0.8]
print(f"\nHigh-confidence errors (>80%): {len(high_conf_errors)} ({len(high_conf_errors)/len(errors)*100:.1f}% of errors)")

# ============================================
# PART 6: Time-based Analysis
# ============================================

df['date'] = df['timestamp'].dt.date
daily_accuracy = df.groupby('date')['correct'].mean()

plt.figure(figsize=(12, 4))
plt.plot(daily_accuracy.index, daily_accuracy.values, 'b-', alpha=0.7)
plt.axhline(y=overall_accuracy, color='r', linestyle='--', label='Overall Average')
plt.fill_between(daily_accuracy.index, overall_accuracy * 0.95, overall_accuracy * 1.05,
                 alpha=0.2, color='green', label='±5% band')
plt.xlabel('Date')
plt.ylabel('Daily Accuracy')
plt.title('Model Performance Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ============================================
# PART 7: Recommendations Summary
# ============================================

print("\n" + "="*60)
print("CAPSTONE FINDINGS SUMMARY")
print("="*60)

print(f"""
1. OVERALL PERFORMANCE
   - Accuracy: {overall_accuracy:.1%}
   - Strongest category: [Identify from classification report]
   - Weakest category: [Identify from classification report]

2. FAIRNESS CONCERNS
   - Consumer segment underperforms by {(overall_accuracy - df[df['customer_segment']=='Consumer']['correct'].mean())*100:.1f} percentage points
   - Recommendation: Investigate training data representation

3. CALIBRATION
   - Model appears [well/poorly] calibrated
   - Confidence scores [can/cannot] be trusted

4. ERROR PATTERNS
   - {len(high_conf_errors)} high-confidence errors require investigation
   - Most common confusion: [Identify from error_pairs]

5. STABILITY
   - Performance [is/is not] stable over time
   - [Any concerning trends?]

RECOMMENDATIONS:
1. Address Consumer segment performance gap
2. Review training data for underperforming categories
3. Implement monitoring for [specific metrics]
4. Consider retraining with [specific improvements]
""")
```

### Option B: Evaluate Your Own AI System

If you have access to a real AI system, use it for your capstone:

1. Export prediction data with ground truth labels
2. Include customer segment or other demographic data
3. Get at least 500 predictions for statistical validity

---

## Report Template

### Executive Summary (1 page)

```markdown
# AI Evaluation Report: [System Name]
## Executive Summary

**System Evaluated**: [Name, version, purpose]
**Evaluation Period**: [Date range]
**Sample Size**: [N predictions]

### Key Findings

| Metric | Value | Status |
|--------|-------|--------|
| Overall Accuracy | X% | ✓/⚠️/✗ |
| AUC | X.XX | ✓/⚠️/✗ |
| Fairness Gap | X% | ✓/⚠️/✗ |

### Top Recommendations
1. [Most critical action]
2. [Second priority]
3. [Third priority]

### Decision
☐ Ready for production
☐ Ready with conditions
☐ Needs improvement before deployment
```

### Full Report Structure

1. **Introduction** (0.5 page)
   - System background
   - Evaluation objectives
   - Methodology overview

2. **Data Overview** (1 page)
   - Dataset description
   - Sample characteristics
   - Exploratory visualizations

3. **Performance Analysis** (2 pages)
   - Overall metrics
   - Per-class performance
   - Confusion matrix analysis

4. **Fairness Analysis** (1 page)
   - Segment-level performance
   - Disparity identification
   - Root cause hypotheses

5. **Error Analysis** (1 page)
   - Error patterns
   - High-confidence errors
   - Representative examples

6. **Calibration & Confidence** (0.5 page)
   - Calibration plot
   - Reliability assessment

7. **Production Readiness** (1 page)
   - Monitoring recommendations
   - Maintenance plan
   - Feedback loop design

8. **Recommendations** (1 page)
   - Prioritized action items
   - Timeline
   - Success metrics

---

## Continued Learning Roadmap

### Skill Development Paths

Based on your interests, choose a specialization path:

#### Path A: ML Engineering Focus
Build deeper technical skills to work closely with data science teams.

**Next Steps**:
1. Complete Python intermediate course
2. Learn scikit-learn in depth
3. Study MLOps fundamentals
4. Practice with Kaggle competitions

**Resources**:
- [Fast.ai Practical Deep Learning](https://www.fast.ai/)
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/)
- [Made With ML](https://madewithml.com/)

#### Path B: Analytics Leadership Focus
Develop skills to lead AI initiatives and communicate with stakeholders.

**Next Steps**:
1. Study A/B testing methodology deeply
2. Learn data storytelling
3. Understand AI ethics frameworks
4. Practice executive communication

**Resources**:
- [Trustworthy Online Controlled Experiments](https://experimentguide.com/)
- [Storytelling with Data](https://www.storytellingwithdata.com/)
- [AI Ethics courses on Coursera](https://www.coursera.org/)

#### Path C: Domain Expert Focus
Combine data science with deep support domain expertise.

**Next Steps**:
1. Study support analytics best practices
2. Learn about conversational AI design
3. Understand customer experience metrics
4. Explore industry certifications

**Resources**:
- TSIA (Technology Services Industry Association)
- HDI (Help Desk Institute)
- Customer experience certifications

### Learning Plan Template

```markdown
# My Data Science Learning Plan

## 6-Month Goals
1. [Specific goal 1]
2. [Specific goal 2]
3. [Specific goal 3]

## Weekly Commitment
- Hours per week: [X]
- Learning time: [When]
- Practice project: [What]

## Monthly Milestones
| Month | Focus Area | Deliverable |
|-------|------------|-------------|
| 1 | [Topic] | [Output] |
| 2 | [Topic] | [Output] |
| ... | ... | ... |

## Resources
- Primary course: [Name]
- Practice platform: [Name]
- Community: [Name]

## Accountability
- Learning partner: [Name]
- Check-in frequency: [Weekly/Biweekly]
```

---

## Course Completion Checklist

### Weekly Assignments

| Week | Topic | Lab Complete | Quiz Passed | Reflection |
|------|-------|--------------|-------------|------------|
| 1 | Foundations | ☐ | ☐ | ☐ |
| 2 | EDA | ☐ | ☐ | ☐ |
| 3 | Statistics | ☐ | ☐ | ☐ |
| 4 | Classification | ☐ | ☐ | ☐ |
| 5 | NLP | ☐ | ☐ | ☐ |
| 6 | Supervised Learning | ☐ | ☐ | ☐ |
| 7 | Clustering | ☐ | ☐ | ☐ |
| 8 | Evaluation | ☐ | ☐ | ☐ |
| 9 | MLOps | ☐ | ☐ | ☐ |
| 10 | Capstone | ☐ | N/A | ☐ |

### Capstone Requirements

| Component | Complete | Reviewed |
|-----------|----------|----------|
| Evaluation Report | ☐ | ☐ |
| Code Notebook | ☐ | ☐ |
| Executive Summary | ☐ | ☐ |
| Presentation | ☐ | ☐ |

---

## Final Reflection

As you complete this course, consider:

1. **What surprised you most** about data science and AI evaluation?

2. **What skill do you feel most confident** applying in your work?

3. **What area do you want to learn more** about?

4. **How will you apply** what you've learned in the next 30 days?

5. **Who will you share** your knowledge with?

---

## Congratulations! 🎉

You've completed **Data Science for Fusion Developers**!

You now have the skills to:

- ✅ Explore and visualize data effectively
- ✅ Apply statistical thinking to AI evaluation
- ✅ Calculate and interpret classification metrics
- ✅ Understand how NLP and ML systems work
- ✅ Detect bias and fairness issues
- ✅ Design monitoring and maintenance plans
- ✅ Communicate findings to stakeholders
- ✅ Make data-driven AI recommendations

### Your Certificate

Upon submitting your capstone project, you'll receive:

**Data Science for Fusion Developers Certificate**

Validating your competency in AI evaluation methodology for support systems.

---

## Stay Connected

- **Alumni Slack Channel**: Continue learning with your cohort
- **Monthly Office Hours**: Ask questions, share experiences
- **Quarterly Updates**: New content and techniques
- **Mentorship Program**: Help future cohorts as a mentor

---

*Thank you for your commitment to learning. The future of AI-powered support depends on professionals like you who understand both the technology and the human elements. Go forth and build better AI systems!*

---

[← Week 9: AI in Production](../Week09/README.md) | [Back to Course Overview](../DataScienceForFusionEngineers.md)
