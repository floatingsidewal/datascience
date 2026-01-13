# Week 6: Supervised Learning Concepts

## How AI Learns from Labeled Examples

This week we explore how supervised learning works—the foundation of most AI systems you'll evaluate and deploy. Understanding these concepts helps you ensure AI training data quality and recognize common failure modes.

---

## Learning Objectives

By the end of this week, you will be able to:

1. Explain the supervised learning workflow
2. Properly split data into training, validation, and test sets
3. Identify overfitting and underfitting from performance metrics
4. Understand cross-validation and why it matters
5. Apply basic feature engineering concepts
6. Evaluate the role of human labeling in AI quality

---

## Core Concepts

### 1. The Supervised Learning Loop

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   Training Data          Model Training        Prediction    │
│   ┌─────────────┐       ┌────────────┐       ┌──────────┐   │
│   │ Features (X)│ ───>  │            │ ───>  │ Predicted│   │
│   │ Labels (y)  │       │   Model    │       │  Labels  │   │
│   └─────────────┘       └────────────┘       └──────────┘   │
│                              │                     │         │
│                              │   Compare & Learn   │         │
│                              └─────────────────────┘         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Key Components**:
- **Features (X)**: Input variables the model uses to make predictions
- **Labels (y)**: The correct answers (what we want to predict)
- **Model**: The algorithm that learns patterns from X to predict y
- **Training**: Adjusting model parameters to minimize prediction errors

### 2. Data Splitting: Why and How

#### The Three Sets

| Set | Purpose | Typical Size |
|-----|---------|--------------|
| **Training** | Model learns from this data | 60-80% |
| **Validation** | Tune hyperparameters, avoid overfitting | 10-20% |
| **Test** | Final evaluation (never seen during training) | 10-20% |

#### Why Split?

**Problem**: If we test on data the model trained on, we don't know if it truly learned patterns or just memorized answers.

**Analogy**: It's like giving students the exact exam questions to study. High scores don't prove understanding.

```python
from sklearn.model_selection import train_test_split

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: separate validation from training
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 of 0.8 = 0.2
)

print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
```

#### Critical Rule: Test Data is Sacred

**Never** use test data during model development. It should only be used once, at the very end, to get an unbiased estimate of performance.

### 3. Overfitting vs. Underfitting

| Condition | Training Performance | Test Performance | Cause |
|-----------|---------------------|------------------|-------|
| **Underfitting** | Poor | Poor | Model too simple |
| **Good Fit** | Good | Good | Just right |
| **Overfitting** | Excellent | Poor | Model memorized training data |

#### Visual Intuition

```
Performance
    │
    │    Underfitting          Overfitting
    │         ↓                    ↓
    │    ┌────────────────────────────┐
    │    │  Training _______________/
    │    │          /
    │    │         /        \
    │    │        /          \ Test
    │    │       /            \___
    │    │______/
    │    └────────────────────────────┘
    └─────────────────────────────────────> Model Complexity
              ↑
          Sweet Spot
```

#### Detecting Overfitting

```python
# Train model
model.fit(X_train, y_train)

# Evaluate on both sets
train_accuracy = model.score(X_train, y_train)
val_accuracy = model.score(X_val, y_val)

print(f"Training Accuracy: {train_accuracy:.2%}")
print(f"Validation Accuracy: {val_accuracy:.2%}")

# Red flag: Big gap suggests overfitting
gap = train_accuracy - val_accuracy
if gap > 0.10:
    print("⚠️ Warning: Possible overfitting!")
```

### 4. Cross-Validation

Instead of a single train/validation split, use multiple splits and average results.

#### K-Fold Cross-Validation

```
Data: [1][2][3][4][5]

Fold 1: Train on [2,3,4,5], Test on [1]
Fold 2: Train on [1,3,4,5], Test on [2]
Fold 3: Train on [1,2,4,5], Test on [3]
Fold 4: Train on [1,2,3,5], Test on [4]
Fold 5: Train on [1,2,3,4], Test on [5]

Final Score = Average of all 5 test scores
```

**Benefits**:
- More robust performance estimate
- Uses all data for both training and testing
- Reduces variance in evaluation

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)

print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.2%} (+/- {scores.std()*2:.2%})")
```

### 5. Feature Engineering

Transforming raw data into features that better represent the underlying problem.

#### Support Ticket Features

| Raw Data | Engineered Feature |
|----------|-------------------|
| Ticket text | TF-IDF vector, word count, sentiment score |
| Created timestamp | Hour of day, day of week, is_weekend |
| Customer ID | Customer tier, tickets in past 30 days |
| Ticket history | Resolution time trend, escalation history |

```python
def engineer_ticket_features(ticket):
    """Transform raw ticket into model features"""
    features = {}

    # Text features
    features['word_count'] = len(ticket['text'].split())
    features['has_question_mark'] = '?' in ticket['text']
    features['contains_urgent'] = 'urgent' in ticket['text'].lower()

    # Time features
    created = pd.to_datetime(ticket['created_at'])
    features['hour'] = created.hour
    features['is_weekend'] = created.dayofweek >= 5
    features['is_business_hours'] = 9 <= created.hour <= 17

    # Customer features
    features['customer_tier'] = ticket['customer_tier']
    features['past_escalations'] = ticket['customer_escalation_count']

    return features
```

---

## Why This Matters for AI

### The Human Role in Supervised Learning

As a Fusion Developer, you play a critical role in AI quality through:

1. **Data Curation**: Selecting representative training examples
2. **Labeling Quality**: Ensuring consistent, accurate labels
3. **Feature Definition**: Identifying what information matters
4. **Validation**: Reviewing model predictions and errors

### Common AI Failures from Training Issues

| Issue | Symptom | Root Cause |
|-------|---------|------------|
| **Biased Training Data** | AI performs poorly for certain customer segments | Underrepresentation in training |
| **Label Inconsistency** | Unpredictable AI behavior | Different people labeled same cases differently |
| **Data Leakage** | Great training results, poor production results | Information from the future leaked into training |
| **Concept Drift** | AI performance degrades over time | Training data no longer represents current reality |

### Data Leakage Example

**Bad**: Training a "will this ticket escalate?" model that includes the actual escalation outcome in features.

```python
# WRONG - This leaks the answer!
features = ['ticket_text', 'customer_tier', 'escalation_notes']  # ← Leakage!

# RIGHT - Only use information available at prediction time
features = ['ticket_text', 'customer_tier', 'time_since_last_update']
```

---

## Hands-On Lab: Building a Ticket Classifier

### Lab Exercise

```python
# Week 6 Lab: Supervised Learning for Ticket Classification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Generate sample ticket data
np.random.seed(42)

tickets_data = {
    'text': [
        # Technical issues
        "App crashes when I try to upload", "Software not responding to clicks",
        "Error message appears on startup", "Application freezes randomly",
        "Can't install the latest update", "Program closes unexpectedly",
        "Bug in the export feature", "Technical glitch in reporting",
        # Billing issues
        "Incorrect charge on my statement", "Need refund for duplicate payment",
        "Invoice shows wrong amount", "Subscription billing question",
        "Cancel my subscription please", "Payment didn't go through",
        "Charged twice this month", "Update my payment method",
        # Account issues
        "Can't log into my account", "Password reset not working",
        "Change my email address", "Account locked out",
        "Two-factor authentication issue", "Update my profile information",
        "Delete my account please", "Merge duplicate accounts"
    ],
    'category': ['Technical'] * 8 + ['Billing'] * 8 + ['Account'] * 8
}

# Duplicate and add noise for more data
df = pd.DataFrame(tickets_data)
df = pd.concat([df] * 10, ignore_index=True)  # 240 tickets

# Add some noise to make it realistic
noise_indices = np.random.choice(len(df), size=20, replace=False)
noise_categories = np.random.choice(['Technical', 'Billing', 'Account'], size=20)
df.loc[noise_indices, 'category'] = noise_categories

print(f"Dataset size: {len(df)} tickets")
print(f"Category distribution:\n{df['category'].value_counts()}")

# Task 1: Proper train/test split
X = df['text']
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)}, Test set: {len(X_test)}")

# Task 2: Feature extraction with TF-IDF
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)  # Note: transform, not fit_transform!

print(f"Feature matrix shape: {X_train_tfidf.shape}")

# Task 3: Train a simple model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Task 4: Evaluate - Check for overfitting
train_accuracy = model.score(X_train_tfidf, y_train)
test_accuracy = model.score(X_test_tfidf, y_test)

print(f"\n=== Model Performance ===")
print(f"Training Accuracy: {train_accuracy:.2%}")
print(f"Test Accuracy: {test_accuracy:.2%}")
print(f"Gap: {train_accuracy - test_accuracy:.2%}")

if train_accuracy - test_accuracy > 0.10:
    print("⚠️ Warning: Possible overfitting!")
else:
    print("✓ Model generalizes well")

# Task 5: Cross-validation for robust estimate
cv_scores = cross_val_score(model,
                            vectorizer.transform(X),  # All data
                            y, cv=5)

print(f"\n=== Cross-Validation Results ===")
print(f"Fold scores: {cv_scores.round(3)}")
print(f"Mean CV Score: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")

# Task 6: Detailed classification report
y_pred = model.predict(X_test_tfidf)
print(f"\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Task 7: Analyze feature importance
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_

plt.figure(figsize=(12, 4))
for idx, category in enumerate(model.classes_):
    top_indices = np.argsort(coefficients[idx])[-5:]
    top_features = [(feature_names[i], coefficients[idx][i]) for i in top_indices]

    print(f"\nTop features for '{category}':")
    for feat, coef in reversed(top_features):
        print(f"  {feat}: {coef:.3f}")

# Task 8: Test with new examples
new_tickets = [
    "My payment was declined",
    "The app keeps crashing on my phone",
    "I forgot my password"
]

new_tfidf = vectorizer.transform(new_tickets)
predictions = model.predict(new_tfidf)
probabilities = model.predict_proba(new_tfidf)

print(f"\n=== Predictions for New Tickets ===")
for ticket, pred, probs in zip(new_tickets, predictions, probabilities):
    confidence = max(probs)
    print(f"'{ticket}'")
    print(f"  → {pred} (confidence: {confidence:.1%})")
```

### Lab Questions

1. What is the gap between training and test accuracy? Does this suggest overfitting?
2. Which category has the best precision? Which has the best recall?
3. What are the most predictive features for each category?
4. For the new ticket predictions, are the confidence levels appropriate?

---

## Strategies to Combat Overfitting

| Strategy | How It Works | When to Use |
|----------|--------------|-------------|
| **More training data** | More examples help model generalize | When data is limited |
| **Simpler model** | Fewer parameters, less memorization | When model is too complex |
| **Regularization** | Penalize large weights (L1/L2) | Standard practice |
| **Cross-validation** | Evaluate on multiple splits | Always for model selection |
| **Early stopping** | Stop training when validation worsens | Deep learning |
| **Dropout** | Randomly disable neurons | Neural networks |

---

## Knowledge Check

1. The purpose of a validation set is to:
   - a) Train the model
   - b) Tune hyperparameters without biasing test evaluation
   - c) Final model evaluation
   - d) Store production data

2. If training accuracy is 98% but test accuracy is 65%, this indicates:
   - a) Underfitting
   - b) Good generalization
   - c) Overfitting
   - d) Data leakage

3. K-fold cross-validation helps by:
   - a) Making training faster
   - b) Providing more robust performance estimates
   - c) Requiring less data
   - d) Eliminating the need for a test set

4. Feature engineering is:
   - a) Fixing bugs in code
   - b) Transforming raw data into useful model inputs
   - c) Selecting which model to use
   - d) Cleaning data

*(Answers: 1-b, 2-c, 3-b, 4-b)*

---

## Reflection Journal

1. Think about your organization's AI training data. Is it properly split? How do you know the AI isn't just memorizing?

2. What features would you engineer from support ticket data to predict escalation likelihood?

3. How might concept drift affect AI in your support context? What would change over time?

---

## Bridge to Week 7

Next week, we explore **unsupervised learning**—how AI finds patterns without labels. You'll learn:
- How clustering groups similar tickets automatically
- When unsupervised learning is more appropriate than supervised
- How to use clustering to discover ticket taxonomies

**Preparation**: Think about how your ticket categories were originally defined. Could AI discover better categories from the data itself?

---

## Additional Resources

### From the Data Science Survey
- [Machine Learning Q1-Q6](../../DataScienceSurvey.md#machine-learning) — ML fundamentals
- [Data Science Q17-Q19](../../DataScienceSurvey.md#data-science) — Overfitting/underfitting

### External Resources
- [Scikit-learn Model Selection](https://scikit-learn.org/stable/model_selection.html)
- [Azure ML - Train Models](https://learn.microsoft.com/en-us/azure/machine-learning/concept-train-machine-learning-model)
- [Microsoft Learn - ML Fundamentals](https://learn.microsoft.com/en-us/training/paths/create-machine-learn-models/)

---

[← Week 5: Text & Language](../Week05/README.md) | [Back to Course Overview](../DataScienceForFusionEngineers.md) | [Week 7: Clustering →](../Week07/README.md)
