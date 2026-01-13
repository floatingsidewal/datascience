# Week 2: Exploratory Data Analysis

## Discovering Patterns and Telling Stories with Data

This week, we transform numbers into insights through visualization. You'll learn to see patterns that raw data hides and communicate findings effectively to stakeholders.

---

## Learning Objectives

By the end of this week, you will be able to:

1. Create effective visualizations using Matplotlib and Seaborn
2. Recognize common distribution shapes and their implications
3. Distinguish between correlation and causation
4. Identify outliers and understand their impact
5. Clean and transform data for analysis
6. Apply EDA techniques to support metrics

---

## Core Concepts

### 1. Distribution Shapes

Understanding how your data is distributed reveals important patterns.

#### Normal Distribution (Bell Curve)
- Most values cluster around the mean
- Symmetric on both sides
- **Support Example**: CSAT scores often follow this pattern

#### Skewed Distributions
- **Right-skewed (positive)**: Long tail to the right
  - *Example*: Resolution times (most quick, some very long)
- **Left-skewed (negative)**: Long tail to the left
  - *Example*: Customer tenure before churn (most leave early)

#### Bimodal Distribution
- Two distinct peaks
- **Support Example**: Response times (quick auto-responses vs. human responses)

### 2. Visualization Types

| Chart Type | Best For | Support Use Case |
|------------|----------|------------------|
| **Histogram** | Distribution of one variable | Resolution time distribution |
| **Bar Chart** | Comparing categories | Tickets by category |
| **Line Chart** | Trends over time | Daily ticket volume |
| **Scatter Plot** | Relationship between two variables | Resolution time vs. CSAT |
| **Box Plot** | Distribution + outliers | Resolution time by priority |
| **Heatmap** | Patterns in matrices | Hourly ticket volume by day |

### 3. Correlation vs. Causation

**Correlation**: Two variables move together
**Causation**: One variable causes the other to change

**Support Example**:
- **Observed**: Tickets with more replies have lower CSAT scores
- **Correlation**: Yes, these are related
- **Causation?**: Not necessarily! Complex issues require more replies AND cause frustration—the replies don't cause the low CSAT.

### 4. Outliers

Extreme values that differ significantly from other observations.

**Questions to Ask**:
1. Is this a data entry error? (typo, wrong units)
2. Is this a genuine extreme case? (VIP escalation, system outage)
3. Should it be included in analysis? (depends on the question)

---

## Why This Matters for AI

### Visualizing Model Performance

EDA skills help you:
- **Spot data drift**: When incoming data patterns change
- **Identify bias**: When AI performs differently for different groups
- **Detect anomalies**: When AI accuracy suddenly drops

### Real Support Example

A chatbot's accuracy over time might reveal:

```
Week 1-4:  85% accuracy (stable)
Week 5:    72% accuracy (sudden drop!)
Week 6-8:  70% accuracy (new normal)
```

**Investigation reveals**: A new product launched in Week 5, introducing questions the AI wasn't trained on.

---

## Hands-On Lab: Visual Analysis of Support Data

### Lab Exercise

```python
# Week 2 Lab: Exploratory Data Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Expanded sample data
import numpy as np
np.random.seed(42)

n_tickets = 200
data = {
    'ticket_id': range(1000, 1000 + n_tickets),
    'category': np.random.choice(['Technical', 'Billing', 'Account', 'Feature Request'],
                                  n_tickets, p=[0.4, 0.25, 0.2, 0.15]),
    'priority': np.random.choice(['Low', 'Medium', 'High', 'Critical'],
                                  n_tickets, p=[0.3, 0.4, 0.2, 0.1]),
    'resolution_time_hours': np.random.exponential(scale=4, size=n_tickets),
    'csat_score': np.random.choice([1, 2, 3, 4, 5], n_tickets, p=[0.05, 0.1, 0.2, 0.35, 0.3]),
    'num_replies': np.random.poisson(lam=4, size=n_tickets),
    'day_of_week': np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], n_tickets)
}

tickets = pd.DataFrame(data)

# Add some realistic patterns
tickets.loc[tickets['priority'] == 'Critical', 'resolution_time_hours'] *= 2
tickets.loc[tickets['category'] == 'Feature Request', 'csat_score'] = \
    np.clip(tickets.loc[tickets['category'] == 'Feature Request', 'csat_score'] - 1, 1, 5)

# Task 1: Distribution of Resolution Times
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(tickets['resolution_time_hours'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Resolution Time (hours)')
plt.ylabel('Frequency')
plt.title('Distribution of Resolution Times')

plt.subplot(1, 2, 2)
plt.hist(tickets['resolution_time_hours'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Resolution Time (hours) - Log Scale')
plt.ylabel('Frequency')
plt.title('Resolution Times (Note the right skew)')
plt.xscale('log')
plt.tight_layout()
plt.show()

# Task 2: Tickets by Category (Bar Chart)
plt.figure(figsize=(8, 5))
category_counts = tickets['category'].value_counts()
sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
plt.xlabel('Category')
plt.ylabel('Number of Tickets')
plt.title('Ticket Distribution by Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Task 3: Resolution Time by Priority (Box Plot)
plt.figure(figsize=(8, 5))
priority_order = ['Low', 'Medium', 'High', 'Critical']
sns.boxplot(x='priority', y='resolution_time_hours', data=tickets, order=priority_order)
plt.xlabel('Priority')
plt.ylabel('Resolution Time (hours)')
plt.title('Resolution Time by Priority Level')
plt.show()

# Task 4: Correlation Heatmap
plt.figure(figsize=(6, 5))
numeric_cols = ['resolution_time_hours', 'csat_score', 'num_replies']
correlation_matrix = tickets[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Between Numeric Variables')
plt.tight_layout()
plt.show()

# Task 5: CSAT by Category
plt.figure(figsize=(8, 5))
sns.boxplot(x='category', y='csat_score', data=tickets)
plt.xlabel('Category')
plt.ylabel('CSAT Score')
plt.title('Customer Satisfaction by Ticket Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n=== Key Insights ===")
print(f"Median resolution time: {tickets['resolution_time_hours'].median():.1f} hours")
print(f"Most common category: {tickets['category'].mode()[0]}")
print(f"Correlation (replies vs CSAT): {tickets['num_replies'].corr(tickets['csat_score']):.2f}")
```

### Lab Questions

1. What shape is the resolution time distribution? Why might this be?
2. Which category has the lowest median CSAT score?
3. Is there a correlation between number of replies and CSAT? What might explain this?
4. Look at the box plot for Critical priority—what do the dots above the box represent?

---

## Data Cleaning Essentials

### Common Data Issues in Support Data

| Issue | Detection | Treatment |
|-------|-----------|-----------|
| **Missing values** | `df.isnull().sum()` | Impute, drop, or investigate |
| **Duplicate records** | `df.duplicated().sum()` | Remove or merge |
| **Inconsistent categories** | `df['col'].unique()` | Standardize (e.g., "Tech" → "Technical") |
| **Outliers** | Box plots, Z-scores | Investigate, cap, or separate |
| **Wrong data types** | `df.dtypes` | Convert (e.g., dates stored as strings) |

### Example: Cleaning Category Names

```python
# Before: Inconsistent category names
# ['Technical', 'Tech', 'TECHNICAL', 'technical ']

# Standardization function
def clean_category(cat):
    cat = str(cat).strip().lower()
    mapping = {
        'tech': 'Technical',
        'technical': 'Technical',
        'bill': 'Billing',
        'billing': 'Billing'
    }
    return mapping.get(cat, cat.title())

tickets['category_clean'] = tickets['category'].apply(clean_category)
```

---

## AI Application: Detecting Data Drift

### What is Data Drift?

When the statistical properties of incoming data change compared to training data.

### Visual Detection

```python
# Compare distributions: Training data vs. Recent data
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Training period (hypothetical)
axes[0].hist(training_data['resolution_time'], bins=20, alpha=0.7)
axes[0].set_title('Training Data Distribution')

# Recent period
axes[1].hist(recent_data['resolution_time'], bins=20, alpha=0.7, color='orange')
axes[1].set_title('Recent Data Distribution')

# If these look very different, you may have data drift!
```

### Why It Matters for AI

- Model was trained on winter data → performs poorly in summer
- New product launch → new question types AI hasn't seen
- Process change → different resolution patterns

---

## Knowledge Check

1. A distribution with a long tail to the right is called:
   - a) Left-skewed
   - b) Right-skewed
   - c) Normal
   - d) Bimodal

2. You observe that tickets with longer resolution times have lower CSAT scores. This is an example of:
   - a) Positive correlation
   - b) Negative correlation
   - c) No correlation
   - d) Causation

3. Which visualization is best for comparing the distribution of resolution times across priority levels?
   - a) Line chart
   - b) Pie chart
   - c) Box plot
   - d) Scatter plot

4. An outlier should always be removed from analysis. True or False?
   - a) True
   - b) False

*(Answers: 1-b, 2-b, 3-c, 4-b)*

---

## Reflection Journal

1. Think about a recent support metric you reviewed. What visualization would best represent it? Why?

2. Have you ever seen data that seemed "off"? How did you investigate it?

3. What correlations would you hypothesize exist in your support data? How would you test them?

---

## Bridge to Week 3

Next week, we dive into **statistical foundations**. You'll learn to:
- Quantify uncertainty with probability
- Design valid A/B tests for AI improvements
- Understand when results are statistically significant

**Preparation**: Think of a change you'd like to test in your support workflow. How would you measure success?

---

## Additional Resources

### From the Data Science Survey
- [Data Analysis Q1-Q12](../../DataScienceSurvey.md#data-analysis) — Analysis techniques
- [Data Science Q7](../../DataScienceSurvey.md#data-science) — Correlation and covariance

### External Resources
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Storytelling with Data](https://www.storytellingwithdata.com/) (blog)
- [Python Graph Gallery](https://www.python-graph-gallery.com/)

---

[← Week 1: Foundations](../Week01/README.md) | [Back to Course Overview](../DataScienceForFusionEngineers.md) | [Week 3: Statistical Foundations →](../Week03/README.md)
