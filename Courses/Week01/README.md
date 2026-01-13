# Week 1: Foundations & Data Thinking

## Building the Mental Models for Thinking About Data

Welcome to Week 1! This week establishes the foundation for everything that follows. We'll learn how to think about data the way data scientists do, while keeping our focus on support-relevant applications.

---

## Learning Objectives

By the end of this week, you will be able to:

1. Identify and classify different data types in support contexts
2. Distinguish between independent and dependent variables
3. Calculate and interpret basic descriptive statistics
4. Set up and use Python with Jupyter notebooks
5. Load and inspect data using Pandas
6. Explain why data quality matters for AI systems

---

## Core Concepts

### 1. Data Types

Understanding data types is fundamental to working with any dataset.

#### Categorical Data
Data that represents categories or groups with no inherent order.

**Support Examples**:
- Ticket category (Billing, Technical, Account)
- Customer segment (Enterprise, SMB, Consumer)
- Channel (Email, Chat, Phone)

#### Numerical Data
Quantitative values that can be measured.

**Support Examples**:
- Resolution time (minutes)
- Customer satisfaction score (1-5)
- Number of replies in a ticket

#### Ordinal Data
Categorical data with a meaningful order.

**Support Examples**:
- Priority level (Low, Medium, High, Critical)
- Severity (Minor, Major, Blocker)
- Experience level (Beginner, Intermediate, Expert)

### 2. Variables

#### Independent Variables (Features)
Factors we use to make predictions or understand outcomes.

**Support Examples**:
- Time of day ticket submitted
- Customer's subscription tier
- Product version

#### Dependent Variables (Targets)
Outcomes we're trying to predict or understand.

**Support Examples**:
- Resolution time
- Customer satisfaction score
- Escalation (yes/no)

### 3. Descriptive Statistics

| Statistic | Description | Support Example |
|-----------|-------------|-----------------|
| **Mean** | Average value | Average resolution time: 4.2 hours |
| **Median** | Middle value when sorted | Median CSAT: 4 out of 5 |
| **Mode** | Most frequent value | Most common category: Technical |
| **Standard Deviation** | Spread of values | Resolution time varies by ±2.1 hours |
| **Range** | Difference between max and min | Tickets range from 1 to 47 replies |

---

## Why This Matters for AI

### Training Data Quality

AI models learn from data. If the training data is:
- **Biased** → The AI will be biased
- **Incomplete** → The AI will miss patterns
- **Mislabeled** → The AI will learn wrong patterns

### Real Support Example

Imagine training a chatbot to classify tickets:

**Good Training Data**:
```
Ticket: "I can't log into my account"
Label: Account Access

Ticket: "My invoice shows the wrong amount"
Label: Billing
```

**Problematic Training Data**:
```
Ticket: "I can't log into my account"
Label: Technical  ← Wrong category!

Ticket: "Help"
Label: ???  ← Too vague to label
```

---

## Hands-On Lab: Your First Data Exploration

### Setup Instructions

**Using Visual Studio Code (Recommended)**

1. **Install VS Code**
   - Download from [code.visualstudio.com](https://code.visualstudio.com)
   - Or install via Microsoft Store (Windows)

2. **Install Python**
   - Download Python 3.x from [python.org](https://python.org) or Microsoft Store
   - During installation, check "Add Python to PATH"

3. **Configure VS Code for Python**
   - Open VS Code
   - Install the "Python" extension (by Microsoft)
   - Install the "Jupyter" extension (by Microsoft)
   - Press `Ctrl+Shift+P`, type "Python: Select Interpreter", choose your Python installation

4. **Install Required Libraries**
   - Open VS Code terminal (`Ctrl+``)
   - Run: `pip install pandas matplotlib jupyter`

5. **Create Your First Notebook**
   - Press `Ctrl+Shift+P`, type "Create: New Jupyter Notebook"
   - Save as `Week1_Lab.ipynb`

### Lab Exercise

```python
# Week 1 Lab: Support Ticket Data Exploration
import pandas as pd

# Load sample support ticket data
# (In practice, this would be your actual ticket export)
data = {
    'ticket_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
    'category': ['Technical', 'Billing', 'Technical', 'Account', 'Technical',
                 'Billing', 'Technical', 'Account', 'Billing', 'Technical'],
    'priority': ['High', 'Low', 'Medium', 'High', 'Low',
                 'Medium', 'High', 'Low', 'Medium', 'High'],
    'resolution_time_hours': [2.5, 1.0, 4.2, 6.0, 0.5,
                              1.5, 8.0, 2.0, 1.2, 5.5],
    'csat_score': [4, 5, 3, 2, 5, 4, 2, 4, 5, 3],
    'num_replies': [3, 2, 5, 8, 1, 2, 10, 3, 2, 6]
}

tickets = pd.DataFrame(data)

# Task 1: View the first few rows
print("=== First 5 Tickets ===")
print(tickets.head())

# Task 2: Check data types
print("\n=== Data Types ===")
print(tickets.dtypes)

# Task 3: Calculate descriptive statistics
print("\n=== Descriptive Statistics ===")
print(tickets.describe())

# Task 4: Count categories
print("\n=== Tickets by Category ===")
print(tickets['category'].value_counts())

# Task 5: Calculate mean resolution time by priority
print("\n=== Mean Resolution Time by Priority ===")
print(tickets.groupby('priority')['resolution_time_hours'].mean())
```

### Lab Questions

1. What is the most common ticket category in this dataset?
2. What is the average CSAT score?
3. Which priority level has the longest average resolution time?
4. Identify one potential data quality issue you notice.

---

## AI Application: Data Labeling Fundamentals

### The Labeling Challenge

When building AI for support, human labeling is critical. Consider these challenges:

| Challenge | Example | Impact on AI |
|-----------|---------|--------------|
| **Ambiguity** | "I need help" - which category? | Inconsistent training signals |
| **Subjectivity** | Is this "High" or "Medium" priority? | Model learns arbitrary patterns |
| **Label Drift** | Category meanings change over time | Model becomes outdated |
| **Missing Labels** | 30% of tickets unclassified | Incomplete learning |

### Best Practices for Labeling

1. **Create clear labeling guidelines** with examples
2. **Use multiple labelers** and check agreement
3. **Handle edge cases explicitly** (e.g., "Other" category)
4. **Review labels periodically** for consistency

---

## Knowledge Check

1. A customer's account type (Free, Pro, Enterprise) is what kind of data?
   - a) Numerical
   - b) Categorical
   - c) Ordinal
   - d) Continuous

2. In predicting whether a ticket will escalate, "ticket escalated (yes/no)" is the:
   - a) Independent variable
   - b) Dependent variable
   - c) Confounding variable
   - d) Categorical variable

3. Which statistic is LEAST affected by extreme values (outliers)?
   - a) Mean
   - b) Standard deviation
   - c) Median
   - d) Range

4. Why does mislabeled training data hurt AI performance?
   - a) It makes the model run slower
   - b) It teaches the model incorrect patterns
   - c) It uses more storage space
   - d) It requires more computing power

*(Answers: 1-c, 2-b, 3-c, 4-b)*

---

## Reflection Journal

Take 10 minutes to answer these questions:

1. Think about the data you work with daily. What are three examples of categorical data? Numerical data?

2. What data quality issues have you noticed in your support tickets? How might these affect an AI trained on that data?

3. If you were building a labeling guide for ticket categories, what would be the hardest categories to define clearly?

---

## Bridge to Week 2

Next week, we'll build on these foundations by learning to **visualize** data. You'll discover how to:
- Create charts that reveal patterns in ticket data
- Identify distributions and outliers visually
- Tell stories with data visualization

**Preparation**: Think about a question you'd like to answer about your support data that a chart might help with.

---

## Additional Resources

### From the Data Science Survey
- [Statistics Q1-Q8](../../DataScienceSurvey.md#statistics) — Sampling, distributions, bias
- [Data Science Q1-Q6](../../DataScienceSurvey.md#data-science) — Core data science concepts

### External Resources
- [Python for Everybody - Chapter 1](https://www.py4e.com/lessons/intro)
- [Pandas 10-minute tutorial](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Khan Academy - Descriptive Statistics](https://www.khanacademy.org/math/statistics-probability)

---

[← Back to Course Overview](../DataScienceForFusionEngineers.md) | [Week 2: Exploratory Data Analysis →](../Week02/README.md)
