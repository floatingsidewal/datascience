# Week 3: Statistical Foundations

## Making Decisions with Confidence

This week, we learn the statistical concepts that underpin data-driven decision making. You'll understand how to design experiments, interpret results, and avoid common statistical pitfalls.

---

## Learning Objectives

By the end of this week, you will be able to:

1. Apply basic probability concepts to support scenarios
2. Formulate and test hypotheses
3. Interpret p-values and understand their limitations
4. Distinguish between Type I and Type II errors
5. Calculate and interpret confidence intervals
6. Design a valid A/B test for AI improvements

---

## Core Concepts

### 1. Probability Basics

#### Key Concepts

| Term | Definition | Support Example |
|------|------------|-----------------|
| **Probability** | Likelihood of an event (0 to 1) | P(escalation) = 0.15 |
| **Conditional Probability** | Probability given another event | P(escalation \| Critical priority) = 0.45 |
| **Independent Events** | One doesn't affect the other | Tickets from different customers |
| **Dependent Events** | One affects the other | Multiple tickets from same customer |

#### Conditional Probability Example

"What's the probability a ticket escalates, given it's marked Critical?"

```
P(Escalation | Critical) = P(Escalation AND Critical) / P(Critical)

If:
- 5% of all tickets are Critical
- 3% of all tickets are both Critical AND Escalated

Then:
P(Escalation | Critical) = 0.03 / 0.05 = 0.60 (60%)
```

### 2. Hypothesis Testing

#### The Framework

1. **Null Hypothesis (H₀)**: The "no effect" assumption
2. **Alternative Hypothesis (H₁)**: What we're trying to prove
3. **Test**: Collect data and calculate probability
4. **Decision**: Reject or fail to reject H₀

#### Support Example

**Scenario**: Testing if a new AI response template improves CSAT

- **H₀**: New template has no effect on CSAT (μ_new = μ_old)
- **H₁**: New template improves CSAT (μ_new > μ_old)
- **Test**: Compare CSAT scores from both groups
- **Decision**: If p-value < 0.05, reject H₀

### 3. Type I and Type II Errors

|  | H₀ is Actually True | H₀ is Actually False |
|--|---------------------|---------------------|
| **Reject H₀** | Type I Error (False Positive) | Correct! ✓ |
| **Fail to Reject H₀** | Correct! ✓ | Type II Error (False Negative) |

#### In Support Context

**Type I Error (False Positive)**:
- Conclude the new AI is better when it's actually not
- **Risk**: Roll out an ineffective (or worse) system

**Type II Error (False Negative)**:
- Conclude the new AI is not better when it actually is
- **Risk**: Miss an opportunity to improve

### 4. P-Values

The p-value is the probability of seeing results this extreme (or more extreme) if H₀ were true.

**Common Misconception**: "p = 0.03 means there's a 3% chance the null hypothesis is true"
**Reality**: "If H₀ were true, there's a 3% chance of seeing data this extreme"

#### Decision Rule
- p < 0.05 → Reject H₀ (statistically significant)
- p ≥ 0.05 → Fail to reject H₀ (not statistically significant)

**Warning**: Statistical significance ≠ Practical significance!

### 5. Confidence Intervals

A range of values that likely contains the true population parameter.

**Example**: "We're 95% confident that the true average resolution time is between 3.8 and 4.4 hours."

```
95% CI = Sample Mean ± (1.96 × Standard Error)
```

**Interpretation**: If we repeated this sampling 100 times, about 95 of those intervals would contain the true value.

---

## A/B Testing for AI

### Designing a Valid A/B Test

#### Step 1: Define Your Hypothesis

```
H₀: AI Response Version A and Version B have equal CSAT
H₁: AI Response Version B has higher CSAT than Version A
```

#### Step 2: Determine Sample Size

Factors that affect required sample size:
- **Effect size**: Smaller effects need more data
- **Variance**: More variable data needs more samples
- **Significance level (α)**: Lower α needs more samples
- **Power (1-β)**: Higher power needs more samples

```python
# Sample size calculation (simplified)
from scipy import stats
import numpy as np

def sample_size_ab_test(baseline_rate, min_detectable_effect, alpha=0.05, power=0.80):
    """Calculate required sample size per group for an A/B test"""
    effect = baseline_rate * min_detectable_effect
    pooled_prob = (baseline_rate + baseline_rate + effect) / 2
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)

    n = (2 * pooled_prob * (1 - pooled_prob) * (z_alpha + z_beta)**2) / (effect**2)
    return int(np.ceil(n))

# Example: Detecting 5% improvement in CSAT (baseline 75%)
n = sample_size_ab_test(0.75, 0.05)
print(f"Need {n} tickets per group")
```

#### Step 3: Randomization

Ensure tickets are randomly assigned to A or B:
- No bias by time of day
- No bias by customer segment
- No bias by agent

#### Step 4: Run the Test

Collect data until you reach your target sample size.

**Important**: Don't peek and stop early! This inflates false positives.

#### Step 5: Analyze Results

```python
from scipy import stats

# Example data
group_a_csat = [4, 5, 4, 3, 5, 4, 4, 5, 3, 4]  # Control
group_b_csat = [5, 5, 4, 5, 5, 4, 5, 5, 4, 5]  # Treatment

# Perform t-test
t_stat, p_value = stats.ttest_ind(group_a_csat, group_b_csat)

print(f"Group A mean: {np.mean(group_a_csat):.2f}")
print(f"Group B mean: {np.mean(group_b_csat):.2f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Result: Statistically significant difference!")
else:
    print("Result: No statistically significant difference")
```

---

## Why This Matters for AI

### Evaluating AI Vendor Claims

When a vendor says "Our AI improves CSAT by 15%":

**Questions to Ask**:
1. What was the sample size?
2. What was the baseline CSAT?
3. Was this a randomized controlled trial?
4. What was the p-value?
5. What's the confidence interval?

### Common Red Flags

| Claim | Concern |
|-------|---------|
| "100% accuracy improvement" | No confidence interval or sample size |
| "Tested with 50 tickets" | Sample size too small for reliable conclusions |
| "Works for all industries" | May not generalize to your specific context |
| "Results in just 2 days" | Likely didn't run long enough |

---

## Hands-On Lab: Designing an A/B Test

### Scenario

You want to test whether adding sentiment detection to your AI routing improves resolution time.

**Current State (Control)**: AI routes based on keywords
**New System (Treatment)**: AI routes based on keywords + detected sentiment

### Lab Exercise

```python
# Week 3 Lab: A/B Test Design and Analysis
import numpy as np
import pandas as pd
from scipy import stats

# Simulate experiment data
np.random.seed(42)

# Control group: Current routing (average 4.5 hours resolution)
control_resolution = np.random.normal(loc=4.5, scale=1.5, size=200)
control_resolution = np.clip(control_resolution, 0.5, 12)  # Realistic bounds

# Treatment group: New routing (average 4.1 hours - 9% improvement)
treatment_resolution = np.random.normal(loc=4.1, scale=1.4, size=200)
treatment_resolution = np.clip(treatment_resolution, 0.5, 12)

# Create dataframe
experiment_data = pd.DataFrame({
    'group': ['control'] * 200 + ['treatment'] * 200,
    'resolution_hours': np.concatenate([control_resolution, treatment_resolution])
})

# Task 1: Calculate descriptive statistics by group
print("=== Descriptive Statistics ===")
print(experiment_data.groupby('group')['resolution_hours'].agg(['mean', 'std', 'count']))

# Task 2: Visualize distributions
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(control_resolution, bins=20, alpha=0.7, label='Control', color='blue')
plt.hist(treatment_resolution, bins=20, alpha=0.7, label='Treatment', color='green')
plt.xlabel('Resolution Time (hours)')
plt.ylabel('Frequency')
plt.legend()
plt.title('Distribution by Group')

plt.subplot(1, 2, 2)
experiment_data.boxplot(column='resolution_hours', by='group')
plt.ylabel('Resolution Time (hours)')
plt.title('Resolution Time by Group')
plt.suptitle('')
plt.tight_layout()
plt.show()

# Task 3: Perform hypothesis test
t_stat, p_value = stats.ttest_ind(control_resolution, treatment_resolution)

print("\n=== Hypothesis Test Results ===")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Task 4: Calculate confidence interval for the difference
diff_mean = np.mean(control_resolution) - np.mean(treatment_resolution)
diff_se = np.sqrt(np.var(control_resolution)/200 + np.var(treatment_resolution)/200)
ci_lower = diff_mean - 1.96 * diff_se
ci_upper = diff_mean + 1.96 * diff_se

print(f"\n=== Effect Size ===")
print(f"Mean difference: {diff_mean:.2f} hours")
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}] hours")

# Task 5: Make a recommendation
print("\n=== Recommendation ===")
if p_value < 0.05 and diff_mean > 0:
    print(f"The treatment significantly reduces resolution time by ~{diff_mean:.1f} hours (p={p_value:.4f})")
    print("Recommendation: Roll out the new routing system")
else:
    print("No significant improvement detected. Gather more data or investigate further.")
```

### Lab Questions

1. What is the null hypothesis for this experiment?
2. Based on the p-value, should we reject the null hypothesis?
3. What does the confidence interval tell us about the effect size?
4. What factors might confound these results?

---

## Knowledge Check

1. If p = 0.03, this means:
   - a) There's a 3% chance the null hypothesis is true
   - b) There's a 3% chance of seeing these results if the null hypothesis were true
   - c) The effect is 3% larger than baseline
   - d) We need 3% more data

2. A Type II error in AI testing would mean:
   - a) Deploying an AI that's actually worse
   - b) Not deploying an AI that's actually better
   - c) Running too small a sample
   - d) Using the wrong metrics

3. "The 95% confidence interval for resolution time improvement is [0.2, 0.8] hours" means:
   - a) Improvement is definitely between 0.2 and 0.8 hours
   - b) 95% of tickets improve by 0.2-0.8 hours
   - c) We're 95% confident the true improvement is in this range
   - d) The improvement is statistically significant 95% of the time

4. To detect smaller effect sizes in an A/B test, you need:
   - a) Fewer samples
   - b) More samples
   - c) Lower confidence
   - d) Higher p-values

*(Answers: 1-b, 2-b, 3-c, 4-b)*

---

## Reflection Journal

1. Think of an AI improvement you'd like to test. What would be your null and alternative hypotheses?

2. What's worse in your context: a Type I or Type II error? Why?

3. Have you seen claims about AI performance that didn't include statistical evidence? What questions would you now ask?

---

## Bridge to Week 4

Next week, we focus on **classification metrics**—the specific ways we measure how well AI categorizes things. You'll learn:
- How to read and interpret confusion matrices
- When to prioritize precision vs. recall
- How to set the right thresholds for AI decisions

**Preparation**: Think about a classification task in your work (e.g., ticket routing, intent detection). What matters more: catching all relevant cases or being right when you make a prediction?

---

## Additional Resources

### From the Data Science Survey
- [Statistics Q3-Q6](../../DataScienceSurvey.md#statistics) — Hypothesis testing, p-values
- [Data Science Q9-Q16](../../DataScienceSurvey.md#data-science) — A/B testing, statistical power

### External Resources
- [Seeing Theory - Probability](https://seeing-theory.brown.edu/)
- [Khan Academy - Significance Tests](https://www.khanacademy.org/math/statistics-probability/significance-tests-one-sample)
- [A/B Testing Calculator](https://www.evanmiller.org/ab-testing/)

---

[← Week 2: EDA](../Week02/README.md) | [Back to Course Overview](../DataScienceForFusionEngineers.md) | [Week 4: Classification & Confusion →](../Week04/README.md)
