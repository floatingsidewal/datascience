# Data Science for Fusion Engineers

## A 10-Week Practical Course for Customer Support Engineers

This course is designed for customer support engineers who need to understand data science fundamentals to effectively curate, evaluate, and deploy AI technology within a support organization. The curriculum follows an iterative, hands-on approach—each week builds on previous concepts while introducing new skills that can be immediately applied to real support scenarios.

---

## Course Philosophy

### Why "Fusion Engineers"?

Fusion Engineers represent a new breed of support professional who combines traditional customer support expertise with data science literacy. Rather than becoming full data scientists, Fusion Engineers develop enough fluency to:

- **Curate** training data and knowledge bases for AI systems
- **Evaluate** AI model performance using proper metrics
- **Collaborate** effectively with data science teams
- **Champion** AI adoption within support organizations
- **Identify** opportunities where AI can improve customer outcomes

### Iterative Learning Approach

Each week follows a consistent structure:

1. **Concept Introduction** — Core theory with support-relevant examples
2. **Hands-On Lab** — Practical exercises using real support data patterns
3. **AI Application** — How the concept applies to AI tools in support
4. **Assessment** — Knowledge check and practical application
5. **Bridge to Next Week** — Preview of upcoming concepts

---

> **Not sure if this course is for you?** Start with [Week 0: AI & LLMs — A Working Survey](Week00/README.md). It's a standalone 2-3 hour tour of how modern AI actually works, with an honest decision rubric at the end that will route you to this course, to [How Tuning Works](How-Tuning-Works/README.md), or to something else entirely. No prerequisites beyond curiosity.

---

## Prerequisites

### Required Background

- **Basic Computer Literacy**: Comfortable with spreadsheets (Excel/Google Sheets)
- **Support Domain Knowledge**: Understanding of ticket workflows, customer interactions, CSAT metrics
- **Curiosity About AI**: Motivation to understand how AI tools work "under the hood"

### Helpful But Not Required

- Basic familiarity with any programming language
- Experience with support analytics dashboards
- Exposure to chatbots or AI-assisted support tools

### Technical Setup

All tools are Microsoft-based for enterprise compatibility:

- **Visual Studio Code** with Python and Jupyter extensions (local development)
- **Python 3.x** installed locally (via Microsoft Store or python.org)
- **Microsoft Excel** for initial data exploration
- **Azure account** (free tier available) for cloud exercises
- Optional: **Azure Machine Learning Studio** for production scenarios
- Optional: **Power BI Desktop** for advanced visualization

---

## Course Overview

| Week | Theme | Key Skills | AI Application |
|------|-------|------------|----------------|
| 1 | [Foundations & Data Thinking](Week01/README.md) | Data types, variables, basic statistics | Understanding AI training data |
| 2 | [Exploratory Data Analysis](Week02/README.md) | Pandas, visualization, distributions | Analyzing support ticket patterns |
| 3 | [Statistical Foundations](Week03/README.md) | Probability, hypothesis testing, A/B tests | Evaluating AI experiments |
| 4 | [Classification & Confusion](Week04/README.md) | Confusion matrix, precision, recall, F1 | Measuring chatbot accuracy |
| 5 | [Text & Language Basics](Week05/README.md) | Text preprocessing, TF-IDF, embeddings | Understanding NLP in support |
| 6 | [Supervised Learning Concepts](Week06/README.md) | Training/test splits, overfitting, validation | How AI models learn from tickets |
| 7 | [Clustering & Categorization](Week07/README.md) | K-means, similarity, topic modeling | Auto-categorizing support issues |
| 8 | [Model Evaluation Deep Dive](Week08/README.md) | ROC/AUC, cross-validation, bias detection | Auditing AI for fairness |
| 8-Alt | [Real-World AI Eval & Chatbot Testing](Week08-Alt/README.md) | Task-based evals, simulated chats, LLM-as-Judge | Testing chatbots before deployment |
| 9 | [AI Systems in Production](Week09/README.md) | MLOps basics, monitoring, feedback loops | Maintaining AI quality over time |
| 10 | [Capstone & Future Learning](Week10/README.md) | End-to-end project, continued learning paths | Building your AI evaluation framework |

---

## Detailed Week-by-Week Curriculum

### Week 1: Foundations & Data Thinking

**Theme**: Building the mental models for thinking about data

**Skills Learned**:
- Understanding data types (categorical, numerical, ordinal)
- Variables: independent vs. dependent
- Basic descriptive statistics (mean, median, mode, standard deviation)
- Introduction to Python and Jupyter notebooks
- Loading and inspecting data with Pandas

**AI Application**:
- Why training data quality matters for AI
- How support tickets become AI training examples
- Data labeling fundamentals

**Lab Project**: Load a sample support ticket dataset, compute basic statistics, identify data quality issues

**Reference**: [DataScienceSurvey.md](../DataScienceSurvey.md) — Statistics Q1-Q8, Data Science Q1-Q6

[→ Week 1 Materials](Week01/README.md)

---

### Week 2: Exploratory Data Analysis

**Theme**: Discovering patterns and telling stories with data

**Skills Learned**:
- Data visualization with Matplotlib and Seaborn
- Understanding distributions (normal, skewed, bimodal)
- Correlation vs. causation
- Identifying outliers and anomalies
- Data cleaning and transformation

**AI Application**:
- Visualizing model performance over time
- Identifying data drift in support patterns
- Spotting seasonal trends affecting AI accuracy

**Lab Project**: Create a visual report of ticket volume, resolution times, and category distributions

**Reference**: [DataScienceSurvey.md](../DataScienceSurvey.md) — Data Analysis Q1-Q12

[→ Week 2 Materials](Week02/README.md)

---

### Week 3: Statistical Foundations

**Theme**: Making decisions with confidence

**Skills Learned**:
- Probability basics and conditional probability
- Hypothesis testing and p-values
- Type I and Type II errors
- Confidence intervals
- A/B testing methodology

**AI Application**:
- Designing experiments to test AI improvements
- Understanding statistical significance in AI evaluations
- When to trust (or distrust) AI performance claims

**Lab Project**: Design and analyze an A/B test comparing two AI response strategies

**Reference**: [DataScienceSurvey.md](../DataScienceSurvey.md) — Statistics Q3-Q6, Q10; Data Science Q9-Q16

[→ Week 3 Materials](Week03/README.md)

---

### Week 4: Classification & Confusion

**Theme**: Understanding how AI makes (and misses) predictions

**Skills Learned**:
- Binary and multi-class classification concepts
- Confusion matrix interpretation
- Precision, Recall, F1-Score, Accuracy
- When each metric matters most
- Trade-offs in classification thresholds

**AI Application**:
- Evaluating chatbot intent classification
- Understanding why some tickets get misrouted
- Setting appropriate thresholds for AI confidence

**Lab Project**: Build a confusion matrix for a ticket classifier, analyze error patterns

**Reference**: [DataScienceSurvey.md](../DataScienceSurvey.md) — Data Science Q4, Q15, Q25; Machine Learning Q24

[→ Week 4 Materials](Week04/README.md)

---

### Week 5: Text & Language Basics

**Theme**: How machines understand human language

**Skills Learned**:
- Text preprocessing (tokenization, stemming, lemmatization)
- Bag of Words and TF-IDF representations
- Introduction to word embeddings
- Sentiment analysis basics
- Named entity recognition concepts

**AI Application**:
- How chatbots understand customer messages
- Why similar questions get different responses
- Improving AI by understanding its text processing

**Lab Project**: Build a simple sentiment analyzer for customer feedback

**Reference**: [DataScienceSurvey.md](../DataScienceSurvey.md) — Q26 (TF-IDF); Top 100 NLP Questions; [NaturalLanguageProcessing.md](../NaturalLanguageProcessing.md)

[→ Week 5 Materials](Week05/README.md)

---

### Week 6: Supervised Learning Concepts

**Theme**: How AI learns from labeled examples

**Skills Learned**:
- Training, validation, and test set splits
- Overfitting and underfitting
- Cross-validation techniques
- Feature engineering basics
- Model selection intuition

**AI Application**:
- Why AI needs diverse training examples
- Recognizing when AI is memorizing vs. learning
- The role of human labeling in AI quality

**Lab Project**: Split a ticket dataset, train a simple classifier, evaluate generalization

**Reference**: [DataScienceSurvey.md](../DataScienceSurvey.md) — Machine Learning Q1-Q6, Q17-Q19; Data Science Q17-Q19

[→ Week 6 Materials](Week06/README.md)

---

### Week 7: Clustering & Categorization

**Theme**: Finding structure in unlabeled data

**Skills Learned**:
- Unsupervised vs. supervised learning
- K-means clustering algorithm
- Determining optimal cluster count
- Similarity measures (cosine, Euclidean)
- Topic modeling introduction (LDA)

**AI Application**:
- Auto-discovering ticket categories
- Grouping similar customer issues
- Building knowledge base taxonomies

**Lab Project**: Cluster support tickets to discover natural groupings

**Reference**: [DataScienceSurvey.md](../DataScienceSurvey.md) — Machine Learning Q3, Q23; Data Science Q1

[→ Week 7 Materials](Week07/README.md)

---

### Week 8: Model Evaluation Deep Dive

**Theme**: Beyond accuracy—comprehensive AI assessment

**Skills Learned**:
- ROC curves and AUC interpretation
- Precision-Recall curves
- Calibration and confidence scores
- Bias and fairness in AI systems
- Error analysis techniques

**AI Application**:
- Auditing AI for demographic fairness
- Understanding AI confidence scores
- Building comprehensive evaluation frameworks

**Lab Project**: Create an AI evaluation report with multiple metrics and bias analysis

**Reference**: [DataScienceSurvey.md](../DataScienceSurvey.md) — Data Science Q25; Machine Learning Q32, Q14; [BusinessAnalytics.md](../BusinessAnalytics.md)

[→ Week 8 Materials](Week08/README.md)

---

### Week 8 (Alternate): Real-World AI Evaluation & Chatbot Testing

**Theme**: Modern practices for evaluating AI in production

> This alternate module can be taken instead of or in addition to the standard Week 8. While Week 8 covers statistical evaluation metrics, this track covers operational evaluation — how to test whether AI systems actually work for real users.

**Skills Learned**:
- Real-world AI benchmarks (Remote Labor Index methodology)
- Task-based evaluation design
- Simulated user personas for chatbot testing
- Multi-turn conversation evaluation
- LLM-as-a-Judge for scoring at scale

**AI Application**:
- Testing chatbots with simulated conversations before deployment
- Building continuous evaluation pipelines for support AI
- Designing task libraries and scoring rubrics

**Lab Project**: Build a chatbot evaluation suite with task library, personas, simulated conversations, and scoring dashboard

**Reference**: [GenerativeAI.md](../GenerativeAI.md); [Scale AI RLI](https://scale.com/leaderboard/rli); [DeepEval](https://deepeval.com/docs/conversation-simulator)

[→ Week 8-Alt Materials](Week08-Alt/README.md)

---

### Week 9: AI Systems in Production

**Theme**: Keeping AI working well over time

**Skills Learned**:
- MLOps fundamentals
- Monitoring model performance
- Data drift and concept drift
- Feedback loops and continuous improvement
- Human-in-the-loop systems

**AI Application**:
- Setting up AI monitoring dashboards
- Detecting when AI quality degrades
- Designing escalation workflows

**Lab Project**: Design a monitoring plan for a support AI system

**Reference**: [MLOps.md](../MLOps.md); [DataScienceSurvey.md](../DataScienceSurvey.md) — Miscellaneous Q9-Q11, Q24-Q25

[→ Week 9 Materials](Week09/README.md)

---

### Week 10: Capstone & Future Learning

**Theme**: Putting it all together and planning your growth

**Skills Learned**:
- End-to-end AI evaluation project
- Communicating findings to stakeholders
- Building a personal learning roadmap
- Connecting to the broader data science community

**AI Application**:
- Complete evaluation of an AI support tool
- Recommendations for improvement
- Ongoing learning strategies

**Capstone Project**: Comprehensive evaluation of a support AI system including data analysis, performance metrics, bias audit, and recommendations

**Reference**: All previous materials; [Graduate Programs overview](../DataScienceSurvey.md#graduate-programs-in-data-science-subdisciplines)

[→ Week 10 Materials](Week10/README.md)

---

## Skills Matrix

### Technical Skills Progression

| Skill | Week Introduced | Mastery Level by End |
|-------|-----------------|---------------------|
| Python/Pandas basics | Week 1 | Intermediate |
| Data visualization | Week 2 | Intermediate |
| Statistical analysis | Week 3 | Foundational |
| Classification metrics | Week 4 | Strong |
| Text processing | Week 5 | Foundational |
| ML concepts | Week 6 | Foundational |
| Clustering | Week 7 | Foundational |
| Model evaluation | Week 8 | Strong |
| AI/chatbot operational testing | Week 8-Alt | Strong |
| MLOps awareness | Week 9 | Awareness |
| End-to-end analysis | Week 10 | Practical |

### Business Skills Developed

- **Critical Evaluation**: Ability to assess AI vendor claims
- **Data Storytelling**: Communicating insights to non-technical stakeholders
- **Experimental Design**: Setting up valid AI experiments
- **Quality Advocacy**: Championing data quality in AI projects
- **Cross-functional Collaboration**: Working effectively with data science teams

---

## Assessment Structure

### Weekly Assessments

Each week includes:
- **Knowledge Check** (10 questions) — Core concept verification
- **Lab Completion** — Hands-on exercise with deliverable
- **Reflection Journal** — Connection to real work scenarios

### Capstone Project (Week 10)

A comprehensive evaluation project that demonstrates:
1. Data exploration and quality assessment
2. Appropriate metric selection
3. Bias and fairness analysis
4. Clear communication of findings
5. Actionable recommendations

---

## Certification Path

Upon completing all weekly assessments and the capstone project, participants earn:

**Data Science for Fusion Engineers Certificate**

This internal certification validates competency in:
- AI evaluation methodology
- Data quality assessment
- Performance metrics interpretation
- Ethical AI considerations
- Cross-functional AI collaboration

---

## Additional Resources

### Reference Materials

- [Data Science Survey](../DataScienceSurvey.md) — Comprehensive Q&A reference
- [NLP Subdiscipline](../NaturalLanguageProcessing.md) — Deep dive into text/language AI
- [Business Analytics](../BusinessAnalytics.md) — Predictive modeling concepts
- [MLOps](../MLOps.md) — Production AI systems

### Recommended External Resources

- **Python**: [Python for Everybody](https://www.py4e.com/) (free online course)
- **Statistics**: Khan Academy Statistics & Probability
- **ML Concepts**: Google's Machine Learning Crash Course
- **AI Ethics**: Partnership on AI resources

### Community

- Weekly office hours with course facilitators
- Peer study groups
- Slack channel for questions and discussions
- Monthly guest speakers from data science teams

---

## Getting Started

1. **Complete the pre-course survey** to assess current knowledge
2. **Set up your technical environment** (instructions in Week 1)
3. **Join the course Slack channel**
4. **Review the Week 1 materials** before the first session

Welcome to your journey as a Fusion Engineer! 🚀

---

*This course is part of the Data Science knowledge base. For questions or feedback, contact the Learning & Development team.*
