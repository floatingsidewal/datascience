# Week 5: Text & Language Basics

## How Machines Understand Human Language

This week we explore Natural Language Processing (NLP)—the technology that allows AI to understand customer messages. You'll learn how text becomes numbers and why "similar" messages sometimes get different AI responses.

---

## Learning Objectives

By the end of this week, you will be able to:

1. Explain text preprocessing steps and their purpose
2. Understand Bag of Words and TF-IDF representations
3. Describe what word embeddings are and why they matter
4. Implement basic sentiment analysis
5. Identify NER (Named Entity Recognition) use cases
6. Troubleshoot why AI might misunderstand customer messages

---

## Core Concepts

### 1. The Challenge: Text to Numbers

Computers work with numbers, not words. NLP converts text into numerical representations that capture meaning.

**Customer Message**: "I can't login to my account since yesterday"

**The AI needs to understand**:
- This is about account access (intent)
- "login" and "account" are key entities
- "yesterday" is a time reference
- The sentiment is frustrated/negative

### 2. Text Preprocessing Pipeline

Before AI can analyze text, it needs cleaning and standardization.

| Step | What It Does | Example |
|------|--------------|---------|
| **Lowercasing** | Normalize case | "Login" → "login" |
| **Tokenization** | Split into words | "can't login" → ["can't", "login"] |
| **Punctuation Removal** | Remove noise | "help!" → "help" |
| **Stop Word Removal** | Remove common words | "I can not login" → "login" |
| **Stemming** | Reduce to root form | "running", "runs" → "run" |
| **Lemmatization** | Dictionary-based root | "better" → "good" |

```python
# Example preprocessing pipeline
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_ticket(text):
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return tokens

# Example
message = "I've been waiting for 3 days and still can't access my account!"
print(preprocess_ticket(message))
# Output: ['waiting', '3', 'day', 'still', 'access', 'account']
```

### 3. Bag of Words (BoW)

The simplest text representation: count word occurrences.

**Messages**:
1. "reset my password please"
2. "I forgot my password"
3. "billing question about invoice"

**Vocabulary**: [reset, my, password, please, i, forgot, billing, question, about, invoice]

**BoW Matrix**:
| Message | reset | my | password | please | i | forgot | billing | question | about | invoice |
|---------|-------|-----|----------|--------|---|--------|---------|----------|-------|---------|
| 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |

**Limitation**: Loses word order and context. "Dog bites man" = "Man bites dog" in BoW.

### 4. TF-IDF: Term Frequency - Inverse Document Frequency

Weights words by importance: common words get lower scores, distinctive words get higher scores.

**Formula**:
```
TF-IDF = TF × IDF

TF (Term Frequency) = (times word appears in document) / (total words in document)
IDF (Inverse Document Frequency) = log(total documents / documents containing word)
```

**Why It Matters**:
- "the", "is", "a" → Low TF-IDF (appear everywhere)
- "password", "invoice" → High TF-IDF (distinctive)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

messages = [
    "I need to reset my password",
    "How do I change my password",
    "Question about my invoice total",
    "Invoice payment not working"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(messages)

# View feature names and scores
feature_names = vectorizer.get_feature_names_out()
print("Features:", feature_names)
print("\nTF-IDF Matrix:")
print(pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names).round(2))
```

### 5. Word Embeddings

Dense vector representations that capture semantic meaning. Similar words have similar vectors.

**Key Insight**: Embeddings encode relationships:
- "king" - "man" + "woman" ≈ "queen"
- "password" and "login" are close together
- "billing" and "invoice" are close together

**Popular Embedding Models**:
- **Word2Vec**: Classic, word-level embeddings
- **GloVe**: Global vectors, trained on co-occurrence
- **BERT**: Contextual embeddings (same word can have different vectors)
- **Sentence Transformers**: Document-level embeddings

```python
# Using sentence transformers for semantic similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

messages = [
    "I can't log in to my account",
    "Unable to access my profile",
    "What's the price of premium plan",
    "Login not working"
]

embeddings = model.encode(messages)

# Calculate similarity between messages
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(embeddings)
print("Similarity Matrix:")
print(pd.DataFrame(similarity_matrix,
                   index=range(1, 5),
                   columns=range(1, 5)).round(2))

# Messages 1, 2, and 4 should have high similarity (all about login)
# Message 3 should be different (about pricing)
```

### 6. Sentiment Analysis

Determining the emotional tone of text.

**Levels of Sentiment**:
- **Binary**: Positive / Negative
- **Ternary**: Positive / Neutral / Negative
- **Fine-grained**: Very Negative → Very Positive (1-5 scale)
- **Aspect-based**: Sentiment toward specific aspects (product: positive, support: negative)

---

## Why This Matters for AI

### Understanding Chatbot Failures

When customers ask the same question differently, AI might respond differently:

| Customer Message | AI Understanding | Issue |
|-----------------|------------------|-------|
| "Can't login" | Intent: Login Issue ✓ | Works |
| "Unable to access" | Intent: Login Issue ✓ | Works |
| "My account is locked out" | Intent: Account Issue ✗ | Different training examples |
| "Getting error on sign in" | Intent: Error Report ✗ | "Error" triggers different intent |

### Improving AI Through Better Training Data

As a Fusion Developer, you can improve AI by:

1. **Identifying synonyms**: Ensure training data includes variations
2. **Reviewing misclassifications**: Find patterns in AI failures
3. **Expanding entity coverage**: Add new product names, features
4. **Balancing sentiment examples**: Include all tones in training

---

## Hands-On Lab: Text Analysis for Support

### Lab Exercise

```python
# Week 5 Lab: NLP for Support Tickets
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Sample support tickets
tickets = [
    "I cannot log into my account",
    "Password reset not working",
    "Unable to access my profile since yesterday",
    "Login page shows error",
    "Need to change my password",
    "My invoice shows wrong amount",
    "Billing charge is incorrect",
    "Question about my recent invoice",
    "How much does premium cost",
    "Your app crashes when I open it",
    "Application keeps freezing",
    "Software bug causing data loss"
]

categories = ['Login', 'Login', 'Login', 'Login', 'Login',
              'Billing', 'Billing', 'Billing', 'Billing',
              'Technical', 'Technical', 'Technical']

# Task 1: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(tickets)

print("=== TF-IDF Features ===")
feature_names = vectorizer.get_feature_names_out()
print(f"Vocabulary size: {len(feature_names)}")
print(f"Features: {list(feature_names)}")

# Task 2: Find similar tickets
print("\n=== Ticket Similarity ===")
similarity_matrix = cosine_similarity(tfidf_matrix)

# Find most similar pair
np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity
most_similar = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
print(f"Most similar tickets: {most_similar[0]+1} and {most_similar[1]+1}")
print(f"  Ticket {most_similar[0]+1}: {tickets[most_similar[0]]}")
print(f"  Ticket {most_similar[1]+1}: {tickets[most_similar[1]]}")
print(f"  Similarity: {similarity_matrix[most_similar]:.2f}")

# Task 3: Visualize similarity within and across categories
plt.figure(figsize=(10, 8))
plt.imshow(similarity_matrix, cmap='YlOrRd')
plt.colorbar(label='Cosine Similarity')
plt.xlabel('Ticket Index')
plt.ylabel('Ticket Index')
plt.title('Ticket Similarity Matrix')

# Add category boundaries
plt.axhline(y=4.5, color='blue', linestyle='--', linewidth=2)
plt.axhline(y=8.5, color='blue', linestyle='--', linewidth=2)
plt.axvline(x=4.5, color='blue', linestyle='--', linewidth=2)
plt.axvline(x=8.5, color='blue', linestyle='--', linewidth=2)

plt.tight_layout()
plt.show()

# Task 4: Simple keyword-based sentiment
positive_words = ['thanks', 'great', 'love', 'excellent', 'helpful', 'amazing']
negative_words = ['cannot', 'error', 'wrong', 'incorrect', 'crash', 'bug',
                  'freezing', 'not working', 'problem', 'issue', 'frustrated']

def simple_sentiment(text):
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    if pos_count > neg_count:
        return 'Positive'
    elif neg_count > pos_count:
        return 'Negative'
    else:
        return 'Neutral'

print("\n=== Simple Sentiment Analysis ===")
for ticket in tickets:
    sentiment = simple_sentiment(ticket)
    print(f"{sentiment:8} | {ticket}")

# Task 5: Identify potential routing issues
print("\n=== Cross-Category Similarity (Potential Routing Issues) ===")
# Find tickets from different categories that are similar
for i in range(len(tickets)):
    for j in range(i+1, len(tickets)):
        if categories[i] != categories[j] and similarity_matrix[i,j] > 0.3:
            print(f"Similarity {similarity_matrix[i,j]:.2f}:")
            print(f"  [{categories[i]}] {tickets[i]}")
            print(f"  [{categories[j]}] {tickets[j]}")
            print()
```

### Lab Questions

1. Which tickets have the highest similarity to each other?
2. Looking at the similarity matrix, do tickets within the same category cluster together?
3. Are there any cross-category pairs that might confuse a routing AI?
4. What words would you add to improve the sentiment analysis?

---

## Named Entity Recognition (NER)

### What is NER?

Identifying and classifying named entities in text:
- **PERSON**: Customer names
- **ORG**: Company names
- **PRODUCT**: Product/feature names
- **DATE/TIME**: Temporal references
- **MONEY**: Amounts, prices

### Support Use Cases

| Entity Type | Example | Use Case |
|-------------|---------|----------|
| PRODUCT | "Slack integration" | Route to product team |
| DATE | "since yesterday" | Calculate issue duration |
| MONEY | "$99.99 charge" | Flag for billing review |
| ORG | "Acme Corp" | Link to account |

```python
import spacy

nlp = spacy.load("en_core_web_sm")

ticket = "John from Acme Corp reported that the Slack integration broke yesterday and they were charged $500."

doc = nlp(ticket)

print("Entities found:")
for ent in doc.ents:
    print(f"  {ent.text:20} → {ent.label_}")
```

---

## Knowledge Check

1. The purpose of TF-IDF is to:
   - a) Count how many times each word appears
   - b) Weight words by their distinctiveness across documents
   - c) Convert text to audio
   - d) Correct spelling errors

2. Word embeddings capture:
   - a) Word frequency
   - b) Semantic meaning and relationships
   - c) Grammar rules
   - d) Document length

3. If two messages have a cosine similarity of 0.95, they are:
   - a) Completely different
   - b) Very similar
   - c) Exactly identical
   - d) Opposite in meaning

4. Lemmatization differs from stemming because it:
   - a) Is faster
   - b) Uses dictionary lookups for proper root forms
   - c) Removes more words
   - d) Works only on nouns

*(Answers: 1-b, 2-b, 3-b, 4-b)*

---

## Reflection Journal

1. Think of customer messages that your current AI handles poorly. What NLP challenges might explain this?

2. How could better entity recognition improve your support workflows?

3. What synonym variations would you add to training data for your most common intents?

---

## Bridge to Week 6

Next week, we explore **supervised learning concepts**—how AI actually learns from labeled examples. You'll understand:
- Why training data needs to be split
- What overfitting means and why it's dangerous
- The human role in creating quality training data

**Preparation**: Think about how your organization creates and maintains AI training data. What's the process for adding new examples?

---

## Additional Resources

### From the Data Science Survey
- [TF-IDF (Q26)](../../DataScienceSurvey.md#data-science) — Detailed explanation
- [Top 100 NLP Questions](../../DataScienceSurvey.md#top-100-nlp-questions) — Comprehensive NLP reference

### Related Subdiscipline
- [Natural Language Processing](../../NaturalLanguageProcessing.md) — Graduate programs and deep dive

### External Resources
- [spaCy 101](https://spacy.io/usage/spacy-101)
- [Hugging Face NLP Course](https://huggingface.co/course)
- [NLTK Book (free)](https://www.nltk.org/book/)

---

[← Week 4: Classification](../Week04/README.md) | [Back to Course Overview](../DataScienceForFusionEngineers.md) | [Week 6: Supervised Learning →](../Week06/README.md)
