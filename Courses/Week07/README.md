# Week 7: Clustering & Categorization

## Finding Structure in Unlabeled Data

This week we explore unsupervised learning—how AI discovers patterns without being told the answers. Clustering helps automatically group similar items, which is invaluable for organizing support data and discovering hidden structures.

---

## Learning Objectives

By the end of this week, you will be able to:

1. Distinguish between supervised and unsupervised learning
2. Apply K-means clustering to support data
3. Determine the optimal number of clusters
4. Use similarity measures to compare items
5. Apply topic modeling concepts to text data
6. Leverage clustering for knowledge base organization

---

## Core Concepts

### 1. Supervised vs. Unsupervised Learning

| Aspect | Supervised | Unsupervised |
|--------|------------|--------------|
| **Requires Labels** | Yes | No |
| **Goal** | Predict labels | Discover structure |
| **Example Task** | Classify ticket category | Group similar tickets |
| **Evaluation** | Compare to ground truth | Internal metrics, human review |
| **Use Case** | Routing, escalation prediction | Discovery, organization, anomaly detection |

### 2. K-Means Clustering

The most common clustering algorithm. Groups data into K clusters by minimizing distance to cluster centers.

#### Algorithm Steps

```
1. Choose K (number of clusters)
2. Randomly initialize K cluster centers (centroids)
3. Repeat until convergence:
   a. Assign each point to nearest centroid
   b. Update centroids to mean of assigned points
```

#### Visual Intuition

```
Initial:                    After Clustering:
    ·  ·  x                     ○  ○  x
  ·  ·     ·                  ○  ○     ·
    ·        · x                ○        □ x
  ·  ·    ·  ·                ○  ○    □  □
    · ·       ·                 ○ ○       □
         x  ·  ·                     x  □  □

x = centroid                ○ = Cluster 1, □ = Cluster 2
```

### 3. Choosing K: The Elbow Method

Plot "inertia" (within-cluster sum of squares) vs. K. Look for the "elbow" where adding more clusters doesn't help much.

```python
from sklearn.cluster import KMeans

inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
```

### 4. Similarity Measures

How do we determine if two items are "similar"?

| Measure | Formula | Best For |
|---------|---------|----------|
| **Euclidean Distance** | √Σ(a-b)² | Numerical features |
| **Cosine Similarity** | (a·b)/(‖a‖‖b‖) | Text/TF-IDF vectors |
| **Jaccard Similarity** | \|A∩B\|/\|A∪B\| | Sets, categories |

#### Cosine Similarity for Text

```python
from sklearn.metrics.pairwise import cosine_similarity

# Two ticket TF-IDF vectors
ticket_1 = [0.5, 0.3, 0.0, 0.8]  # "login", "password", "billing", "account"
ticket_2 = [0.6, 0.4, 0.0, 0.7]  # Similar topic

similarity = cosine_similarity([ticket_1], [ticket_2])[0][0]
print(f"Similarity: {similarity:.2f}")  # High value = similar
```

### 5. Topic Modeling (LDA)

Latent Dirichlet Allocation discovers topics in document collections.

**Input**: Collection of documents
**Output**: Topics (word distributions) and document-topic assignments

```
Topic 1 (Login Issues):
  "password" (0.15), "login" (0.12), "account" (0.10), "reset" (0.08)...

Topic 2 (Billing):
  "invoice" (0.18), "charge" (0.14), "payment" (0.12), "refund" (0.09)...
```

---

## Why This Matters for AI

### Use Cases in Support

| Application | How Clustering Helps |
|-------------|---------------------|
| **Knowledge Base Organization** | Discover natural article groupings |
| **Ticket Taxonomy** | Find categories that exist in the data |
| **Similar Ticket Search** | Retrieve related past tickets |
| **Anomaly Detection** | Identify tickets that don't fit any cluster |
| **Customer Segmentation** | Group customers by behavior patterns |
| **Agent Workload Balancing** | Understand ticket complexity clusters |

### Discovery vs. Classification

**When to use Classification (Supervised)**:
- You have well-defined categories
- Historical labeled data exists
- Categories are unlikely to change

**When to use Clustering (Unsupervised)**:
- Categories are unknown or evolving
- You want to discover what's in the data
- Labels are expensive or unavailable
- You want to validate existing categories

---

## Hands-On Lab: Clustering Support Tickets

### Lab Exercise

```python
# Week 7 Lab: Clustering Support Tickets
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Sample ticket data (no labels!)
tickets = [
    # Cluster: Login/Account
    "Cannot log into my account",
    "Password reset link not working",
    "Account locked after failed attempts",
    "Two-factor authentication issues",
    "Forgot my username",
    "Can't access my profile",
    "Login page shows error",
    "Session keeps timing out",

    # Cluster: Billing
    "Incorrect charge on statement",
    "Need a refund please",
    "Invoice shows wrong amount",
    "Double charged this month",
    "Cancel my subscription",
    "Update payment method",
    "Billing cycle question",
    "Promotional discount not applied",

    # Cluster: Technical/Product
    "App crashes on startup",
    "Feature not working correctly",
    "Bug in the export function",
    "Slow performance issues",
    "Error when saving data",
    "Integration not syncing",
    "Mobile app freezes",
    "Data not loading properly",

    # Cluster: Feature Requests (might emerge)
    "Would like dark mode",
    "Can you add export to PDF",
    "Suggestion for improvement",
    "Feature request for calendar view"
]

print(f"Total tickets: {len(tickets)}")

# Task 1: Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
X = vectorizer.fit_transform(tickets)

print(f"Feature matrix shape: {X.shape}")
print(f"Features: {list(vectorizer.get_feature_names_out())[:10]}...")

# Task 2: Find optimal K using Elbow Method
inertias = []
silhouettes = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X, kmeans.labels_))

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')

axes[1].plot(K_range, silhouettes, 'go-')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Analysis')

plt.tight_layout()
plt.show()

# Task 3: Cluster with chosen K
optimal_k = 4  # Based on elbow/silhouette analysis
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Task 4: Analyze clusters
print(f"\n=== Cluster Analysis (K={optimal_k}) ===")

for cluster_id in range(optimal_k):
    cluster_tickets = [tickets[i] for i in range(len(tickets)) if clusters[i] == cluster_id]

    print(f"\n--- Cluster {cluster_id} ({len(cluster_tickets)} tickets) ---")
    for ticket in cluster_tickets[:5]:  # Show first 5
        print(f"  • {ticket}")
    if len(cluster_tickets) > 5:
        print(f"  ... and {len(cluster_tickets) - 5} more")

    # Find top terms for this cluster
    cluster_center = kmeans.cluster_centers_[cluster_id]
    top_indices = cluster_center.argsort()[-5:][::-1]
    top_terms = [vectorizer.get_feature_names_out()[i] for i in top_indices]
    print(f"  Top terms: {', '.join(top_terms)}")

# Task 5: Find similar tickets to a new query
def find_similar_tickets(query, tickets, vectorizer, kmeans, top_n=3):
    """Find tickets in the same cluster as the query"""
    query_vec = vectorizer.transform([query])
    query_cluster = kmeans.predict(query_vec)[0]

    # Get all tickets in same cluster
    same_cluster = [tickets[i] for i in range(len(tickets))
                    if clusters[i] == query_cluster]

    print(f"Query: '{query}'")
    print(f"Assigned to Cluster {query_cluster}")
    print(f"Similar tickets in cluster:")
    for ticket in same_cluster[:top_n]:
        print(f"  • {ticket}")

print("\n=== Similar Ticket Search ===")
find_similar_tickets("I can't sign in to my account", tickets, vectorizer, kmeans)

# Task 6: Identify outliers (tickets far from all centroids)
from sklearn.metrics import pairwise_distances

distances = pairwise_distances(X, kmeans.cluster_centers_, metric='euclidean')
min_distances = distances.min(axis=1)

# Tickets with high minimum distance are potential outliers
outlier_threshold = np.percentile(min_distances, 90)
outliers = [i for i, d in enumerate(min_distances) if d > outlier_threshold]

print(f"\n=== Potential Outliers (hard to classify) ===")
for idx in outliers:
    print(f"  • {tickets[idx]} (distance: {min_distances[idx]:.3f})")
```

### Lab Questions

1. Based on the elbow and silhouette plots, what's the optimal K?
2. Looking at the cluster contents, what label would you give each cluster?
3. Do any tickets seem misplaced? Why might that be?
4. What terms best distinguish each cluster?

---

## Topic Modeling with LDA

### When to Use LDA vs. K-Means

| K-Means | LDA |
|---------|-----|
| Each document belongs to ONE cluster | Each document can have MULTIPLE topics |
| Works on any features | Specifically designed for text |
| Finds groups | Discovers themes |

### Example: Discovering Topics in Tickets

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# For LDA, use count vectors (not TF-IDF)
count_vectorizer = CountVectorizer(stop_words='english', max_features=100)
X_counts = count_vectorizer.fit_transform(tickets)

# Fit LDA
n_topics = 4
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X_counts)

# Display top words per topic
feature_names = count_vectorizer.get_feature_names_out()

print("=== Discovered Topics ===")
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[-8:][::-1]]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")
```

---

## Practical Applications

### Building a Ticket Router Using Clusters

```python
class ClusterBasedRouter:
    def __init__(self, tickets, labels=None):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.X = self.vectorizer.fit_transform(tickets)
        self.kmeans = KMeans(n_clusters=4, random_state=42)
        self.kmeans.fit(self.X)

        # Optionally map clusters to queues
        self.cluster_to_queue = {
            0: "Account Team",
            1: "Billing Team",
            2: "Technical Support",
            3: "Product Feedback"
        }

    def route(self, new_ticket):
        vec = self.vectorizer.transform([new_ticket])
        cluster = self.kmeans.predict(vec)[0]
        return self.cluster_to_queue.get(cluster, "General Queue")

router = ClusterBasedRouter(tickets)
print(router.route("I need help with my invoice"))  # → Billing Team
print(router.route("The app keeps crashing"))        # → Technical Support
```

---

## Knowledge Check

1. Unsupervised learning differs from supervised learning because:
   - a) It requires more data
   - b) It doesn't need labeled training data
   - c) It's more accurate
   - d) It's faster to train

2. In K-means clustering, K represents:
   - a) The number of features
   - b) The number of data points
   - c) The number of clusters to create
   - d) The learning rate

3. The "elbow method" helps you:
   - a) Choose the right features
   - b) Choose the optimal number of clusters
   - c) Clean the data
   - d) Evaluate model accuracy

4. A silhouette score close to 1 indicates:
   - a) Poor clustering
   - b) Well-defined, separated clusters
   - c) Too many clusters
   - d) Overfitting

*(Answers: 1-b, 2-c, 3-b, 4-b)*

---

## Reflection Journal

1. Could clustering reveal better ticket categories than your current taxonomy? How would you validate this?

2. Think of a support scenario where discovering hidden patterns would be valuable. What data would you cluster?

3. How might you combine clustering (unsupervised) with classification (supervised) in a workflow?

---

## Bridge to Week 8

Next week, we dive deep into **model evaluation**—going beyond basic metrics to comprehensive assessment. You'll learn:
- ROC curves and AUC interpretation
- How to detect bias in AI systems
- Building thorough evaluation frameworks

**Preparation**: Think about different customer segments in your support data. Does your AI perform equally well for all of them?

---

## Additional Resources

### From the Data Science Survey
- [Machine Learning Q3](../../DataScienceSurvey.md#machine-learning) — Unsupervised learning
- [Machine Learning Q23](../../DataScienceSurvey.md#machine-learning) — Clustering

### External Resources
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Interactive K-means Visualization](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)

---

[← Week 6: Supervised Learning](../Week06/README.md) | [Back to Course Overview](../DataScienceForFusionEngineers.md) | [Week 8: Model Evaluation →](../Week08/README.md)
