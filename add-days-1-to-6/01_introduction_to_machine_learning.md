# Introduction to Machine Learning

## What is Machine Learning?

Machine Learning (ML) is a subset of **Artificial Intelligence (AI)** that enables systems to **learn from data** and improve their performance over time — without being explicitly programmed for each task.

> "Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed."
> — Arthur Samuel, 1959

### Core Idea
Instead of writing rules manually, we **feed data** to an algorithm, and the algorithm **learns the patterns** on its own.

---

## Types of Machine Learning

| Type | Description | Example |
|---|---|---|
| **Supervised Learning** | Learns from labeled data (input → output pairs) | Spam detection, House price prediction |
| **Unsupervised Learning** | Finds patterns in unlabeled data | Customer segmentation, Topic modeling |
| **Reinforcement Learning** | Agent learns by interacting with environment & getting rewards/penalties | Game playing (Chess, Go), Robotics |

---

## How ML Works — The General Pipeline

```
Raw Data → Data Preprocessing → Feature Engineering
        → Model Selection → Training → Evaluation → Deployment
```

1. **Collect Data** — gather relevant dataset
2. **Preprocess Data** — handle missing values, normalize, encode categories
3. **Choose a Model** — linear regression, decision tree, SVM, etc.
4. **Train the Model** — model learns patterns from training data
5. **Evaluate** — test on unseen data (accuracy, precision, recall, F1)
6. **Deploy** — use the model in real-world applications

---

## Common ML Algorithms

- **Linear Regression** — predict continuous values
- **Logistic Regression** — binary classification
- **Decision Trees** — rule-based classification/regression
- **Random Forest** — ensemble of decision trees
- **Support Vector Machine (SVM)** — classification with margin maximization
- **K-Nearest Neighbors (KNN)** — classify based on nearest data points
- **K-Means** — unsupervised clustering

---

## Machine Learning vs Deep Learning

### Machine Learning (Traditional ML)

- Works well with **structured/tabular data**
- Requires **manual feature engineering** (humans select important features)
- Uses algorithms like SVM, Decision Trees, Random Forest
- Works with **smaller datasets** (thousands to hundreds of thousands of records)
- Models are generally **interpretable** (you can understand why a prediction was made)
- **Faster to train** on standard hardware (CPU is sufficient)

### Deep Learning (DL)

- A **subset of Machine Learning** using **Artificial Neural Networks (ANNs)** with many layers
- Works well with **unstructured data** — images, audio, text, video
- Does **automatic feature extraction** — learns features directly from raw data
- Requires **large datasets** (millions of examples) to perform well
- Models are often a **"black box"** — hard to interpret
- Requires **more compute power** — GPUs/TPUs often necessary
- Achieves **state-of-the-art results** in computer vision, NLP, speech recognition

---

## ML vs DL — Quick Comparison Table

| Feature | Machine Learning | Deep Learning |
|---|---|---|
| **Data Requirement** | Small to Medium | Large (millions of samples) |
| **Feature Engineering** | Manual (human-driven) | Automatic (learned from data) |
| **Compute Requirement** | CPU sufficient | GPU/TPU recommended |
| **Interpretability** | High (mostly) | Low (black box) |
| **Training Time** | Faster | Slower |
| **Best For** | Structured/tabular data | Images, Audio, Text, Video |
| **Examples** | Random Forest, SVM, XGBoost | CNN, RNN, Transformers, GANs |
| **Performance on small data** | Better | Worse |
| **Performance on big data** | Saturates | Keeps improving |

---

## The AI Hierarchy

```
Artificial Intelligence (AI)
    └── Machine Learning (ML)
            └── Deep Learning (DL)
                    └── Generative AI (LLMs, Diffusion Models)
```

- **AI** — broad concept of machines performing intelligent tasks
- **ML** — machines that learn from data
- **DL** — ML using deep neural networks
- **Generative AI** — DL models that can generate new content (text, images, code)

---

## When to Use ML vs DL?

**Use ML when:**
- Dataset is small or medium-sized
- Data is structured/tabular (spreadsheets, databases)
- Interpretability is important (healthcare, finance)
- Limited compute resources
- You need a quick baseline

**Use Deep Learning when:**
- Data is unstructured (images, audio, text)
- Dataset is very large
- Task complexity is high (translation, image recognition)
- Compute resources are available
- High accuracy is the priority

---

## Key Terms to Remember

| Term | Definition |
|---|---|
| **Feature** | Input variable used for prediction |
| **Label / Target** | Output variable we want to predict |
| **Training Data** | Data used to train the model |
| **Test Data** | Unseen data used to evaluate model |
| **Overfitting** | Model memorizes training data, fails on new data |
| **Underfitting** | Model is too simple to capture patterns |
| **Epoch** | One full pass through the training dataset |
| **Loss Function** | Measures how wrong the model's predictions are |
| **Gradient Descent** | Optimization algorithm to minimize loss |
| **Neural Network** | Layers of interconnected nodes inspired by the brain |

---

## Summary

- **Machine Learning** = teaching machines to learn from data without explicit programming
- It has 3 main types: Supervised, Unsupervised, Reinforcement Learning
- **Deep Learning** is a *subset* of ML using neural networks with many layers
- ML is best for structured data + small datasets; DL shines with unstructured data + big datasets
- The hierarchy: AI ⊃ ML ⊃ DL ⊃ Generative AI

---

*Next Topic →* Neural Networks & Deep Learning Architecture
