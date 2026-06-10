# 📰 Fake News Detection using Contrastive Learning & LLM Embeddings

A semantic fake news detection system that leverages **pre-trained Large Language Model (LLM) embeddings**, **contrastive learning**, and a lightweight **Multi-Layer Perceptron (MLP)** classifier to accurately distinguish between real and fake news articles.

Instead of fine-tuning an entire LLM, this project learns a better representation space through contrastive learning, making classification more robust, efficient, and computationally inexpensive.

---

## 🚀 Overview

Traditional text classification models often rely solely on supervised learning, which may not fully capture semantic relationships between articles.

This project addresses that by:

- Generating semantic embeddings using a pre-trained LLM
- Learning a representation space where similar news articles are closer together
- Separating fake and real news in the embedding space using contrastive learning
- Using an MLP classifier for efficient final prediction

---

## 🏗️ Architecture

```text
                News Article
                      │
                      ▼
        ┌───────────────────────────┐
        │  Pre-trained LLM Encoder  │
        └───────────────────────────┘
                      │
               Semantic Embedding
                  (e.g. 768-d)
                      │
                      ▼
          Contrastive Learning Stage
      (Pull similar samples together,
       push different samples apart)
                      │
                      ▼
          Optional Projection Head
                      │
                      ▼
            Refined Embedding Space
                      │
                      ▼
          Multi-Layer Perceptron (MLP)
                      │
                      ▼
             Fake  ✅ / Real ✅
```

---

# ⚙️ Methodology

## 1. Text → LLM Embeddings

Each news article is passed through a pre-trained language model to obtain a dense semantic representation.

```text
News Text
      │
      ▼

 Pre-trained LLM

      │
      ▼

768-dimensional embedding
```

These embeddings capture contextual and semantic information beyond traditional bag-of-words or TF-IDF representations.

---

## 2. Contrastive Learning

Instead of directly classifying embeddings, contrastive learning reshapes the embedding space.

The objective is:

- ✅ Real ↔ Real → closer
- ✅ Fake ↔ Fake → closer
- ❌ Real ↔ Fake → farther apart

```text
Before Training:

Real ●      Fake ●

Real ●             Fake ●


After Contrastive Learning:

        Real ● ● ●




                    Fake ● ● ●
```

Possible loss functions:

- Contrastive Loss
- NT-Xent Loss
- Triplet Loss

This improves representation quality and creates a classification-friendly embedding space.

---

## 3. Projection Head

A lightweight projection head is optionally added during contrastive training.

```text
Embedding
      │

Linear Layer

      │

Projection Space
```

Benefits:

- Keeps the LLM frozen
- Faster training
- Better representation learning
- Reduces overfitting
- Improves contrastive objective performance

The projection head is only used during representation learning and does not affect inference complexity.

---

## 4. MLP Classifier

The refined embeddings are then passed through a lightweight Multi-Layer Perceptron (MLP) for final classification.

```text
Refined Embedding

        │

      Dense

        │

      ReLU

        │

      Dense

        │

 Fake / Real
```

### Why MLP?

- Captures non-linear decision boundaries
- Computationally efficient
- Faster than end-to-end LLM fine-tuning
- Works effectively on dense semantic embeddings

---

## 5. Final Prediction

The complete inference pipeline:

```text
News Article
      │
      ▼

Pre-trained LLM

      │
      ▼

Semantic Embedding

      │
      ▼

Contrastively Learned Representation

      │
      ▼

MLP Classifier

      │
      ▼

Prediction

   Fake ❌
      or
   Real ✅
```

---

# ✨ Key Features

- 🧠 Pre-trained LLM semantic embeddings
- 🔥 Contrastive representation learning
- 📌 Projection head for improved embedding quality
- 🚀 Lightweight MLP classifier
- ⚡ Efficient training without full LLM fine-tuning
- 🎯 Better semantic separation between fake and real news

---

# 🛠️ Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Contrastive Learning
- Multi-Layer Perceptron (MLP)
- NumPy
- Pandas
- Scikit-learn

---

# 📈 Future Improvements

- Fine-tuning transformer encoders
- Hard negative mining
- Data augmentation for contrastive learning
- Multi-modal fake news detection (Text + Images)
- Explainable AI with attention visualization
- Knowledge graph integration
- Cross-domain generalization

---

# 🎯 Motivation

Fake news often relies on subtle semantic manipulation rather than obvious lexical differences. By learning a discriminative semantic representation through contrastive learning, the model becomes more robust and better equipped to distinguish misleading information from genuine news.

---

## 📜 License

This project is intended for research and educational purposes. Feel free to explore, improve, and build upon it.