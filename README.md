# Sentiment Analysis with Multinomial Naive Bayes

This project implements a simple yet effective sentiment analysis pipeline using tokenization, TF-IDF vectorization, and a Multinomial Naive Bayes classifier.

## Overview

Given a dataset of labeled text samples (e.g., reviews), the model learns to classify whether the sentiment is **positive** or **negative**.

### Technologies Used
- Python 3
- scikit-learn
- pandas
- numpy

###  Model Architecture
- **Tokenizer:** Custom tokenizer with optional stopword removal
- **Embedder:** TF-IDF vectorizer (max_features = 5000)
- **Classifier:** Multinomial Naive Bayes
