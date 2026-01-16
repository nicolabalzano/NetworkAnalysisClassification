# NetworkAnalysisClassification

This repository contains the exam project for the **AI for Security** course. The project explores the use of unsupervised learning (K-Means Clustering) to generate "geometric context" features and evaluates their impact on the performance of deep learning models for network traffic classification.

## 📖 Project Overview

The core objective was to determine if augmenting a dataset with spatial information (Euclidean distances to cluster centroids) could enhance the predictive power of a Neural Network compared to traditional classifiers like Random Forest and standard Multi-Layer Perceptrons (MLP).

### Key Phases:

1. **Exploratory Data Analysis & Cleaning:** Dimensionality reduction and handling class imbalance.
2. **Baseline Modeling:** Training a Random Forest and a standard MLP.
3. **Unsupervised Feature Engineering:** Optimizing K-Means using Purity and the Elbow Method.
4. **Augmented Modeling:** Training an MLP on the dataset expanded with geometric features.
5. **Comparative Analysis:** Performance benchmarking across all architectures.

---

## 📊 Dataset & Pre-processing

The dataset consists of network traffic flows with 79 initial features and 5 target classes.

* **Cleaning:** 12 non-informative features (zero variance or constant values) were removed, reducing the feature set to 66.
* **Scaling:** `MinMaxScaler` was applied to prevent large-scale temporal features from dominating distance calculations.
* **Class Imbalance:** Significant imbalance was noted (Class 0 being the majority), requiring the use of **Macro F1-Score** as a primary evaluation metric.

---

## 🤖 Models & Methodology

### 1. Random Forest Classifier

Optimized via Stratified 5-Fold Cross-Validation.

* **Result:** Achieved the highest accuracy (**99.40%**).
* **Strength:** Highly robust to non-linear boundaries and feature overlap without requiring complex scaling.

### 2. Multi-Layer Perceptron (MLP)

A baseline deep learning model with three hidden layers (128, 64, 32 neurons).

* **Result:** **98.70% Accuracy**.
* **Observation:** Showed slight symmetrical confusion between Class 1 and Class 3.

### 3. K-Means Augmented Neural Network

The dataset was augmented with **23 new features** representing the Euclidean distance of each sample to optimized cluster centroids ().

* **Clustering Optimization:** Validated with a Purity score of **0.8408**.
* **Architecture:** Optimized through a grid search of 162 training sessions.
* **Performance:** Matched the baseline MLP at **98.70% Accuracy**.

---

## 📈 Performance Comparison

| Metric | Random Forest | Baseline MLP | K-Means Augmented NN |
| --- | --- | --- | --- |
| **Accuracy** | **99.40%** | 98.70% | 98.70% |
| **F1-Score** | **0.9920** | 0.9865 | 0.9862 |
| **Precision** | 0.9920 | 0.9871 | 0.9862 |
| **Recall** | 0.9920 | 0.9860 | 0.9863 |
| **Training Complexity** | Low | Medium | High |

---

## 🔍 Key Findings & Conclusions

* **Redundancy of Geometric Features:** While the addition of 23 K-Means distance features provided "geometric context," it failed to improve performance beyond the baseline MLP. The neural network likely captures these spatial relationships internally, rendering the manual addition of distance features redundant.
* **The Power of Ensembles:** Random Forest remains the superior solution for this specific task, offering maximum accuracy with significantly lower computational overhead and training time.
* **Feature Overlap:** Class 1 and Class 3 share highly similar statistical signatures, representing the primary area where all models faced slight misclassification.

---

## 👤 Author

**Nicola Balzano** * **Email:** [n.balzano2@studenti.uniba.it](mailto:n.balzano2@studenti.uniba.it)

* **Degree:** Computer Science - Security Engineering
* **Institution:** Università degli Studi di Bari "Aldo Moro"

---
