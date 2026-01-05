# CreditGuard-ML: End-to-End Fintech Scoring Engine

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**CreditGuard-ML** is a production-grade credit risk analytics engine developed to automate lending decisions. Built from scratch in a mobile-constrained environment (Termux), this project demonstrates the ability to implement complex financial mathematics without relying on high-level black-box libraries like Scikit-Learn.

---

## 🚀 Executive Summary

The engine processes a dataset of over **148,000 loan records** to predict default probability. By implementing a custom **Logistic Regression** kernel via **Gradient Descent**, the system identifies high-risk profiles and calibrates them into a standard **300–850 credit scorecard**.

### Business Impact
- **Auto-Approval Rate:** 58.5% (Low-risk applicants processed instantly)
- **Operational Efficiency:** 29.5% flagged for manual review, reducing human overhead
- **Risk Mitigation:** 12% auto-rejection based on high-volatility DTI metrics

---

## 🧠 Mathematical Architecture

### 1. Optimization via Gradient Descent

Instead of static rules, the model learns the relationship between variables by minimizing the **Binary Cross-Entropy Loss** function \( J(\theta) \):

\[
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m}
\left[
y^{(i)} \log(h_\theta(x^{(i)})) +
(1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))
\right]
\]

Model weights are updated iteratively using:

\[
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)
\]

---

### 2. Feature Engineering: Debt-to-Income (DTI)

The **Debt-to-Income (DTI)** ratio was engineered as the primary predictor.  
DTI acts as a risk amplifier—once it exceeds **40%**, the log-odds of default increase exponentially. This behavior is reflected in the model’s high learned coefficient (\(\beta = 1.17\)).

---

## 📂 Project Structure

| File | Description |
|-----:|-------------|
| `train_model.py` | Core ML kernel implementing Gradient Descent optimization |
| `decision_engine.py` | Calibration logic mapping logits to the 300–850 score range |
| `app.py` | Interactive CLI for real-time loan risk assessment |
| `clean_data.py` | Data pipeline: scaling and outlier removal for 148k+ rows |
| `credit_score_audit.png` | Validation plot showing score separation (Goods vs Bads) |

---

## 🛠️ Installation & Usage

### 1. Clone & Setup
```bash
git clone https://github.com/Wilfredumba/CreditGuard-ML-Fintech.git
cd CreditGuard-ML-Fintech
pip install numpy pandas matplotlib
