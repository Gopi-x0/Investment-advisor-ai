# Machine Learning Plan — Investment Advisor AI

## Goal

The goal of this plan is to integrate **machine learning** into the existing rule-based investment advisor in a **beginner-friendly, modular, and safe** way.

The ML system will **not replace the entire project**, but will replace the *decision-making logic* currently handled by `advisor.py`.

---

## Current System (Baseline)

Current pipeline:

```
Market Data → Indicators → Rule-Based Advisor → BUY / SELL / HOLD
```

This baseline is important because:

* It provides a working reference
* It allows comparison between rule-based and ML-based decisions
* It helps validate whether ML is actually improving results

---

## ML-Enhanced System (Target)

Planned ML pipeline:

```
Market Data → Indicators → ML Model → BUY / SELL / HOLD
```

The following components will remain unchanged:

* `data_loader.py`
* `indicators.py`
* `main.py` (minimal changes only)

The ML logic will live in **new files**, keeping the system modular.

---

## Features (Model Inputs)

The ML model will use numerical indicators computed from historical data:

* Short Moving Average (e.g., 20-day)
* Long Moving Average (e.g., 50-day)
* Momentum (e.g., 10-day)
* Trend Strength
* Volatility

These features are chosen because:

* They are already implemented or planned
* They are common in financial analysis
* They are numerical and suitable for ML models

---

## Labels (Model Outputs)

The model will predict one of three actions:

* **BUY = 1**
* **HOLD = 0**
* **SELL = -1**

Labels are generated using **future price movement**, as defined in `ml_labels.md`.

---

## Machine Learning Approach

### Model Type

Initial model:

* **Logistic Regression (scikit-learn)**

Reasons:

* Simple and interpretable
* Works well with tabular data
* Easy to train and debug
* Industry-relevant for finance

Later experimentation may include:

* Random Forest
* Gradient Boosting

---

## Training Strategy

* Training will be done **offline** using historical data
* A separate script (`train_model.py`) will be used
* The trained model will be saved to disk (`model.pkl`)

Training steps:

1. Load historical data
2. Compute indicators
3. Generate labels
4. Split data into train/test sets
5. Train ML model
6. Evaluate basic performance
7. Save trained model

---

## Inference Strategy (Using the Model)

During app execution:

* Load the saved model
* Compute indicators from recent data
* Pass indicators to the model
* Receive BUY / SELL / HOLD prediction

This logic will live in:

```
ml_model.py
```

---

## Evaluation Plan

The ML model will be evaluated by:

* Comparing predictions to true labels on test data
* Comparing ML signals to rule-based signals
* (Later) simple backtesting

The goal is **learning and comparison**, not profit guarantees.

---

## Constraints & Ethics

* This project is for **educational purposes only**
* Predictions are not financial advice
* No real money should be traded using this system

---

## Next Implementation Steps

1. Implement label creation code
2. Build training dataset (features + labels)
3. Write training script (`train_model.py`)
4. Integrate ML inference into the app
5. Compare ML vs rule-based performance
