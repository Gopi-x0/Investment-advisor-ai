# ML Label Design for Investment Advisor AI

## Purpose

This document defines **how labels are created** for the machine learning version of our investment advisor. Labels represent the *correct action* (BUY / HOLD / SELL) based on **future price movement**, and are used to train a supervised ML model.

---

## Label Definitions

We use three classes:

* **BUY = 1**
* **HOLD = 0**
* **SELL = -1**

These numeric labels are compatible with common ML models (e.g., Logistic Regression).

---

## Future Window (Prediction Horizon)

* **Days ahead:** `5` trading days
* For each day *t*, we compare today’s close price with the close price at *t + 5*.

**Reasoning:**

* Short enough to reduce long-term uncertainty
* Long enough to avoid daily noise
* Beginner-friendly and commonly used in finance ML

---

## Future Return Calculation

For each day *t*:

```
future_return = (price[t + 5] - price[t]) / price[t]
```

This gives a percentage return over the next 5 days.

---

## Thresholds for Labeling

We apply thresholds to avoid noisy signals:

* **BUY (1):** `future_return > +2%`
* **SELL (-1):** `future_return < -2%`
* **HOLD (0):** otherwise

**Why thresholds matter:**

* Prevents reacting to tiny price changes
* Produces more stable and realistic signals
* Reduces overfitting

---

## Example

| Today Price | Price in 5 Days | Return | Label |
| ----------- | --------------- | ------ | ----- |
| 100         | 103             | +3%    | BUY   |
| 100         | 99              | -1%    | HOLD  |
| 100         | 96              | -4%    | SELL  |

---

## Features Used (Inputs to ML Model)

Each training row will include indicators computed **only using past data**:

* Short Moving Average (e.g., 20-day)
* Long Moving Average (e.g., 50-day)
* Momentum (e.g., 10-day)
* Trend Strength
* Volatility

> ⚠️ No future data is allowed in features (to avoid data leakage).

---

## Important Constraints

* Labels depend on **future prices**, but features must use **only historical data**
* Rows near the end of the dataset (last 5 days) will not have labels and should be dropped
* This labeling strategy is an assumption, not financial advice

---

## Limitations

* Market behavior changes over time
* Labels are not guaranteed to represent optimal trades
* This model is for **educational purposes only**

---

## Next Steps

1. Implement label creation code (`create_labels` function)
2. Build a training dataset combining features + labels
3. Train a simple ML model using scikit-learn
4. Compare ML signals vs rule-based advisor
