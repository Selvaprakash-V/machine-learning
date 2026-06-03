# Classification Metrics

## 1. The Classification Problem
Classification predicts a discrete label for an input. Two approaches:

- **Hard prediction** → directly outputs a class ("Spam" / "Not Spam").
- **Soft prediction** → outputs a probability (0 to 1), then you decide the label using a threshold.

### Threshold Logic

| Threshold | When to use |
|---:|---|
| 0.5 (default) | General-purpose, balanced data |
| 0.3–0.4 | High-stakes domains (prefer catching all positives) |
| 0.6–0.7 | When false alarms are costly |

Rule: Probability > threshold → Positive class. Probability ≤ threshold → Negative class.

## Class Distribution — The First Check You Must Always Do

| Distribution | Recommended Metric |
|---:|---|
| 50:50, 60:40, 70:30 | Accuracy is fine |
| 80:20, 90:10 | Accuracy is misleading — use Precision/Recall/F1 |

The lazy model trap: on a 90:10 imbalanced dataset, a model that always predicts the majority class gets 90% accuracy but catches zero positive cases. Never trust accuracy alone on imbalanced data.

## 2. Confusion Matrix — The Foundation of Everything

                  Actual: 1        Actual: 0
Predicted: 1    True Positive (TP)   False Positive (FP)
Predicted: 0    False Negative (FN)  True Negative (TN)

Error types:

- **False Positive (FP)** — also called Type I Error: predicted positive, actually negative.
- **False Negative (FN)** — also called Type II Error: predicted negative, actually positive.

### Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + FP + FN + TN}$$

When it lies: 900 healthy + 100 sick → model predicts everyone healthy → 90% accuracy, TP = 0. Completely useless model.

## 3. Precision & Recall — The Real Metrics

Both metrics focus on the positive class.

### Recall (Sensitivity / True Positive Rate)

$$\text{Recall} = \frac{TP}{TP + FN}$$

Question it answers: "Of all actual positives, how many did I catch?"

Use when missing a positive is catastrophic (e.g., cancer detection, fraud).

### Precision (Positive Predictive Value)

$$\text{Precision} = \frac{TP}{TP + FP}$$

Question it answers: "Of everything I called positive, how many actually were?"

Use when false alarms are costly (e.g., spam filtering).

### The Precision–Recall Trade-off

- Lower the threshold → catch more positives → Recall ↑, Precision ↓.
- Raise the threshold → only confident positives → Precision ↑, Recall ↓.

Use-case examples:

- Cancer detection → **Recall**
- Spam filter → **Precision**

## 4. F-Beta Score — Balancing Precision & Recall

Harmonic mean of Precision and Recall:

$$F_{\beta} = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}$$

Why harmonic mean? It penalizes extreme imbalances: if Precision = 1.0 and Recall = 0.0, the harmonic mean is 0.

The Beta dial:

- $\beta = 1$ (F1): equal weight to Precision and Recall.
- $\beta < 1$ (e.g., 0.5): more weight on Precision.
- $\beta > 1$ (e.g., 2): more weight on Recall.

## 5. Quick Decision Framework — Which Metric to Use?

START

- Is your data balanced (≤ 70:30)? → **YES**: Accuracy is a reasonable starting metric.
- Is your data imbalanced (> 70:30)? → **YES**: Never use Accuracy alone.
  - Is missing a positive very costly? → **Maximize RECALL**, consider F2.
  - Is a false alarm very costly? → **Maximize PRECISION**, consider F0.5.
  - Both equally bad? → Use **F1**.

## 6. Key Formulas — Quick Reference

- **Accuracy:** $\frac{TP + TN}{TP + FP + FN + TN}$
- **Recall:** $\frac{TP}{TP + FN}$
- **Precision:** $\frac{TP}{TP + FP}$
- **F1 Score:** $2 \cdot \frac{P \cdot R}{P + R}$
- **F-Beta:** see above
- **Sigmoid:** $\sigma(x) = \frac{1}{1 + e^{-x}}$

## 7. Common Exam / Interview Traps

- "90% accuracy means good model" → Not if data is 90:10 — check the confusion matrix.
- "Use linear regression for binary output" → Output range is unbounded; use logistic regression.
- "Precision and Recall can both be maximized" → They trade off; lowering threshold raises Recall, drops Precision.
- "F1 always the best F-score" → Only when FP and FN are equally costly; adjust $\beta$ for real costs.
- "Logistic regression can only do binary" → Use OvR strategy for multiclass.
- "Threshold is always 0.5" → Domain-dependent — healthcare often uses 0.3–0.4.

---

See also: `logistic-regression.md` for logistic regression theory and multiclass strategies.
