# Logistic Regression

## Why Not Linear Regression for Classification?

Problems with using linear regression for binary labels:

- **Outlier sensitivity:** a single extreme outlier can rotate the best-fit line and change predictions for many points.
- **Invalid output range:** linear regression can predict values outside $[0,1]$, which are meaningless as probabilities.

Example: an obesity threshold at 75kg can be broken by a 150kg outlier shifting the line.

## The Sigmoid Function

Logistic regression wraps the linear equation inside a sigmoid that squashes values into $(0,1)$:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Where $x$ is the linear equation $\theta^T x + b$.

How it behaves:

- Very large positive $x$ → $\sigma(x) \approx 1$.
- Very large negative $x$ → $\sigma(x) \approx 0$.

Full pipeline: Raw features → linear equation → sigmoid → probability → threshold → class label.

## Multiclass Classification — One-vs-Rest (OvR)

Logistic regression is inherently binary. For $N$ classes, train $N$ binary classifiers, each one distinguishing class $C_i$ vs the rest. At prediction time, pick the class with the highest probability.

Example (3 classes):

- Model1 (C1 vs Rest): 0.20
- Model2 (C2 vs Rest): 0.25
- Model3 (C3 vs Rest): 0.55 ← pick C3

In scikit-learn:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(multi_class='ovr')
model.fit(X_train, y_train)
```

## Notes

- Threshold selection is domain-specific. Default 0.5 is common, but healthcare and safety-critical domains often use lower thresholds to increase recall.
- Regularization, feature scaling, and proper class weighting are important for real-world performance.

---

See also: `metrics.md` for evaluation metrics and decision guidance.
