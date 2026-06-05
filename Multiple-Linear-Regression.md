# Machine Learning — Complete Master Notes
### Topics: Multicollinearity · Bias & Variance · R² & Adjusted R² · Multiple Linear Regression

---

# PART 1: Multicollinearity in Linear Regression

---

## 1. What is Multicollinearity?

Multicollinearity occurs when **two or more independent features** in a dataset are **highly correlated** with each other.

> Before diagnosing multicollinearity, you must understand **correlation** and **covariance** — how features change in relation to one another.

### Example Setup

| Feature | Role |
|---|---|
| $X_1$ → Age | Independent feature |
| $X_2$ → Years of Experience | Independent feature |
| $Y$ → Salary | Output / Dependent feature |

The multiple linear regression equation:

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2$$

| Symbol | Meaning |
|---|---|
| $\beta_0$ | Intercept |
| $\beta_1$ | Coefficient (slope) for Age |
| $\beta_2$ | Coefficient (slope) for Experience |

---

## 2. Why Multicollinearity is a Problem

In an ideal regression setup:
- $\beta_1$ = change in Y for every 1-unit change in $X_1$, **keeping all other features constant**

But when $X_1$ (Age) and $X_2$ (Experience) are highly correlated:

```
Change X₁ (Age) ──────► X₂ (Experience) ALSO changes
                                │
                                ▼
              You CANNOT hold X₂ constant anymore
                                │
                                ▼
         β₁ and β₂ become HIGHLY UNSTABLE estimates
                                │
                                ▼
        Impossible to determine independent effect
        of each feature on the target Y
```

### Visual — Correlated vs Independent Features

```
IDEAL (No Multicollinearity)      PROBLEM (Multicollinearity)

X₁ ──────────────► Y              X₁ ──┐
                                        ├──► Entangled ──► Y
X₂ ──────────────► Y              X₂ ──┘   (unstable β)

Each feature has clear,           Features overlap — coefficients
independent contribution          are unreliable
```

---

## 3. How to Detect Multicollinearity

### Method 1 — p-values in Statistical Summary

```python
import statsmodels.api as sm

X_with_const = sm.add_constant(X)
model = sm.OLS(y, X_with_const).fit()
print(model.summary())
```

| p-value | Interpretation |
|---|---|
| p < 0.05 | Feature is statistically significant ✓ |
| p > 0.05 | Feature is NOT significant — likely multicollinear ✗ |

### Method 2 — Variance Inflation Factor (VIF)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)
```

| VIF Score | Interpretation |
|---|---|
| VIF = 1 | No correlation |
| VIF 1–5 | Moderate correlation (acceptable) |
| VIF > 5 | High multicollinearity — action needed |
| VIF > 10 | Severe multicollinearity |

### Method 3 — Correlation Heatmap

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()
```

```
Correlation Matrix Visual:

         Age   Exp   Salary
Age    [ 1.0  0.98   0.95 ]  ← Age & Exp = 0.98 → PROBLEM
Exp    [ 0.98  1.0   0.97 ]
Salary [ 0.95  0.97  1.0  ]
```

---

## 4. How to Resolve Multicollinearity

### Resolution 1 — Drop Redundant Features

```
Age shares 98% variance with Experience
         │
         ▼
Age is REDUNDANT — drop it
         │
         ▼
Train model using Experience only
         │
         ▼
No loss in predictive power ✓
```

### Resolution 2 — Interpret Negative Coefficients

```
Feature: Newspaper Advertising Spend
Coefficient: -0.87  ← NEGATIVE

Meaning: This feature hurts the model.
Action:  Drop it — it is non-contributing or redundant.
```

### Resolution 3 — Principal Component Analysis (PCA)

```
Original correlated features          PCA Transformed features
X₁ = Age          ─────────►         PC₁ (captures max variance)
X₂ = Experience   ─────────►         PC₂ (captures remaining)

Result: Features are now UNCORRELATED
        Multicollinearity resolved ✓
```

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
```

---

## 5. Multicollinearity — Quick Decision Flow

```
START: Multiple features in your model
              │
              ▼
   Check Correlation Matrix
              │
     ┌────────┴────────┐
     ▼                 ▼
 r > 0.9           r < 0.9
(High corr)     (No problem)
     │                 │
     ▼                 ▼
Check VIF          Proceed with
     │             model training ✓
  VIF > 5?
     │
  ┌──┴──────────────────┐
  ▼                     ▼
Drop one          Apply PCA for
redundant         dimensionality
feature           reduction
```

---

---

# PART 2: Bias and Variance — In-Depth Intuition

---

## 1. Core Definitions

**Bias**
- Error from approximating a complex real-world problem with a **too-simple model**
- High bias = model ignores patterns in data = **underfitting**

**Variance**
- Model's sensitivity to **fluctuations** in training data
- High variance = model memorizes noise = **overfitting**

---

## 2. The Three Model States

```
UNDERFITTING              IDEAL MODEL              OVERFITTING
(High Bias, Low Var)   (Low Bias, Low Var)    (Low Bias, High Var)

  Y│                       Y│    /                Y│  .
   │  ────────              │   /                  │ /\/\/\
   │  (flat line,           │  /  (clean fit)      │/  (wiggly,
   │   too simple)          │ /                    │   memorized noise)
   └──────── X              └────── X              └──────── X

Train Error: HIGH          Train: LOW               Train: VERY LOW
Test Error:  HIGH          Test:  LOW               Test:  HIGH
```

### Scenario Table

| Scenario | Model State | Train Error | Test Error | Characteristics |
|---|---|---|---|---|
| High Bias, Low Variance | **Underfitting** | High | High | Too simple, misses all patterns |
| Low Bias, High Variance | **Overfitting** | Low | High | Memorized noise, fails on new data |
| Low Bias, Low Variance | **Ideal Model** | Low | Low | Generalizes perfectly ✓ |

---

## 3. Intuition — Why Does This Happen?

### Underfitting
```
Real data pattern:  ∿∿∿∿∿∿ (curved/complex)
Model output:       ─────── (straight line, degree=1)

Model is TOO SIMPLE to capture the real relationship.
Increasing model complexity → fixes underfitting.
```

### Overfitting
```
Training data:    . . . . . . . (some noise points)
Model output:     /\/\/\/\/\/\/ (connects every single dot)

Model is TOO COMPLEX — captures noise, not pattern.
Reducing complexity or adding regularization → fixes overfitting.
```

---

## 4. Algorithmic Context

### Decision Tree vs Random Forest

```
DECISION TREE (Single)        RANDOM FOREST (Ensemble)
──────────────────────        ──────────────────────────
Low Bias                      Low Bias
HIGH Variance  ←problem       LOW Variance  ← solved!
Prone to overfitting          Generalizes well

How Random Forest fixes it:
80 records split into 4 batches of 20
     │
     ├── M₁ trained on 20 records
     ├── M₂ trained on 20 records
     ├── M₃ trained on 20 records
     └── M₄ trained on 20 records
               │
               ▼
     Aggregate predictions
    (average / majority vote)
               │
               ▼
    High variance trees → Low variance ensemble ✓
```

---

## 5. The Bias-Variance Tradeoff Visual

```
Error
│\
│ \   ← Total Error
│  \        /‾‾‾‾‾‾
│   \      / ← Variance
│    \    /
│     \  /
│      \/  ← Sweet spot (ideal model)
│      /\
│     /  \── Bias
│    /
└──────────────────── Model Complexity
   Simple          Complex
   (underfit)      (overfit)
```

| Zone | Bias | Variance | Action |
|---|---|---|---|
| Left of sweet spot | High | Low | Increase complexity |
| Right of sweet spot | Low | High | Regularize / simplify |
| Sweet spot | Low | Low | Deploy model ✓ |

---

---

# PART 3: R² and Adjusted R² — Clearly Explained

---

## 1. R² — Coefficient of Determination

### Purpose
Measures what **proportion of variance** in Y is explained by your model compared to a simple baseline (predicting the mean every time).

### Formula

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

| Term | Formula | Meaning |
|---|---|---|
| $SS_{res}$ | $\sum(y_i - \hat{y}_i)^2$ | Sum of squared errors — actual vs predicted |
| $SS_{tot}$ | $\sum(y_i - \bar{y})^2$ | Total variance — actual vs mean |

### Visual — What R² Measures

```
YOUR MODEL                    BASELINE (predict mean always)
                                      ȳ
Y│  .  *                      Y│─────────────────
 │ . * .  *                    │  .   .   .   .
 │. *  .  ← model line         │         ← flat mean line
 └──────── X                   └──────── X

SS_res = small gaps           SS_tot = large gaps (from mean)

R² = 1 - (small/large) = close to 1  ← great model
R² = 1 - (large/large) = close to 0  ← no better than guessing mean
```

### Interpreting R²

| R² Value | Meaning |
|---|---|
| 1.0 | Perfect fit — model explains 100% of variance |
| 0.85 | Model explains 85% of the variation in Y |
| 0.5 | Model explains 50% — mediocre |
| 0.0 | Model is no better than predicting the mean |
| < 0 | Model is worse than the mean (very bad) |

---

## 2. The Fatal Flaw of R²

> **R² never decreases when you add more features — even useless ones.**

```
Model 1: Salary ~ Age + Experience
R² = 0.91

Model 2: Salary ~ Age + Experience + Toilet Paper Consumption
R² = 0.91 or 0.912  ← same or slightly higher!

R² is LYING — Toilet Paper Consumption has zero real value
but R² doesn't penalize you for adding it.
```

---

## 3. Adjusted R² — The Fix

### Purpose
Penalizes the model for adding features that **do not contribute** to predictive accuracy.

### Formula

$$\text{Adjusted } R^2 = 1 - \left[\frac{(1 - R^2)(n - 1)}{n - p - 1}\right]$$

| Symbol | Meaning |
|---|---|
| $n$ | Total number of data points |
| $p$ | Number of independent features/predictors |
| $R^2$ | Standard R-squared score |

### How It Behaves

```
Adding a RELEVANT feature:          Adding an IRRELEVANT feature:

R² increases significantly          R² barely changes
p increases by 1                    p increases by 1
                                    denominator (n-p-1) SHRINKS
                                    penalty fraction GROWS

→ Adjusted R² INCREASES ✓          → Adjusted R² DECREASES ✗
  (feature is validated)              (feature is rejected)
```

### Visual — R² vs Adjusted R²

```
         R²        Adjusted R²
Feature 1 added:  0.70  →  0.70   (both go up — feature is relevant)
Feature 2 added:  0.85  →  0.84   (both go up — feature is relevant)
Feature 3 added:  0.86  →  0.82   (R² up, Adj R² DOWN — irrelevant feature!)
                   ↑               ↑
                 Lies!           Catches it ✓
```

---

## 4. R² vs Adjusted R² — Comparison

| Property | R² | Adjusted R² |
|---|---|---|
| Range | 0 to 1 | Can be negative |
| Penalizes irrelevant features | No ✗ | Yes ✓ |
| Used for simple regression | ✓ | Not necessary |
| Used for multiple regression | Unreliable alone | Always use this ✓ |
| Behavior when adding noise feature | Stays same or increases | Decreases |

---

---

# PART 4: Multiple Linear Regression — Theory to Implementation

---

## 1. Definition

Multiple Linear Regression is a **supervised learning algorithm** that models the linear relationship between **one continuous output (Y)** and **multiple input features (X₁, X₂, ..., Xₙ)**.

### Mathematical Equation

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \cdots + \beta_n X_n + \epsilon$$

| Symbol | Meaning |
|---|---|
| $Y$ | Predicted output (dependent variable) |
| $\beta_0$ | Intercept — value of Y when all inputs = 0 |
| $\beta_1 \dots \beta_n$ | Coefficients/slopes for each feature |
| $X_1 \dots X_n$ | Independent input features |
| $\epsilon$ | Residual error |

---

## 2. Geometric Intuition

```
Simple Linear Regression         Multiple Linear Regression
(1 feature → 2D line)           (2 features → 3D plane)

Y │    /                              Y
  │   /                              │   /‾‾‾/
  │  /  ← best-fit LINE              │  /   /  ← best-fit PLANE
  │ /                                │ /___/
  └──── X₁                          └──────── X₁
                                    /
                                  X₂

3+ features → Hyperplane (n-dimensional, same math)
```

---

## 3. Optimization — How Coefficients Are Found

```
OBJECTIVE: Minimize cost function (Sum of Squared Residuals)

Method 1: Ordinary Least Squares (OLS)
──────────────────────────────────────
Solves β directly using matrix algebra:
β = (XᵀX)⁻¹Xᵀy
Fast for small datasets. Exact solution.

Method 2: Gradient Descent
──────────────────────────
Iteratively updates each β simultaneously:
β₁ = β₁ - α·(dJ/dβ₁)
β₂ = β₂ - α·(dJ/dβ₂)
...
Preferred for large datasets.
```

---

## 4. Full Implementation Pipeline

```python
# ─────────────────────────────────────────────
# IMPORT LIBRARIES
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             r2_score)

# ─────────────────────────────────────────────
# STEP 1: Load Dataset
# ─────────────────────────────────────────────
data = {
    'Age':        [25, 27, 30, 35, 40, 45, 50, 55],
    'Experience': [1,  3,  5,  9,  13, 17, 21, 24],
    'Salary':     [45000, 50000, 62000, 78000,
                   95000, 110000, 132000, 145000]
}
df = pd.DataFrame(data)

# Separate features and target
X = df[['Age', 'Experience']]   # Independent features
y = df['Salary']                # Dependent target

# ─────────────────────────────────────────────
# STEP 2: Train-Test Split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 80% training, 20% testing

# ─────────────────────────────────────────────
# STEP 3: Feature Scaling (Standardization)
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform on train
X_test_scaled  = scaler.transform(X_test)       # transform only (no leakage!)

# ─────────────────────────────────────────────
# STEP 4: Train the Model (OLS via Scikit-Learn)
# ─────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print(f"Intercept  (β₀): {model.intercept_:.2f}")
print(f"Coefficients (β₁, β₂): {model.coef_}")

# ─────────────────────────────────────────────
# STEP 5: Predict and Evaluate
# ─────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")
```

---

## 5. Understanding the Pipeline Step by Step

```
RAW DATA (Age, Experience, Salary)
          │
          ▼
   Split → X_train / X_test / y_train / y_test
          │
          ▼
   StandardScaler on X_train (fit + transform)
   StandardScaler on X_test  (transform only)
          │
          ▼
   LinearRegression().fit(X_train_scaled, y_train)
   Finds β₀, β₁, β₂ that minimize Σ(y - ŷ)²
          │
          ▼
   model.predict(X_test_scaled) → y_pred
          │
          ▼
   Compare y_pred vs y_test using MAE, MSE, RMSE, R²
```

---

## 6. Why Feature Scaling Matters

```
BEFORE Scaling:
Age        → range [25, 55]    (small numbers)
Experience → range [1, 24]     (small numbers)
Salary     → range [45k, 145k] (large numbers)

Gradient descent steps are UNEVEN across features.
Some features dominate others. Slow convergence.

AFTER StandardScaler:
Age        → mean=0, std=1
Experience → mean=0, std=1

All features on same scale.
Gradient descent converges faster and more stably ✓
```

> **Critical rule:** Fit the scaler ONLY on training data. Apply (transform) it to test data. Never fit on test data — that's **data leakage**.

---

## 7. Interpreting Coefficients

```
model.coef_ = [β₁, β₂]

Example output: [1200.5,  3500.2]
                  ↑         ↑
               Age coef  Experience coef

β₁ = 1200.5 → For every 1-unit increase in Age (scaled),
               Salary increases by ₹1200.5,
               holding Experience constant.

β₂ = 3500.2 → For every 1-unit increase in Experience (scaled),
               Salary increases by ₹3500.2,
               holding Age constant.

NEGATIVE coefficient example:
β = -850 → As that feature increases, Y DECREASES.
           Consider dropping this feature if it's non-contributing.
```

---

## 8. Evaluation Metrics — Full Reference

| Metric | Formula | Interpretation |
|---|---|---|
| **MAE** | $\frac{1}{n}\sum\|y - \hat{y}\|$ | Average error in same units as Y. Robust to outliers. |
| **MSE** | $\frac{1}{n}\sum(y - \hat{y})^2$ | Penalizes large errors heavily. Used in cost function. |
| **RMSE** | $\sqrt{MSE}$ | Same units as Y. Most interpretable error metric. |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | % variance explained. Higher = better (max 1.0). |
| **Adj R²** | $1 - \frac{(1-R^2)(n-1)}{n-p-1}$ | Penalizes irrelevant features. Use for multiple regression. |

---

## 9. Full Diagnostic Checklist Before Deploying

```
✅ 1. Check for multicollinearity (VIF or correlation matrix)
✅ 2. Scale features before training
✅ 3. Split data before scaling (prevent data leakage)
✅ 4. Check coefficients — negative ones may signal issues
✅ 5. Use Adjusted R² (not just R²) to validate features
✅ 6. Analyze residuals — should be randomly distributed
✅ 7. Check for outliers — can distort coefficient estimates
✅ 8. Verify linear relationship assumption holds
```

---

## 10. Common Mistakes & Interview Traps

| Mistake | Correct Understanding |
|---|---|
| Fitting scaler on test data | Always fit only on train, transform test |
| Using R² alone for feature validation | Use **Adjusted R²** for multiple regression |
| Ignoring multicollinearity | Always check VIF before finalizing features |
| Adding more features to improve R² | Irrelevant features inflate R² falsely |
| Ignoring negative coefficients | They signal inverse or redundant relationships |
| Not scaling features | Gradient descent becomes slow and unstable |
| p > 0.05 means keep the feature | p > 0.05 means the feature is NOT significant — consider dropping |

---

# Master Summary — All Topics at a Glance

```
MULTIPLE LINEAR REGRESSION WORKFLOW
═════════════════════════════════════

Raw Data
   │
   ▼
EDA + Correlation Matrix
   │
   ├── Multicollinearity found?
   │         │
   │      YES │ → Drop redundant feature or apply PCA
   │         │
   ▼         ▼
Feature Scaling (StandardScaler)
   │
   ▼
Train/Test Split
   │
   ▼
LinearRegression().fit()
   │
   ├── Check coefficients (positive/negative/near-zero)
   │
   ▼
Predict on Test Set
   │
   ├── MAE / RMSE → How far off are predictions?
   ├── R²         → How much variance explained?
   └── Adj R²     → Are all features actually contributing?
   │
   ▼
Bias-Variance Check
   │
   ├── Train error HIGH, Test error HIGH → Underfitting → More complexity
   ├── Train error LOW,  Test error HIGH → Overfitting  → Regularize
   └── Both LOW                         → Ideal model ✓
   │
   ▼
DEPLOY MODEL ✓
```

---

> **Core philosophy:** A good regression model is not the one that fits training data perfectly. It's the one that **generalizes** — low error, meaningful coefficients, validated features, and consistent performance on unseen data.