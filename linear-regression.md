# Machine Learning — Complete Master Notes
### Topics: Linear Regression · Ridge Regression · Lasso Regression
 
---
 
# PART 1: Linear Regression
 
---
 
## 1. What is Linear Regression?
 
Linear Regression is a **supervised learning** algorithm that solves **regression problems** — where the output is a continuous numerical value (not a class label).
 
**Core idea:** Find the single best straight line that passes through a scatter of data points such that the total error between predicted and actual values is as small as possible.
 
### Real-World Examples
 
| Input (X) | Output (Y) |
|---|---|
| House size (sq ft) | House price |
| Years of experience | Salary |
| Hours studied | Exam score |
| Temperature | Ice cream sales |
 
---
 
## 2. The Linear Equation: y = mx + c
 
Every possible straight line on a 2D graph is fully described by just two numbers — **m** and **c**.
 
$$\hat{y} = mx + c$$
 
### Components Explained
 
**m — The Slope**
- Measures steepness: *how much does Y change when X changes by 1 unit?*
- If m = 5000 in a house price model → every extra sq ft adds ₹5000 to predicted price
- Positive m → line goes up left to right
- Negative m → line goes down left to right
**c — The Intercept**
- The value of Y when X = 0
- Where the line crosses the Y-axis
- In the house price example: the base price even before size is considered
**y vs ŷ (y-hat)**
- **y** = actual real data point (what really happened)
- **ŷ** = predicted point sitting exactly on the best-fit line
- The gap between y and ŷ is the **error** (also called residual)
### Visual — What the Line Looks Like
 
```
Price (Y)
  │                              * ← actual point (y)
  │                           /  ↕ ← error (y - ŷ)
  │                        /     • ← predicted point (ŷ)
  │                     /
  │                  /
  │               /
  │            /
  │─────────/──────────────────── Size (X)
  c (intercept)
```
 
> **The algorithm's entire job:** minimize all those vertical gaps simultaneously.
 
---
 
## 3. The Cost Function J — Measuring How Wrong You Are
 
Since infinite lines can be drawn through a dataset, you need a **score** to compare them. That score is the **Cost Function J**.
 
$$J = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$
 
**Where:**
- `m` = total number of data points
- $(\hat{y} - y)^2$ = squared error for each point
- The sum adds up all individual errors
- The $\frac{1}{2m}$ averages and simplifies calculus later
### Why Square the Errors?
- Raw errors can be positive or negative → they'd cancel out
- Squaring makes all errors **positive**
- Squaring also **penalizes large errors more heavily** than small ones
### Visual — What J Looks Like Across Different Lines
 
```
     J (Cost)
     │\
     │  \
     │    \         ← too steep (high error)
     │      \
     │        \____/‾‾‾‾
     │             ↑
     │          Global Minima ← best line, minimum error
     │
     └──────────────────────────── m (slope value)
```
 
- **J = 0** → Perfect fit, line passes through every single point (rare in practice)
- **J = large** → Poor fit, predictions are far from reality
- **Goal: slide down this curve to the bottom**
---
 
## 4. Gradient Descent — Finding the Best Line Efficiently
 
You can't just "try every possible line" — there are infinitely many. Gradient Descent is the systematic process of **walking downhill on the cost curve** to find the minimum.
 
### The Convergence Update Rule
 
$$m = m - \alpha \cdot \frac{dJ}{dm}$$
 
$$c = c - \alpha \cdot \frac{dJ}{dc}$$
 
Both m and c are updated **simultaneously** on every iteration.
 
| Symbol | Meaning |
|---|---|
| α (alpha) | Learning rate — size of each step |
| dJ/dm | Derivative — slope of the cost curve at current position |
| The whole term | How much to adjust m in this iteration |
 
---
 
### The Learning Rate α — The Most Critical Hyperparameter
 
```
α too SMALL                α just RIGHT              α too LARGE
 
J│\                        J│\                       J│\
 │  \  . . .               │  \                      │  \        •
 │    \ . .                │    \.                   │    \   ↗
 │      \. ←slow           │      \__.               │      \/
 │       ↓ many steps      │          ↑fast+stable   │      jumps over!
 └─────────────            └──────────────           └──────────────
  Takes forever             Ideal                    Never converges
```
 
- **Too small (e.g., 0.00001):** Converges correctly but takes thousands of iterations — slow training
- **Just right (e.g., 0.001):** Smooth convergence, reaches minima efficiently
- **Too large (e.g., 1.0):** Overshoots the minima, bounces back and forth, may diverge upward
---
 
### How the Derivative Guides the Steps
 
**Case 1 — You are on the RIGHT side of the curve (positive slope)**
```
J │         /← you are here
  │       /
  │     /
  │   /_____ minima
  └──────────── m
 
dJ/dm is POSITIVE
m = m - α × (positive) → m DECREASES → moves LEFT toward minima ✓
```
 
**Case 2 — You are on the LEFT side of the curve (negative slope)**
```
J │ you are here →\
  │                 \
  │                   \
  │                ____\_____ minima
  └──────────── m
 
dJ/dm is NEGATIVE
m = m - α × (negative) → m INCREASES → moves RIGHT toward minima ✓
```
 
**Case 3 — You are AT the minima (zero slope)**
```
dJ/dm = 0
m = m - α × 0 = m  →  no change
Training stops. Optimal m found. ✓
```
 
> The math **automatically moves in the correct direction** regardless of where you start — this is the elegance of gradient descent.
 
---
 
## 5. The Full Training Loop — Step by Step
 
```
STEP 1: Initialize m = 0, c = 0 (or random small values)
          │
STEP 2: Feed all X values → compute ŷ = mx + c
          │
STEP 3: Compute cost J = (1/2m) Σ(ŷ - y)²
          │
STEP 4: Compute derivatives dJ/dm and dJ/dc
          │
STEP 5: Update: m = m - α·(dJ/dm)
                c = c - α·(dJ/dc)
          │
STEP 6: Repeat Steps 2–5 for N iterations (epochs)
          │
STEP 7: Stop when dJ/dm ≈ 0 → Global Minima reached
          │
STEP 8: Final m and c define your best-fit line → use for predictions
```
 
---
 
## 6. Multiple Features — Extending to the Real World
 
Real datasets rarely have just one input. A house price depends on size, number of rooms, location, age, etc.
 
### The Generalized Equation
 
$$\hat{y} = m_1x_1 + m_2x_2 + m_3x_3 + \cdots + m_nx_n + c$$
 
Or in vector notation:
 
$$\hat{y} = \theta^T x + b$$
 
| Feature (x) | Its own slope (m) |
|---|---|
| House size | m₁ |
| Number of rooms | m₂ |
| Distance from city | m₃ |
| Age of building | m₄ |
 
Each feature has its **own slope** that gradient descent optimizes independently and simultaneously.
 
### Visual — From 2D to 3D
 
**1 Feature → 2D line**
```
Y │    /
  │   /
  │  /
  └──── X₁
```
 
**2 Features → 3D plane**
```
     Y
     │   /‾‾‾/
     │  /   /  ← best-fit PLANE (not a line)
     │ /___/
     └──────── X₁
    /
  X₂
```
 
**3+ Features → Hyperplane** (impossible to visualize, but same math applies)
 
- The cost function J still exists in higher dimensions
- Gradient descent still moves all parameters (m₁, m₂, ..., c) toward the global minima simultaneously
- The update rule applies to **each mᵢ separately** on every iteration
---
 
## 7. Assumptions of Linear Regression
 
| Assumption | Meaning |
|---|---|
| **Linearity** | X and Y have a straight-line relationship |
| **Independence** | Data points don't influence each other |
| **Homoscedasticity** | Error spread is roughly equal across all X values |
| **Normality of errors** | Residuals (y − ŷ) follow a normal distribution |
| **No multicollinearity** | Input features are not highly correlated with each other |
 
---
 
## 8. Evaluating a Regression Model — Key Metrics
 
Unlike classification, you can't use accuracy. Use these instead:
 
### Mean Absolute Error (MAE)
$$MAE = \frac{1}{m}\sum|y - \hat{y}|$$
- Average of absolute errors. Easy to interpret. Not sensitive to outliers.
### Mean Squared Error (MSE)
$$MSE = \frac{1}{m}\sum(y - \hat{y})^2$$
- Penalizes large errors more. Same as the cost function.
### Root Mean Squared Error (RMSE)
$$RMSE = \sqrt{MSE}$$
- Same units as Y. Most commonly reported metric.
### R² Score (Coefficient of Determination)
$$R^2 = 1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2}$$
- Ranges from 0 to 1. Measures how much variance in Y your model explains.
- R² = 0.85 → model explains 85% of the variation in Y
- R² = 1.0 → perfect fit. R² = 0 → model is no better than just predicting the mean.
---
 
## 9. Common Pitfalls & Interview Traps
 
| Pitfall | Correct Understanding |
|---|---|
| "Just pick any line" | No — infinite lines exist; cost function ranks them objectively |
| "Large learning rate trains faster" | It may never converge — always tune α carefully |
| "Linear regression works for classification" | No — output is unbounded; use Logistic Regression instead |
| "More features always helps" | Correlated features cause multicollinearity and hurt the model |
| "R² = 1 means a great model" | Could mean overfitting — check on test data too |
| "Cost = 0 is always the goal" | On test data, cost = 0 = overfitting, not good generalization |
 
---
 
## 10. Everything at a Glance
 
```
RAW DATA (X, y)
      │
      ▼
  Initialize m, c
      │
      ▼
  ŷ = mx + c  ──────────────────────────────────────────┐
      │                                                  │
      ▼                                               UPDATE
  J = Σ(ŷ-y)² / 2m  ──→  Compute derivatives           │
      │                         │                        │
      ▼                         ▼                        │
  Is J ≈ 0?    ──NO──►   m = m - α·dJ/dm  ──────────────┘
      │                   c = c - α·dJ/dc
     YES
      │
      ▼
  BEST-FIT LINE FOUND
  Use ŷ = mx + c to predict new values
```
 
---
 
---
 
# PART 2: Ridge & Lasso Regression
 
---
 
## 1. Why Do We Need These Algorithms?
 
Linear Regression finds the best-fit line by minimizing error on training data. But **minimizing training error too aggressively creates a new problem — overfitting.**
 
Ridge and Lasso solve this by adding a **deliberate penalty** that discourages the model from fitting the training data *too* perfectly.
 
> **Regular Linear Regression asks:** *"What line fits training data best?"*
> **Ridge/Lasso ask:** *"What line fits training data well AND stays simple enough to generalize?"*
 
---
 
## 2. The Core Problem — Overfitting vs Underfitting
 
### The Three Model States
 
```
UNDERFITTING          GOOD MODEL           OVERFITTING
(High Bias)           (Balanced)           (High Variance)
 
  Y│                    Y│    /             Y│  .
   │  ────────           │   /               │ /\/\/\  ← memorized
   │                     │  /  ← clean       │/       every point
   └──────── X           └────── X           └──────── X
 
Train acc: 60%         Train: 90%           Train: 99%
Test acc:  60%         Test:  88%           Test:  70%
```
 
### Bias vs Variance Framework
 
| Condition | Bias | Variance | Meaning |
|---|---|---|---|
| **Underfitting** | High | Low | Model is too simple, misses patterns |
| **Overfitting** | Low | High | Model memorized training noise |
| **Ideal (Generalized)** | Low | Low | Performs consistently on both datasets |
 
### Why Does Overfitting Happen?
 
```
Simple model              Overfit model
(1 feature, clean line)   (many features, wiggly line)
 
Y│   /                    Y│  .
 │  /                      │.  . .
 │ /   ← misses some        │ \/\/\/ ← hits every point
 │/      points but         │        but useless on
 └────── generalizes        └──────── new data
```
 
- Too many features relative to data points
- Polynomial terms that chase every noise point
- Coefficients (slopes) become **extremely large** to fit every training point
> **Root cause of overfitting: very large slope values (m).** When slopes are huge, tiny changes in X produce enormous swings in ŷ — the model is hypersensitive to training data.
 
---
 
## 3. The Cost Function — How Regularization Works
 
### Standard Linear Regression Cost Function
 
$$J = \frac{1}{2m}\sum_{i=1}^{m}(\hat{y}_i - y_i)^2$$
 
**Problem:** Minimizing this alone allows slopes to grow as large as needed to fit training points.
 
### The Regularization Idea
 
Add a **penalty term** directly into the cost function that grows larger when slopes grow larger:
 
$$J_{regularized} = \underbrace{\frac{1}{2m}\sum(\hat{y} - y)^2}_{\text{original error}} + \underbrace{\text{penalty term}}_{\text{controls slope size}}$$
 
Now the model must balance two competing goals simultaneously:
- **Minimize prediction error** (fit the data well)
- **Minimize the penalty** (keep slopes small)
This tug-of-war prevents any single slope from exploding in size.
 
---
 
## 4. Ridge Regression — L2 Regularization
 
### The Cost Function
 
$$J_{Ridge} = \frac{1}{2m}\sum(\hat{y} - y)^2 + \lambda\sum m_j^2$$
 
The penalty term is **lambda × sum of squared slopes.**
 
### Breaking It Down
 
| Component | Role |
|---|---|
| $\frac{1}{2m}\sum(\hat{y}-y)^2$ | Standard error — fit the data |
| $\lambda$ | Hyperparameter — controls how hard you penalize |
| $\sum m_j^2$ | Sum of squared slopes — penalizes large coefficients |
 
### Visual — How Ridge Shrinks the Slope
 
```
WITHOUT Ridge               WITH Ridge
(Overfit line)              (Regularized line)
 
Y│  .                       Y│  .
 │.  . ← steep slope         │  .  .
 │  /\/\  m = 8.5             │   /     m = 1.2
 │/                           │  /  ← gentler slope
 └──────── X                  └──────── X
 
Cost = error only           Cost = error + λ×(8.5)²
     = near 0 (overfit)          = error + big penalty
                            → forces slope DOWN
```
 
### The Lambda (λ) Dial
 
```
λ = 0            λ = small         λ = large         λ = ∞
│                │                 │                 │
▼                ▼                 ▼                 ▼
Pure Linear   Slight            Strong            All slopes
Regression    regularization    regularization    → 0
(overfit)     (balanced)        (underfit risk)   (flat line)
```
 
- **λ too small:** Barely penalizes → still overfits
- **λ just right:** Balanced model, good generalization
- **λ too large:** Penalizes so hard all slopes shrink → underfitting
### What Ridge Does to Coefficients
 
```
Feature slopes BEFORE Ridge:    Feature slopes AFTER Ridge:
 
m₁ = 9.2  ██████████            m₁ = 1.1  █
m₂ = 7.8  ████████              m₂ = 0.9  █
m₃ = 0.1  ░                     m₃ = 0.05 ░
m₄ = 8.9  █████████             m₄ = 1.0  █
m₅ = 0.2  ░                     m₅ = 0.08 ░
 
All coefficients shrink significantly but NONE reach zero.
```
 
> **Key property of Ridge: slopes approach zero but never actually become zero.**
 
---
 
## 5. Lasso Regression — L1 Regularization
 
### The Cost Function
 
$$J_{Lasso} = \frac{1}{2m}\sum(\hat{y} - y)^2 + \lambda\sum |m_j|$$
 
The only difference: the penalty uses **absolute value** instead of square.
 
### Why Absolute Value Changes Everything
 
```
Ridge penalty: λ × m²         Lasso penalty: λ × |m|
 
  penalty                        penalty
  │         /                    │       /
  │        /                     │      /
  │       /  ← smooth curve      │     /  ← sharp V-shape
  │      /   slope always        │    /   slope changes
  │─────/    approaches 0        │───/    abruptly AT zero
  └──────── m                    └──────── m
       never reaches 0                CAN reach exactly 0
```
 
The **sharp corner at zero** in the absolute value function is what mathematically allows Lasso to push coefficients to exactly zero — while Ridge's smooth curve only asymptotically approaches zero.
 
### Lasso as Automatic Feature Selection
 
```
Feature slopes BEFORE Lasso:    Feature slopes AFTER Lasso:
 
m₁ = 9.2  ██████████            m₁ = 2.1  ██
m₂ = 7.8  ████████              m₂ = 0.0  ← ELIMINATED
m₃ = 0.1  ░                     m₃ = 0.0  ← ELIMINATED
m₄ = 8.9  █████████             m₄ = 1.8  █
m₅ = 0.2  ░                     m₅ = 0.0  ← ELIMINATED
 
Weak/irrelevant features get zeroed out completely.
Only the truly important features survive.
```
 
> This is **automatic feature selection** — Lasso literally removes useless features from the model without you manually deciding which to drop.
 
---
 
## 6. Ridge vs Lasso — Side-by-Side Comparison
 
| Feature | Ridge (L2) | Lasso (L1) |
|---|---|---|
| **Penalty term** | λ × Σm² | λ × Σ\|m\| |
| **Coefficient behavior** | Shrinks toward 0, never reaches it | Can shrink to exactly 0 |
| **Feature selection** | No — keeps all features | Yes — eliminates weak features |
| **Best used when** | All features are somewhat useful | Many irrelevant/redundant features |
| **Output model** | Dense (all features retained) | Sparse (only key features remain) |
| **Sensitivity to outliers** | More sensitive (squaring amplifies) | Less sensitive |
 
---
 
## 7. Visual — The Full Regularization Spectrum
 
```
            ◄─────────────────────────────────────────────►
        Underfitting                                Overfitting
            │                                           │
            │              IDEAL ZONE                   │
     ───────┼──────────────────────────────────────┼────
            │                                           │
     λ = ∞  │  λ large → λ optimal → λ small → λ = 0   │
     (flat  │                                    (pure  │
      line) │    Ridge or Lasso with tuned λ     linear │
            │    lives here                      regr.) │
 
           High Bias                           High Variance
           Low Variance                        Low Bias
```
 
---
 
## 8. How to Choose λ — Cross Validation
 
You don't guess λ. You use **k-fold cross validation:**
 
```
STEP 1: Try multiple λ values: [0.001, 0.01, 0.1, 1, 10, 100]
            │
STEP 2: For each λ, train on k-1 folds, validate on 1 fold
            │
STEP 3: Average validation error across all folds
            │
STEP 4: Pick the λ that gives lowest average validation error
            │
STEP 5: Retrain final model on ALL training data using best λ
```
 
---
 
## 9. When to Use Which
 
```
START
  │
  ├─ Do you suspect many features are irrelevant or redundant?
  │     └─ YES → Use LASSO (it will zero them out for you)
  │
  ├─ Are all your features likely meaningful?
  │     └─ YES → Use RIDGE (shrink all, keep all)
  │
  ├─ Not sure? Want both effects?
  │     └─ Use ELASTIC NET (combines Ridge + Lasso penalties)
  │
  └─ Is your data small with many features (p >> n)?
        └─ LASSO preferred (produces sparse, interpretable model)
```
 
---
 
## 10. Implementation — Quick Reference
 
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
 
# Ridge
ridge = Ridge(alpha=1.0)   # alpha is λ in sklearn
ridge.fit(X_train, y_train)
 
# Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
 
# Check which features Lasso zeroed out
import pandas as pd
coef_df = pd.Series(lasso.coef_, index=feature_names)
print(coef_df[coef_df == 0])   # eliminated features
print(coef_df[coef_df != 0])   # surviving features
 
# Finding best lambda via cross validation
from sklearn.linear_model import RidgeCV, LassoCV
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100])
ridge_cv.fit(X_train, y_train)
print("Best lambda:", ridge_cv.alpha_)
```
 
---
 
## 11. Key Formulas at a Glance
 
| Algorithm | Cost Function |
|---|---|
| Linear Regression | $\frac{1}{2m}\sum(\hat{y}-y)^2$ |
| Ridge (L2) | $\frac{1}{2m}\sum(\hat{y}-y)^2 + \lambda\sum m_j^2$ |
| Lasso (L1) | $\frac{1}{2m}\sum(\hat{y}-y)^2 + \lambda\sum\|m_j\|$ |
| Elastic Net | $\frac{1}{2m}\sum(\hat{y}-y)^2 + \lambda_1\sum\|m_j\| + \lambda_2\sum m_j^2$ |
 
---
 
## 12. Exam & Interview Traps
 
| Trap | Correct Answer |
|---|---|
| "Ridge can zero out coefficients" | **No** — only Lasso can reach exactly zero |
| "Lasso always beats Ridge" | Depends on data — if all features matter, Ridge is better |
| "λ = 0 means regularization is on" | λ = 0 means **no regularization** — pure linear regression |
| "Higher λ always improves model" | Too high → underfitting. Must tune λ carefully |
| "Regularization changes the algorithm" | No — it only changes the **cost function**; gradient descent still runs |
| "Ridge removes irrelevant features" | No — Ridge keeps all features (just shrinks them) |
 
---
 
## 13. Everything at a Glance
 
```
OVERFITTING DETECTED
(Train acc >> Test acc)
        │
        ▼
Add Penalty Term to Cost Function
        │
        ├─────────────────────────────────────┐
        ▼                                     ▼
   RIDGE (L2)                            LASSO (L1)
   J + λΣm²                              J + λΣ|m|
        │                                     │
        ▼                                     ▼
  All slopes shrink               Weak slopes → exactly 0
  None reach zero                 Strong slopes survive
        │                                     │
        ▼                                     ▼
  Good for: all               Good for: many irrelevant
  features relevant            features present
        │                                     │
        └──────────────┬──────────────────────┘
                       ▼
              Tune λ via Cross Validation
                       │
                       ▼
              Generalized Model ✓
         (Low Bias + Low Variance)
```
 
---
 
> **The entire intuition in one sentence:**
> Large slopes = overfit model. Penalize large slopes = generalized model.
> Ridge and Lasso are just two different mathematical ways of enforcing that same principle.