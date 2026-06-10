# Machine Learning — Complete Enhanced Notes
### Topics: Imbalanced Datasets · Under-sampling · Over-sampling · SMOTE · XGBoost Hyperparameter Tuning · AdaBoost

---

# PART 1: Handling Imbalanced Datasets — Core Problem

---

## 1. What is an Imbalanced Dataset?

An imbalanced dataset occurs when one class **drastically outnumbers** the other class, making standard classifiers biased toward the majority.

```
BALANCED DATASET               IMBALANCED DATASET

Class 0: ████████████ 500      Class 0: ████████████████████████ 9,000
Class 1: ████████████ 500      Class 1: █ 100

Ratio = 1:1 (ideal)            Ratio = 90:1 (dangerous)
```

### Real-World Examples

| Domain | Majority Class | Minority Class |
|---|---|---|
| **Fraud Detection** | Legitimate transactions | Fraudulent transactions |
| **Medical Diagnosis** | Healthy patients | Sick patients |
| **Spam Detection** | Normal emails | Spam emails |
| **Loan Default** | Non-defaulters | Defaulters |

---

## 2. Why Standard Algorithms Fail

```
Dataset: 9,000 Class 0 (Not Fraud) + 100 Class 1 (Fraud)
                    │
                    ▼
     Model predicts EVERYTHING as Class 0
                    │
                    ▼
     Accuracy = 9000/9100 = 98.9% ← looks great!

But:
  True Positives (Fraud caught)  = 0
  False Negatives (Fraud missed) = 100

The model is USELESS for its actual purpose.
High accuracy is LYING here.
```

> **Rule:** Never trust accuracy alone on imbalanced data. Always check Precision, Recall, and F1-Score on the minority class.

---

## 3. The Three Core Strategies

```
IMBALANCED DATA
       │
       ├─────────────────────────────┐─────────────────────┐
       ▼                             ▼                     ▼
UNDER-SAMPLING               OVER-SAMPLING           CLASS WEIGHTS
Remove majority              Increase minority        Penalize minority
class records                class records            misclassification
       │                             │                     │
       ▼                             ▼                     ▼
Fast but loses data          Preserves all data       No data change
Use only for huge            Preferred in practice    Built into algorithm
datasets                                              parameters
```

---

---

# PART 2: Under-Sampling — Strategy 1

---

## 1. How It Works

Under-sampling **reduces** the majority class to match the minority class size.

```
BEFORE Under-sampling:              AFTER Under-sampling:

Class 0: ██████████████ 9,000   →   Class 0: █ 100
Class 1: █              100     →   Class 1: █ 100

Action: Randomly DISCARD 8,900
        records from Class 0
```

### Mathematical View

```
Original:
  Class 0 = 9,000 records
  Class 1 =   100 records

Under-sampling removes:
  9,000 - 100 = 8,900 Class 0 records discarded

Result:
  Class 0 = 100 records ✓
  Class 1 = 100 records ✓
  Total   = 200 records (was 9,100)
```

---

## 2. The Critical Disadvantage

```
DATA LOSS IMPACT:
─────────────────

Original Class 0 distribution:

Feature X distribution (9,000 samples):
  ╔══════════════════════════════════╗
  ║  Rich, full distribution         ║
  ║  covering all edge cases         ║
  ║  and rare feature combinations   ║
  ╚══════════════════════════════════╝

After Under-sampling (100 samples):
  ╔═══╗
  ║   ║ ← Tiny fraction remains
  ╚═══╝
  Most variance and feature
  patterns permanently lost
```

**Consequences:**
- Model has much less data to learn from
- Poor generalization on test data
- Valuable feature distributions destroyed
- Rare patterns within majority class are lost entirely

---

## 3. When to Use Under-Sampling

```
DECISION TREE:

Is your dataset extremely large (millions of rows)?
            │
     ┌──────┴──────┐
     YES            NO
     │              │
     ▼              ▼
Under-sampling   Use Over-sampling
may be viable    or Class Weights
(savings justify (data loss too
 the data loss)   costly here)
```

### Python Implementation

```python
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Before
print("Before:", Counter(y))
# Output: Counter({0: 9000, 1: 100})

# Apply under-sampling (default = 1:1 ratio)
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# After
print("After:", Counter(y_resampled))
# Output: Counter({0: 100, 1: 100})
```

---

---

# PART 3: Over-Sampling & SMOTE — Strategy 2

---

## 1. Random Over-Sampler

The simplest over-sampling method — **duplicates** existing minority class records until balance is achieved.

```
BEFORE:                          AFTER (Random Over-Sampler):

Class 0: ██████████ 1,000    →   Class 0: ██████████ 1,000
Class 1: ██          200     →   Class 1: ██████████ 1,000

Action: Duplicate 800 records
        from existing Class 1
        samples (with replacement)
```

**Problem with random duplication:**

```
Original minority points:          After duplication:

    *                                  * * *
      *    *                         * * * * *
    *    *                           * * * * * *

Only 5 unique points exist         Still only 5 unique points
but duplicated 200 times           — overfitting risk!
```

---

## 2. SMOTE — Synthetic Minority Over-sampling Technique

SMOTE is **far superior** — it generates **entirely new, synthetic data points** using k-nearest neighbors interpolation.

### How SMOTE Generates Synthetic Points

```
STEP 1: Take a minority class point X₁
STEP 2: Find its K nearest neighbors (e.g., K=5)
STEP 3: Randomly pick one neighbor X₂
STEP 4: Generate a new point along the line between X₁ and X₂

Formula:
  X_new = X₁ + λ × (X₂ - X₁)
  where λ is a random value between 0 and 1

Visual:

  X₁ ●─────────────────────● X₂
       ↑                   ↑
  existing point        nearest neighbor

      X₁ ●──●──●──●──●─────● X₂
              ↑  ↑  ↑
          NEW synthetic points
          generated between them
```

### SMOTE vs Random Over-Sampler

```
Random Over-Sampler:             SMOTE:

  * ← (copied)                   * ← original
  * ← (copied)                   * ← synthetic (new point)
  * ← original                   * ← synthetic (new point)
  * ← (copied)                   * ← original
  * ← (copied)                   * ← synthetic (new point)

Exact duplicates                 Diverse, new data points
Overfitting risk                 Better generalization
No new information               New feature combinations created
```

---

## 3. SMOTETomek — Hybrid Technique (Best of Both Worlds)

SMOTETomek combines **two operations**:

1. **SMOTE** → Generates synthetic minority samples
2. **Tomek Links** → Cleans noisy boundary overlap points

```
BEFORE SMOTETomek:

  Class 0: ■ ■ ■ ■ ■ ■ ■ ■ ■
                ■ ■ ● ■           ← Class 0 points near boundary
                ○ ● ○ ●           ← Noisy overlap zone (Tomek Links)
             ○ ○ ○ ○ ○ ○
  Class 1:  ○ ○ ○ ○ ○ ○ ○

AFTER SMOTE generates new ○ points:
  Class 1 is numerically balanced but boundary is noisy.

AFTER Tomek Links removes boundary noise:

  Class 0: ■ ■ ■ ■ ■ ■ ■ ■ ■
                ■ ■ ■               ← Clean boundary
                ─────────────       ← Clear decision line
                ○ ○ ○ ○
  Class 1:  ○ ○ ○ ○ ○ ○ ○

Result: Balanced + clean separation ✓
```

---

## 4. Sampling Strategy — Ratio Control

```
sampling_strategy parameter controls the final ratio:

Default (1.0):  minority = majority  → 1:1 ratio
  Class 0: 284,315  →  Class 0: 284,315
  Class 1:   1,492  →  Class 1: 284,315

Custom (0.5):   minority = 0.5 × majority
  Class 0: 284,315  →  Class 0: 284,315
  Class 1:   1,492  →  Class 1: 142,157

Formula:
  target_minority_count = sampling_strategy × majority_count
  0.5 × 284,315 = 142,157 synthetic/duplicated samples
```

---

## 5. Complete Python Implementation Pipeline

```python
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek

# ─────────────────────────────────────────────
# STEP 1: Check class distribution
# ─────────────────────────────────────────────
print("Original distribution:", Counter(y))
# Counter({0: 284315, 1: 492})

# ─────────────────────────────────────────────
# STEP 2: Train-Test Split BEFORE resampling
# (never resample test data — data leakage!)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# STEP 3: Apply SMOTE on training data only
# ─────────────────────────────────────────────
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("After SMOTE:", Counter(y_train_res))

# ─────────────────────────────────────────────
# STEP 4: Apply SMOTETomek (hybrid approach)
# ─────────────────────────────────────────────
smt = SMOTETomek(random_state=42)
X_train_smt, y_train_smt = smt.fit_resample(X_train, y_train)

print("After SMOTETomek:", Counter(y_train_smt))

# ─────────────────────────────────────────────
# STEP 5: Train model on resampled data
# ─────────────────────────────────────────────
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_smt, y_train_smt)

# ─────────────────────────────────────────────
# STEP 6: Evaluate on ORIGINAL test data
# ─────────────────────────────────────────────
y_pred = clf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Always check precision, recall, f1 for BOTH classes
# Not just overall accuracy
```

---

## 6. Strategy Comparison Table

| Technique | Data Change | Pros | Cons | Best For |
|---|---|---|---|---|
| **Under-sampling** | Removes majority | Fast, simple | Massive data loss | Very large datasets |
| **Random Over-Sampler** | Duplicates minority | Simple, no data loss | Overfitting risk | Small datasets |
| **SMOTE** | Synthesizes minority | New data, no exact copies | Can create noisy samples | Most real-world cases |
| **SMOTETomek** | Synthesizes + cleans | Balanced + clean boundaries | Slower | Best overall quality |
| **Class Weights** | None | No data change | Only adjusts loss function | Algorithm supports it |

---

---

# PART 4: XGBoost Hyperparameter Optimization

---

## 1. Why Hyperparameter Tuning Matters

```
DEFAULT PARAMETERS               TUNED PARAMETERS

  Model accuracy: 84%    →→→     Model accuracy: 91%
  Overfitting present    →→→     Well-generalized model
  Sub-optimal splits     →→→     Optimal tree structure

The algorithm is the same.
Only the configuration changes.
Tuning finds the BEST configuration
without changing the algorithm itself.
```

---

## 2. Grid Search CV vs Randomized Search CV

```
GRID SEARCH CV                   RANDOMIZED SEARCH CV
──────────────────────           ──────────────────────────
Tries EVERY combination          Tries N RANDOM combinations

params = {                       params = {
  lr: [0.01, 0.1, 0.3]            lr: uniform(0.01, 0.3)
  depth: [3, 5, 7]                depth: randint(3, 10)
  gamma: [0, 0.1, 0.2]            gamma: uniform(0, 0.5)
}                                }

3 × 3 × 3 = 27 combinations      n_iter = 20 (you choose)
EVERY one evaluated               20 RANDOM samples evaluated

Exhaustive but SLOW               Fast and nearly as accurate
Good for small param grids        Preferred for large param spaces
```

### Computational Cost Comparison

```
Grid Search:
  5 hyperparameters × 6 values each = 7,776 combinations
  × 10-fold CV = 77,760 model fits
  Time: HOURS

Randomized Search:
  n_iter = 50 random combinations
  × 10-fold CV = 500 model fits
  Time: MINUTES

Result quality difference: minimal
Speed difference: massive
→ Randomized Search preferred in practice
```

---

## 3. Key XGBoost Hyperparameters

### learning_rate (eta) — Step Size

```
learning_rate controls how much each tree contributes:

High lr (0.3):                   Low lr (0.05):
  ●                                ●
   ↓ big jump                       ↓ small step
     ● overshoot risk               ●
                                    ●
                                    ●
                                    ● more iterations
                                      but better precision

Typical range: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
Default: 0.3
```

### max_depth — Tree Complexity

```
max_depth = 2 (simple):          max_depth = 6 (complex):

     [Feature A]                      [Feature A]
    /           \                    /             \
[Class 0]  [Feature B]         [Feat B]          [Feat C]
           /          \        /      \           /     \
      [Class 1]  [Class 0]  [F D]  [F E]      [F F]  [Class]
                            / \    / \         / \
                          ...  ... ... ...   ...  ...

Low depth = underfitting         High depth = overfitting
Typical range: [3, 4, 5, 6, 8, 10, 12, 15]
```

### All Key Hyperparameters Reference

| Parameter | What it Controls | Typical Range | Effect When High |
|---|---|---|---|
| **learning_rate** | Step size per tree | [0.05 → 0.30] | Faster but less precise |
| **max_depth** | Tree depth/complexity | [3 → 15] | More complex, overfit risk |
| **min_child_weight** | Min samples in leaf | [1, 3, 5, 7] | Simpler trees, underfitting |
| **gamma** | Min loss for split | [0.0 → 0.4] | More conservative splits |
| **colsample_bytree** | Feature subsample per tree | [0.3 → 0.7] | More randomness, less overfit |

---

## 4. Complete Hyperparameter Tuning Pipeline

```python
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1: Define Parameter Space
# ─────────────────────────────────────────────
params = {
    'learning_rate':    [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    'max_depth':        [3, 4, 5, 6, 8, 10, 12, 15],
    'min_child_weight': [1, 3, 5, 7],
    'gamma':            [0.0, 0.1, 0.2, 0.3, 0.4],
    'colsample_bytree': [0.3, 0.4, 0.5, 0.7]
}

# ─────────────────────────────────────────────
# STEP 2: Initialize Base Classifier
# ─────────────────────────────────────────────
xgb_base = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# ─────────────────────────────────────────────
# STEP 3: Randomized Search CV
# ─────────────────────────────────────────────
random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=params,
    n_iter=50,           # Try 50 random combinations
    scoring='roc_auc',   # Optimize for AUC-ROC
    cv=5,                # 5-fold cross validation
    verbose=1,
    random_state=42,
    n_jobs=-1            # Use all CPU cores
)

random_search.fit(X_train, y_train)

# ─────────────────────────────────────────────
# STEP 4: Extract Best Parameters
# ─────────────────────────────────────────────
print("Best Parameters Found:")
print(random_search.best_params_)
print(f"Best CV Score: {random_search.best_score_:.4f}")

# ─────────────────────────────────────────────
# STEP 5: Cross-Validate the Best Model
# ─────────────────────────────────────────────
best_model = random_search.best_estimator_

cv_scores = cross_val_score(
    best_model, X_train, y_train,
    cv=10,              # 10-fold cross validation
    scoring='roc_auc'
)

print(f"\nCross-Validation Results (10-fold):")
print(f"Individual fold scores: {cv_scores.round(4)}")
print(f"Mean Score:  {cv_scores.mean():.4f}")
print(f"Std Dev:     {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# STEP 6: Final Evaluation on Test Set
# ─────────────────────────────────────────────
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\nFinal Test Set Performance:")
print(classification_report(y_test, y_pred))
```

---

## 5. Cross-Validation — Why score.mean() Matters

```
10-Fold Cross Validation:

Full Training Data split into 10 equal folds:
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘

Run 1:  [VAL│ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 ] → score₁
Run 2:  [ 1 │VAL│ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 ] → score₂
Run 3:  [ 1 │ 2 │VAL│ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 ] → score₃
...
Run 10: [ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │VAL] → score₁₀

Final Score = mean(score₁, score₂, ..., score₁₀)
            = Stable, unbiased performance estimate ✓

Why mean? → Removes luck of a single train/test split
            Tests model on ALL parts of the data
            Low std dev = model is consistently reliable
```

---

---

# PART 5: AdaBoost — Adaptive Boosting

---

## 1. What is AdaBoost?

AdaBoost (Adaptive Boosting) converts **multiple weak learners** into a **single strong learner** by training them sequentially — each new model focusing on the mistakes of the previous one.

```
WEAK LEARNER:                    STRONG LEARNER (AdaBoost):
─────────────                    ──────────────────────────
Accuracy ≈ 51-60%                Accuracy >> individual stumps
Just slightly better             Combined power of many
than random guessing             focused stumps

One decision stump               Ensemble of 100+ stumps
splits on 1 feature              each correcting the last
```

---

## 2. Decision Stumps — The Building Block

A **Decision Stump** is a decision tree with **depth = 1** (one split, two leaves).

```
DECISION STUMP (depth=1):        DECISION TREE (depth=5):

     [Age ≤ 30?]                      [Age ≤ 30?]
    /           \                    /             \
[Class 0]   [Class 1]          [Salary ≤ 50k]  [Exp ≤ 3yr]
                               /        \        /       \
                           [C0]       [C1]   [C0]     [Gender?]
                                                       /    \
                                                      ...   ...

Single split on ONE feature      Deep complex tree
Weak learner (barely useful)     Strong learner alone
Perfect for AdaBoost             Prone to overfitting
```

---

## 3. Step-by-Step AdaBoost Mechanism

### STEP 1 — Initialize Equal Weights

```
Dataset: N = 8 training samples

Row:    1     2     3     4     5     6     7     8
Weight: 1/8   1/8   1/8   1/8   1/8   1/8   1/8   1/8

All weights equal = 0.125
Sum of weights = 1.0 ✓
```

### STEP 2 — Train First Decision Stump

```
Stump 1 splits on Feature A ≤ threshold:

Row:    1   2   3   4   5   6   7   8
Actual: +   +   +   -   -   -   +   -
Pred:   +   +   -   -   -   -   +   -
                ↑
           WRONG on row 3 (+ predicted as -)
```

### STEP 3 — Update Weights

```
AFTER Stump 1 predictions:

Correct predictions   → DECREASE weight
Incorrect predictions → INCREASE weight

Row:    1     2     3     4     5     6     7     8
Actual: +     +     +     -     -     -     +     -
Result: ✓     ✓     ✗     ✓     ✓     ✓     ✓     ✓
Weight: 0.071 0.071 0.500 0.071 0.071 0.071 0.071 0.071

↑ Row 3 (misclassified) gets MUCH higher weight
  Normalize all weights so they sum to 1.0 again

Visual weight bars:
Row 3: ████████████████████ 0.500  ← AMPLIFIED
Row 1: ██ 0.071
Row 2: ██ 0.071
Row 4: ██ 0.071
...
```

### STEP 4 — Train Second Stump on Updated Weights

```
Stump 2 is now FORCED to focus on Row 3
(high weight = high importance in training)

Stump 2 splits on Feature B ≤ threshold
→ Correctly classifies Row 3 this time ✓
→ May now misclassify other low-weight rows
→ Those rows get amplified weights for Stump 3
```

### STEP 5 — Repeat Sequentially

```
Stump 1 → Update weights → Stump 2 → Update weights → Stump 3 → ...

Each stump is a specialist:
  Stump 1: Good at easy cases
  Stump 2: Focuses on cases Stump 1 missed
  Stump 3: Focuses on cases Stump 2 missed
  ...
  Stump N: Each fills in gaps left by previous
```

---

## 4. Weight Flow Visual

```
Initial:    [●●●●●●●●]  All equal weights

After S1:   [●●█●●●●●]  Row 3 amplified

After S2:   [●●●●●█●●]  Row 6 now amplified (S2 missed it)

After S3:   [●█●●●●●●]  Row 2 now amplified (S3 missed it)

Final:      Each stump trained on a DIFFERENT
            version of the dataset based on
            which samples matter most

● = normal weight    █ = amplified weight
```

---

## 5. Final Prediction — Weighted Majority Vote

Unlike Random Forest (equal votes), AdaBoost uses **weighted voting** based on each stump's performance.

### Amount of Say (Stump Weight)

```
A stump's Amount of Say ∝ 1 / error rate

Low error rate  → HIGH Amount of Say
High error rate → LOW Amount of Say

Example stumps:
  Stump 1: Error = 0.10 → Amount of Say = 1.10
  Stump 2: Error = 0.30 → Amount of Say = 0.42
  Stump 3: Error = 0.45 → Amount of Say = 0.10

Weighted Vote for new test point:
  Stump 1 says: Class + (weight 1.10)
  Stump 2 says: Class + (weight 0.42)
  Stump 3 says: Class - (weight 0.10)

  Class +: 1.10 + 0.42 = 1.52
  Class -: 0.10

  Final Prediction: Class + (higher weighted sum) ✓
```

---

## 6. AdaBoost vs Random Forest

```
RANDOM FOREST                    ADABOOST
─────────────────────────        ─────────────────────────────
Deep decision trees              Decision stumps (depth=1)
Built in PARALLEL                Built SEQUENTIALLY
All trees equal vote             Weighted vote (by accuracy)
Bootstrap sampling               Weighted sampling
Reduces variance (overfit)       Reduces bias (underfit)
Robust to noise                  Sensitive to noise/outliers
Independent trees                Each tree depends on previous
```

| Property | Random Forest | AdaBoost |
|---|---|---|
| **Base learner** | Full decision tree | Decision stump (depth=1) |
| **Training** | Parallel | Sequential |
| **Voting** | Equal weight | Weighted by error |
| **Focus** | Variance reduction | Bias reduction |
| **Noise sensitivity** | Low | High |
| **Overfitting risk** | Low | Medium (with noisy data) |

---

## 7. Python Implementation

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
# STEP 1: Base estimator = Decision Stump
# ─────────────────────────────────────────────
base_stump = DecisionTreeClassifier(
    max_depth=1,        # stump = depth 1
    random_state=42
)

# ─────────────────────────────────────────────
# STEP 2: Build AdaBoost Ensemble
# ─────────────────────────────────────────────
ada = AdaBoostClassifier(
    base_estimator=base_stump,
    n_estimators=200,       # number of stumps
    learning_rate=1.0,      # shrinkage per stump
    algorithm='SAMME.R',    # uses probability estimates
    random_state=42
)

# ─────────────────────────────────────────────
# STEP 3: Train and Evaluate
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)

print("AdaBoost Results:")
print(classification_report(y_test, y_pred))

# ─────────────────────────────────────────────
# STEP 4: Plot accuracy vs number of estimators
# ─────────────────────────────────────────────
train_scores = []
test_scores = []
estimator_range = range(10, 201, 10)

for n in estimator_range:
    ada_temp = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n,
        random_state=42
    )
    ada_temp.fit(X_train, y_train)
    train_scores.append(ada_temp.score(X_train, y_train))
    test_scores.append(ada_temp.score(X_test, y_test))

plt.figure(figsize=(10, 5))
plt.plot(estimator_range, train_scores, 'b-o',
         label='Train Accuracy', linewidth=2)
plt.plot(estimator_range, test_scores, 'r-s',
         label='Test Accuracy', linewidth=2)
plt.title('AdaBoost — Accuracy vs Number of Stumps')
plt.xlabel('Number of Estimators (Stumps)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

---

---

# PART 6: Putting It All Together — Complete Pipeline

---

## End-to-End Workflow

```
RAW IMBALANCED DATA
        │
        ▼
Exploratory Data Analysis
  ├── Check class distribution (Counter / value_counts)
  ├── Visualize imbalance (bar chart)
  └── Choose resampling strategy
        │
        ▼
Train-Test Split FIRST (stratify=y)
        │
        ▼
Apply Resampling on TRAIN SET ONLY
  ├── Small dataset → SMOTE or SMOTETomek
  ├── Large dataset → Under-sampling or SMOTE(ratio=0.5)
  └── Algorithm supports it → class_weight='balanced'
        │
        ▼
Feature Scaling (if needed)
        │
        ▼
Model Selection + Hyperparameter Tuning
  ├── RandomizedSearchCV (faster, preferred)
  └── GridSearchCV (exhaustive, small grids only)
        │
        ▼
Cross-Validation (10-fold)
  └── score.mean() ± score.std()
        │
        ▼
Final Evaluation on TEST SET
  ├── Confusion Matrix
  ├── Classification Report (precision, recall, f1 per class)
  └── AUC-ROC Score (best for imbalanced)
        │
        ▼
DEPLOYED MODEL ✓
```

---

## Quick Reference — Key Formulas

| Formula | Name | Used In |
|---|---|---|
| $X_{new} = X_1 + \lambda(X_2 - X_1)$ | SMOTE interpolation | Synthetic point generation |
| $\text{target} = \text{strategy} \times \text{majority}$ | Sampling ratio | Over/Under-sampling |
| $\text{CV Score} = \frac{1}{k}\sum_{i=1}^k \text{score}_i$ | k-fold mean score | Hyperparameter validation |
| $\text{Amount of Say} \propto \frac{1}{\text{error}}$ | Stump weight | AdaBoost weighted vote |
| $W_{wrong} \uparrow,\ W_{correct} \downarrow$ | Weight update | AdaBoost sequential learning |

---

## Interview & Exam Traps

| Trap | Correct Answer |
|---|---|
| "High accuracy = good model on imbalanced data" | **No** — a model predicting all majority class gets high accuracy but is useless |
| "Under-sampling is always the worst choice" | It's valid for **very large datasets** where computational savings justify data loss |
| "SMOTE duplicates minority class records" | **No** — SMOTE creates **entirely new synthetic points** using KNN interpolation |
| "Apply SMOTE before train-test split" | **Never** — apply SMOTE only on training data; applying before split = **data leakage** |
| "Grid Search always finds better params than Randomized" | Not necessarily — Randomized Search finds near-optimal in a fraction of the time |
| "All AdaBoost stumps get equal votes" | **No** — stumps vote with **weighted Amount of Say** based on their error rate |
| "AdaBoost builds trees in parallel like Random Forest" | **No** — AdaBoost is **sequential**; each stump depends on the previous one's errors |
| "RandomizedSearchCV with n_iter=50 means 50 models" | It means 50 **parameter combinations** × **cv folds** = 50 × cv total model fits |

---

> **Core philosophy:** Real-world data is messy, imbalanced, and rarely perfect. The best ML engineers don't just pick an algorithm — they handle the data distribution first (SMOTE/sampling), then find the optimal configuration (hyperparameter tuning), and always validate with robust cross-validation, not a single train-test split.