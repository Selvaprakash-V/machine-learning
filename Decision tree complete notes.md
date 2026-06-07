# Decision Trees — Complete Master Notes
### Topics: Entropy · Information Gain · Gini Impurity · Numerical Feature Splitting

---

# PART 1: Entropy in Decision Trees

---

## 1. What is Entropy?

Entropy is a metric borrowed from **information theory** that measures the **impurity, randomness, or uncertainty** within a subset of data at any node in a Decision Tree.

> **Core goal of a Decision Tree:** Split data repeatedly to group similar labels together. Entropy tells the algorithm *how mixed* the classes are at each node.

---

## 2. The Purity Principle

```
PURE NODE                        IMPURE NODE
(Entropy = 0)                    (Entropy = 1)

  Node                             Node
  ┌─────────────┐                  ┌─────────────┐
  │ Yes Yes Yes │                  │ Yes No Yes  │
  │ Yes Yes Yes │                  │ No Yes No   │
  │ Yes Yes Yes │                  │ Yes No Yes  │
  └─────────────┘                  └─────────────┘
  All same class                   Mixed classes
  Zero uncertainty                 Maximum uncertainty
  Algorithm LOVES this             Algorithm wants to split this
```

| Node Type | Definition | Entropy Value |
|---|---|---|
| **Pure Node** | All records belong to ONE class | 0 |
| **Impure Node** | Records are randomly/evenly mixed | Close to 1 |

---

## 3. Mathematical Formula

**Binary Classification:**

$$H(S) = -P(+)\log_2(P(+)) - P(-)\log_2(P(-))$$

**Multi-class (c classes):**

$$H(S) = -\sum_{i=1}^{c} P_i \log_2(P_i)$$

| Symbol | Meaning |
|---|---|
| $H(S)$ | Entropy of dataset/node S |
| $P(+)$ | Proportion of positive class instances |
| $P(-)$ | Proportion of negative class instances |
| $P_i$ | Proportion of class i instances |

---

## 4. Worked Example

**Suppose a node has 10 records: 5 Yes, 5 No**

$$P(+) = \frac{5}{10} = 0.5 \quad P(-) = \frac{5}{10} = 0.5$$

$$H(S) = -(0.5)\log_2(0.5) - (0.5)\log_2(0.5)$$

$$H(S) = -(0.5 \times -1) - (0.5 \times -1) = 0.5 + 0.5 = \mathbf{1.0}$$

Maximum entropy — perfectly mixed, worst case for classification.

**Now suppose: 10 records: 10 Yes, 0 No**

$$H(S) = -(1.0)\log_2(1.0) - (0)\log_2(0) = 0 + 0 = \mathbf{0}$$

Zero entropy — perfectly pure, best case.

---

## 5. Entropy Curve Visual

```
Entropy (H)
    │
1.0 ┤          ╭━━━━━╮
    │        ╭─╯     ╰─╮
0.5 ┤      ╭─╯         ╰─╮
    │    ╭─╯             ╰─╮
0.0 ┤────┴─────────────────┴────
    0.0        0.5          1.0
               P(+) →

Key points:
• P(+) = 0.0 → H = 0 (all negative, pure)
• P(+) = 0.5 → H = 1 (50/50 split, max uncertainty)
• P(+) = 1.0 → H = 0 (all positive, pure)
```

---

## 6. Entropy Value Interpretation

| Entropy Value | Distribution | Meaning |
|---|---|---|
| **0.0** | All one class | Perfectly pure — no uncertainty |
| **0.0 – 0.5** | Mostly one class | Low impurity — good node |
| **0.5 – 1.0** | Somewhat mixed | High impurity — needs splitting |
| **1.0** | Exactly 50/50 | Maximum impurity — worst classification structure |

---

## 7. Algorithm Context — ID3

```
ID3 Algorithm Flow:
───────────────────

ROOT NODE (full dataset)
        │
        ▼
  Calculate H(S) ← entropy of current node
        │
        ▼
  Try splitting on each feature
        │
        ▼
  Pick feature with highest Information Gain
        │
        ├──────────────┐
        ▼              ▼
   Branch A        Branch B
  H(S_A) low?     H(S_B) low?
        │              │
        ▼              ▼
  Keep splitting until H = 0 (pure leaf nodes)
```

> **Goal:** Drive entropy down to 0 at every leaf node as fast as possible — keeping the tree shallow and efficient.

---

---

# PART 2: Information Gain

---

## 1. What is Information Gain?

While Entropy measures impurity at **one node**, Information Gain (IG) measures how much impurity is **reduced** by splitting on a specific feature.

> **Simple definition:** How much does knowing a feature's value reduce our uncertainty about the target label?

```
BEFORE SPLIT              AFTER SPLIT ON FEATURE F
Parent Node               ┌──────────┬──────────┐
H(S) = 0.94              Child A    │         Child B
(mixed data)              H = 0.2   │          H = 0.3
                          (mostly   │         (mostly
                           Yes)     │          No)

Information Gain = 0.94 - weighted_average(0.2, 0.3)
                 = HIGH → this was a GREAT split!
```

---

## 2. Mathematical Formula

$$\text{Gain}(S, F) = H(S) - \sum_{v \in \text{Values}(F)} \left(\frac{|S_v|}{|S|} \times H(S_v)\right)$$

| Symbol | Meaning |
|---|---|
| $H(S)$ | Entropy of the parent node |
| $F$ | The feature being evaluated for splitting |
| $v$ | Each unique value/branch of feature F |
| $\|S_v\|$ | Number of samples that fall into branch v |
| $\|S\|$ | Total samples in the parent node |
| $H(S_v)$ | Entropy of child node v |

---

## 3. Worked Example

**Parent node:** 14 records, H(S) = 0.94

**Feature: Outlook** → values: Sunny, Overcast, Rain

| Branch | Records | Entropy |
|---|---|---|
| Sunny | 5 | 0.97 |
| Overcast | 4 | 0.00 |
| Rain | 5 | 0.97 |

$$\text{Gain} = 0.94 - \left(\frac{5}{14}(0.97) + \frac{4}{14}(0.00) + \frac{5}{14}(0.97)\right)$$

$$= 0.94 - (0.346 + 0 + 0.346) = 0.94 - 0.692 = \mathbf{0.248}$$

Repeat this for EVERY feature, then pick the one with the **highest gain**.

---

## 4. Feature Selection Logic

```
Calculate IG for ALL available features at current node:

Feature: Outlook      → IG = 0.248  ← HIGHEST → SELECT THIS
Feature: Temperature  → IG = 0.029
Feature: Humidity     → IG = 0.152
Feature: Wind         → IG = 0.048

Rule: Highest Information Gain = Best Split Feature
      This feature becomes the split node at this level.
      Repeat for every subsequent child node.
```

---

## 5. Information Gain Interpretation

| IG Value | Meaning |
|---|---|
| **High IG** | Splitting on this feature greatly reduces uncertainty — great split |
| **Low IG** | Splitting on this feature barely helps — poor split |
| **IG = 0** | Feature provides zero useful information — never split on this |

---

## 6. Full Decision Tree Building Flow

```
Dataset (Root Node)
        │
        ▼
For each feature → compute Information Gain
        │
        ▼
Select feature with MAX Information Gain → Root Split
        │
        ├──────────────┬──────────────┐
        ▼              ▼              ▼
    Branch 1       Branch 2       Branch 3
        │
        ▼
For each feature (remaining) → compute IG again
        │
        ▼
Select MAX IG feature → Next level split
        │
        ▼
Repeat until:
  - All leaf nodes are pure (H = 0), OR
  - Max depth reached, OR
  - Minimum samples threshold hit
```

---

---

# PART 3: Gini Impurity

---

## 1. What is Gini Impurity?

Gini Impurity is an **alternative to Entropy** for measuring node impurity. It is the **default metric in Scikit-Learn's DecisionTreeClassifier** because it avoids expensive logarithm calculations.

> **Simple definition:** The probability that a randomly chosen record from a node would be *incorrectly* classified if labeled randomly according to the class distribution.

---

## 2. Mathematical Formula

**General formula (c classes):**

$$\text{G.I.} = 1 - \sum_{i=1}^{c}(P_i)^2$$

**Binary classification expanded:**

$$\text{G.I.} = 1 - \left[(P(+))^2 + (P(-))^2\right]$$

---

## 3. Worked Examples

**Case 1 — Pure Node (all Yes):**

$$P(+) = 1.0, \quad P(-) = 0.0$$

$$\text{G.I.} = 1 - [1.0^2 + 0.0^2] = 1 - [1 + 0] = \mathbf{0}$$

**Case 2 — Perfectly Mixed Node (50/50):**

$$P(+) = 0.5, \quad P(-) = 0.5$$

$$\text{G.I.} = 1 - [0.5^2 + 0.5^2] = 1 - [0.25 + 0.25] = 1 - 0.5 = \mathbf{0.5}$$

**Case 3 — Mostly one class (8 Yes, 2 No):**

$$P(+) = 0.8, \quad P(-) = 0.2$$

$$\text{G.I.} = 1 - [0.8^2 + 0.2^2] = 1 - [0.64 + 0.04] = 1 - 0.68 = \mathbf{0.32}$$

---

## 4. Gini Impurity Range Visual

```
Gini Impurity
    │
0.5 ┤           ╭━━━━━╮
    │         ╭─╯     ╰─╮
0.25┤       ╭─╯         ╰─╮
    │     ╭─╯             ╰─╮
0.0 ┤─────┴─────────────────┴─────
    0.0          0.5          1.0
                P(+) →

Max = 0.5 (50/50 split)
Min = 0.0 (pure node)
```

| G.I. Value | Meaning |
|---|---|
| **0.0** | Perfectly pure node |
| **0.0 – 0.25** | Low impurity — good node |
| **0.25 – 0.5** | High impurity — needs splitting |
| **0.5** | Maximum impurity — 50/50 mix |

---

## 5. Gini vs Entropy — Why Gini Wins on Speed

```
ENTROPY Calculation:              GINI Calculation:
────────────────────              ────────────────────
-P(+) × log₂(P(+))               1 - [P(+)² + P(-)]²
-P(-) × log₂(P(-))

Requires LOGARITHM                Requires only
operations at every node          MULTIPLICATION and
                                  SUBTRACTION

log₂ is expensive for             Simple arithmetic —
CPU when repeated                 extremely fast at scale
millions of times
```

> **At large scale:** On a dataset with millions of rows and dozens of features, Gini's arithmetic simplicity results in **noticeably faster training times** compared to Entropy's logarithmic operations.

---

---

# PART 4: Decision Tree Splitting for Numerical Features

---

## 1. The Problem with Continuous Values

Categorical features like `["Red", "Green", "Blue"]` provide **natural discrete branches**. But continuous numerical features like `Age`, `Salary`, `Temperature` have **no natural split points** — the algorithm must discover the optimal threshold boundary itself.

```
CATEGORICAL (easy)               NUMERICAL (must find threshold)

Color = ?                        Age = ?
  ├── Red                          ├── Age ≤ 25.5?  YES/NO
  ├── Green                        ├── Age ≤ 30.0?  YES/NO
  └── Blue                         ├── Age ≤ 42.5?  YES/NO
                                   └── Age ≤ 55.0?  YES/NO

Natural branches exist           Threshold must be discovered
```

---

## 2. Step-by-Step Splitting Process

```
STEP 1: SORT
────────────
Take all values of the numerical feature in the active node.
Sort them in ascending order.

Age values: [22, 25, 28, 35, 40, 45, 50, 55]
Sorted:     [22, 25, 28, 35, 40, 45, 50, 55]


STEP 2: GENERATE THRESHOLD CANDIDATES
───────────────────────────────────────
Take midpoints between adjacent unique values:

(22+25)/2 = 23.5  → candidate threshold
(25+28)/2 = 26.5  → candidate threshold
(28+35)/2 = 31.5  → candidate threshold
(35+40)/2 = 37.5  → candidate threshold
... and so on


STEP 3: EVALUATE EACH THRESHOLD
─────────────────────────────────
For EACH candidate threshold, split data into two branches:

  Branch A: Age ≤ 23.5   |   Branch B: Age > 23.5
  Calculate Entropy/Gini  |   Calculate Entropy/Gini
  Compute weighted average entropy of children
  Record Information Gain

Repeat for every single threshold candidate.


STEP 4: SELECT OPTIMAL THRESHOLD
──────────────────────────────────
Compare Information Gain across ALL thresholds.
Lock in the threshold with the HIGHEST Information Gain.

Example result:
  Age ≤ 23.5 → IG = 0.12
  Age ≤ 31.5 → IG = 0.34  ← WINNER
  Age ≤ 37.5 → IG = 0.21

Final split: Age ≤ 31.5
  ├── YES (Age ≤ 31.5)
  └── NO  (Age > 31.5)
```

---

## 3. Visual — The Full Scan Process

```
Sorted Ages: [22, 25, 28, 35, 40, 45, 50, 55]

Scan →  23.5  26.5  31.5  37.5  42.5  47.5  52.5
         ↓     ↓     ↓     ↓     ↓     ↓     ↓
        IG=   IG=   IG=   IG=   IG=   IG=   IG=
        0.12  0.18  0.34  0.21  0.15  0.09  0.05
                     ↑
               MAXIMUM IG
               Split here: Age ≤ 31.5
```

---

## 4. Computational Cost — The Major Drawback

```
Dataset size: 1,000,000 rows
Numerical features: 20

For EACH numerical feature:
  - Sort 1M values
  - Generate ~1M threshold candidates
  - For each candidate: split + compute entropy × 2 child nodes
  - Compare all IGs

Total operations per node ≈ 1M × 20 = 20,000,000 calculations

And this repeats at EVERY node in the tree.
```

| Scale Factor | Impact |
|---|---|
| More rows | More thresholds to evaluate per feature |
| More numerical features | More full scans per node |
| Deeper tree | More nodes × the above cost |
| Ensemble methods (RF, XGBoost) | Multiple trees × the above cost |

> **This is why Random Forests and XGBoost can be slow to train on large numerical datasets** — they each run hundreds/thousands of Decision Trees, each performing this expensive scan internally.

---

---

# PART 5: Comprehensive Comparison — All Three Methods

---

## Side-by-Side Comparison Table

| Property | Entropy | Gini Impurity | Numerical Splits |
|---|---|---|---|
| **Math operation** | Logarithm ($\log_2$) | Squaring (arithmetic) | Sorting + scanning |
| **Binary range** | 0.0 to 1.0 | 0.0 to 0.5 | Variable boundary |
| **Pure node value** | 0 | 0 | N/A |
| **Max impurity value** | 1.0 | 0.5 | N/A |
| **Default in sklearn** | No | **Yes** | Handled automatically |
| **Computation speed** | Slower | **Faster** | Expensive at scale |
| **Used in** | ID3 Algorithm | CART (sklearn default) | All tree algorithms |
| **Best for** | Theoretical study | Production/large datasets | Continuous features |

---

## When to Use Which Criterion

```
CHOOSING YOUR SPLITTING CRITERION
──────────────────────────────────

START
  │
  ├── Small dataset, academic/research use?
  │         └─ Use ENTROPY (ID3) — more theoretically grounded
  │
  ├── Large dataset, production system?
  │         └─ Use GINI — faster training, similar accuracy
  │
  ├── Features are continuous/numerical?
  │         └─ Both Entropy and Gini handle this automatically
  │            via the threshold scanning mechanism
  │
  └── Using sklearn?
            └─ Default is GINI — good for most cases
               Switch to Entropy only if you want to experiment
```

```python
# sklearn usage:
from sklearn.tree import DecisionTreeClassifier

# Default (Gini)
model = DecisionTreeClassifier(criterion='gini')

# Entropy / ID3-style
model = DecisionTreeClassifier(criterion='entropy')
```

---

---

# PART 6: Complete Decision Tree Training Flow

---

```
RAW TRAINING DATA
        │
        ▼
┌───────────────────────────────────┐
│          ROOT NODE                │
│  Calculate H(S) or G.I. of data   │
└───────────────────────────────────┘
        │
        ▼
For EACH feature:
  ├── If CATEGORICAL → compute IG directly for each category
  └── If NUMERICAL   → sort → scan thresholds → find best IG
        │
        ▼
SELECT feature with MAXIMUM Information Gain
        │
        ▼
SPLIT data into child branches on that feature
        │
        ├──────────────────┬──────────────────┐
        ▼                  ▼                  ▼
   Child Node 1        Child Node 2       Child Node 3
   Repeat above        Repeat above       Repeat above
   for remaining       for remaining      for remaining
   features            features           features
        │
        ▼
STOPPING CONDITIONS (any one met):
  ✓ Node is pure (Entropy = 0 / G.I. = 0)
  ✓ Max depth reached
  ✓ Minimum samples per node threshold hit
  ✓ No more features left to split on
        │
        ▼
LEAF NODES → Final class predictions
```

---

# Quick Reference — Key Formulas

| Metric | Formula |
|---|---|
| **Entropy (binary)** | $H(S) = -P(+)\log_2 P(+) - P(-)\log_2 P(-)$ |
| **Entropy (multi-class)** | $H(S) = -\sum_{i=1}^{c} P_i \log_2 P_i$ |
| **Information Gain** | $\text{Gain}(S,F) = H(S) - \sum_v \frac{\|S_v\|}{\|S\|} H(S_v)$ |
| **Gini Impurity** | $\text{G.I.} = 1 - \sum_{i=1}^{c}(P_i)^2$ |
| **Gini (binary)** | $\text{G.I.} = 1 - [(P(+))^2 + (P(-))^2]$ |

---

# Interview & Exam Traps

| Trap | Correct Answer |
|---|---|
| "Entropy and Gini give very different results" | In practice they produce **nearly identical trees** — the difference is speed, not accuracy |
| "Gini range is 0 to 1 like Entropy" | **No** — Gini binary range is **0 to 0.5**, not 0 to 1 |
| "Higher IG is bad" | **No** — Higher IG = better split = always preferred |
| "sklearn uses Entropy by default" | **No** — sklearn default is **Gini** (`criterion='gini'`) |
| "Numerical features can't be used in Decision Trees" | **Yes they can** — via the threshold scanning mechanism |
| "Decision Trees are always fast to train" | **No** — large numerical datasets make training very slow due to threshold scanning overhead |
| "IG = 0 means great split" | **No** — IG = 0 means the feature provides **zero useful information** |

---

> **Core philosophy of Decision Trees:** At every node, always ask — *"Which feature, split at which point, will make the resulting groups as pure as possible?"*
> Entropy, Gini, and Information Gain are simply three mathematical tools that answer that one question.