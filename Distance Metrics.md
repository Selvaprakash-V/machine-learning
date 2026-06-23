# Distance Metrics — Complete Enhanced Notes
### Topics: Euclidean Distance · Manhattan Distance · Applications · Comparisons

---

## 1. What Are Distance Metrics?

Distance metrics are **fundamental mathematical tools** used in Machine Learning to measure how **alike or different** two data points are from each other.

> **Core idea:** The smaller the distance between two points, the more similar they are. The larger the distance, the more different they are.

### Where Distance Metrics Are Used

| Algorithm | Distance Metric Used | Purpose |
|---|---|---|
| **K-Nearest Neighbors (KNN)** | Euclidean / Manhattan | Find the K closest training points |
| **K-Means Clustering** | Euclidean | Assign points to nearest centroid |
| **DBSCAN** | Euclidean | Define ε-neighborhood radius |
| **Hierarchical Clustering** | Euclidean / Manhattan | Build proximity matrix |
| **Support Vector Machines** | Euclidean | Margin maximization |

---

```
WHY DISTANCE MATTERS IN ML:

Training Data:                   New Point (?) needs classification

  ★ (Class A)                         ★ (Class A)    ★ (Class A)
  ★ (Class A)                                  ?
  ● (Class B)                         ● (Class B)

  KNN asks: which training points
  are CLOSEST to the new point?

  → Measure distances → assign class
    of the K nearest neighbors
```

---

---

# PART 1: Euclidean Distance

---

## 1. Core Concept

Euclidean distance is the **shortest straight-line distance** between two points — like measuring with a ruler in a straight line, ignoring all obstacles.

```
EUCLIDEAN DISTANCE VISUAL (2D):

  Y
  │
5 ┤              B (4, 5)
  │             /|
4 ┤            / |
  │           /  |  ← vertical difference = 5-2 = 3
3 ┤          /   |
  │         /    |
2 ┤  A(1,2)/     |
  │        │─────┘
  │        ← horizontal difference = 4-1 = 3
  └─────────────────── X
    1  2  3  4  5

d = √(3² + 3²) = √(9 + 9) = √18 ≈ 4.24
    ↑
Shortest possible path = straight line
```

---

## 2. Derivation from Pythagorean Theorem

Euclidean distance is a **direct generalization** of the Pythagorean theorem.

```
Right-angled triangle:

     C
     │\
     │  \  ← AC = hypotenuse (Euclidean distance)
  BC │    \
     │      \
     └────────  A
         AB

Pythagorean Theorem:
  AC² = AB² + BC²
  AC  = √(AB² + BC²)

In coordinate terms:
  AB = horizontal difference = (x₂ - x₁)
  BC = vertical difference   = (y₂ - y₁)
  AC = distance              = √((x₂-x₁)² + (y₂-y₁)²)
```

---

## 3. Formulas Across All Dimensions

### 1D — Single Feature

$$d = \sqrt{(x_2 - x_1)^2} = |x_2 - x_1|$$

```
Example: x₁ = 2,  x₂ = 7

  ←──────────────────→
  2    3    4    5    6    7
  x₁                      x₂

  d = |7 - 2| = 5
```

---

### 2D — Two Features

$$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

```
Example: A = (1, 2),  B = (4, 6)

  Y
  │
6 ┤        B(4,6)
  │       /|
  │      / | ← (6-2) = 4
  │     /  |
2 ┤  A(1,2)│
  │    └───┘
  │    (4-1)=3
  └────────────── X

  d = √(3² + 4²) = √(9 + 16) = √25 = 5.0
```

---

### 3D — Three Features

$$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2}$$

```
Example: P1 = (1, 2, 3),  P2 = (4, 6, 3)

  Z
  │   P2 (4,6,3)
  │  /
  │ /  ← 3D straight-line distance
  │/
  └──────── Y
  /
 / P1 (1,2,3)
X

  d = √((4-1)² + (6-2)² + (3-3)²)
    = √(9 + 16 + 0)
    = √25 = 5.0
```

---

### n-Dimensions — General Formula

$$d = \sqrt{\sum_{i=1}^{n}(q_i - p_i)^2}$$

```
For a dataset with n features:

Point P = (p₁, p₂, p₃, ..., pₙ)
Point Q = (q₁, q₂, q₃, ..., qₙ)

d = √[(q₁-p₁)² + (q₂-p₂)² + (q₃-p₃)² + ... + (qₙ-pₙ)²]

Each feature contributes its squared difference.
The sum gets square-rooted at the end.
Works for 2D, 3D, 100D, 1000D — same formula.
```

---

## 4. Euclidean Distance — Properties

```
PROPERTY 1: Non-negative
  d(A, B) ≥ 0 always
  d(A, A) = 0 (distance to itself)

PROPERTY 2: Symmetric
  d(A, B) = d(B, A)

PROPERTY 3: Triangle Inequality
  d(A, C) ≤ d(A, B) + d(B, C)

  A ────────────────────── C
    ↑ direct distance
  A ───── B ───────────── C
    ↑ detour is always ≥ direct
```

---

## 5. Euclidean Distance — Sensitivity to Scale

```
PROBLEM: Features with large values dominate distance

Feature 1: Age       = [25, 30]  → difference = 5
Feature 2: Salary    = [50000, 80000] → difference = 30000

d = √(5² + 30000²) ≈ 30000

Salary COMPLETELY dominates!
Age contributes almost nothing.

SOLUTION: Feature Scaling (StandardScaler or MinMaxScaler)
BEFORE computing any distance metric.
```

---

## 6. Python Implementation

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean

# ─────────────────────────────────────────────
# Manual Euclidean Distance
# ─────────────────────────────────────────────
def euclidean_distance(p, q):
    """
    Compute Euclidean distance between two n-dimensional points.
    p, q: array-like of shape (n,)
    """
    return np.sqrt(np.sum((np.array(q) - np.array(p))**2))

# 2D Example
A = [1, 2]
B = [4, 6]
d = euclidean_distance(A, B)
print(f"2D Euclidean Distance: {d:.4f}")   # Output: 5.0000

# 3D Example
P1 = [1, 2, 3]
P2 = [4, 6, 3]
d3 = euclidean_distance(P1, P2)
print(f"3D Euclidean Distance: {d3:.4f}")  # Output: 5.0000

# nD Example
P = [1, 2, 3, 4, 5]
Q = [6, 7, 8, 9, 10]
dn = euclidean_distance(P, Q)
print(f"5D Euclidean Distance: {dn:.4f}")

# ─────────────────────────────────────────────
# Using scipy (recommended for production)
# ─────────────────────────────────────────────
d_scipy = euclidean([1, 2], [4, 6])
print(f"Scipy Euclidean:       {d_scipy:.4f}")

# ─────────────────────────────────────────────
# Distance Matrix for multiple points
# ─────────────────────────────────────────────
from sklearn.metrics import pairwise_distances

points = np.array([[1, 2],
                   [4, 6],
                   [7, 8],
                   [2, 3]])

dist_matrix = pairwise_distances(points, metric='euclidean')
print("\nEuclidean Distance Matrix:")
print(dist_matrix.round(2))
```

---

---

# PART 2: Manhattan Distance

---

## 1. Core Concept

Manhattan distance calculates the total distance by travelling **only along grid axes** — no diagonal shortcuts allowed. Like navigating city blocks where you can only go up/down or left/right.

```
MANHATTAN vs EUCLIDEAN VISUAL (2D):

  Y
  │
6 ┤    B (2, 6)
  │    │          ← Manhattan path 1: go right then up
  │    │ ↑↑↑↑↑
  │    │         (always follows grid lines)
2 ┤    A (2, 2)
  │    └──────────── or go up then right — same total
  └────────────── X

Euclidean (straight line):
  d = √((2-2)² + (6-2)²) = 4.0

Manhattan (grid path):
  d = |2-2| + |6-2| = 0 + 4 = 4.0

  Both same here because no horizontal change.

─────────────────────────────────────────────

More interesting example:
  A = (1, 2),  B = (4, 6)

  Y
  │
6 ┤        B(4,6)
  │        │
  │        │  ↑ Manhattan goes: right 3, up 4
  │        │
2 ┤  A─────┘  (or up 4 then right 3 — same total)
  │    →→→
  └─────────────── X

  Manhattan = |4-1| + |6-2| = 3 + 4 = 7
  Euclidean = √(3² + 4²)    = 5.0 (shorter!)
```

> **Key Insight:** Euclidean distance is ALWAYS ≤ Manhattan distance. The straight line is always the shortest path.

---

## 2. The Manhattan Metaphor

```
BIRD'S EYE VIEW OF MANHATTAN:

  ┌───┬───┬───┬───┬───┐
  │   │   │   │   │   │
  ├───┼───┼───┼───┼───┤
  │   │   │   │[B]│   │  ← destination
  ├───┼───┼───┼───┼───┤
  │   │   │   │   │   │
  ├───┼───┼───┼───┼───┤
  │[A]│   │   │   │   │  ← start
  └───┴───┴───┴───┴───┘

Taxi can ONLY drive along streets:
→ right 3 blocks + up 2 blocks = 5 total

Taxi CANNOT cut diagonally through buildings!
That's Manhattan distance.
```

---

## 3. Formulas Across All Dimensions

### 1D — Single Feature

$$d = |x_2 - x_1|$$

```
Example: x₁ = 3,  x₂ = 8

  ←────────────────→
  3    4    5    6    7    8
  x₁                      x₂

  d = |8 - 3| = 5
```

---

### 2D — Two Features

$$d = |x_2 - x_1| + |y_2 - y_1|$$

```
Example: A = (1, 2),  B = (4, 6)

  Y
  │
6 ┤    ────────B(4,6)
  │            │
  │            │ ← |6-2| = 4 (vertical)
  │            │
2 ┤  A(1,2)────┘
  │    └───────┘
  │     |4-1| = 3 (horizontal)
  └─────────────────── X

  d = 3 + 4 = 7

  Multiple valid paths — all have SAME total length:

  Path 1: →→→↑↑↑↑    (right 3, up 4)
  Path 2: ↑↑↑↑→→→    (up 4, right 3)
  Path 3: →↑→↑→↑↑    (any combination)

  All equal 7 units. Manhattan doesn't care about path!
```

---

### n-Dimensions — General Formula

$$d = \sum_{i=1}^{n}|q_i - p_i|$$

```
For a dataset with n features:

Point P = (p₁, p₂, p₃, ..., pₙ)
Point Q = (q₁, q₂, q₃, ..., qₙ)

d = |q₁-p₁| + |q₂-p₂| + |q₃-p₃| + ... + |qₙ-pₙ|

Each feature contributes its ABSOLUTE difference.
All contributions are summed (no square root).
```

---

## 4. Why Absolute Value?

```
WITHOUT absolute value:
  Point A = (1, 5),  Point B = (4, 2)

  x diff = 4 - 1 =  3
  y diff = 2 - 5 = -3

  Sum = 3 + (-3) = 0  ← WRONG! Points are NOT at same location

WITH absolute value:
  |x diff| = |4 - 1| = 3
  |y diff| = |2 - 5| = 3

  Sum = 3 + 3 = 6  ← CORRECT distance

Absolute value ensures all differences are POSITIVE
so they can't cancel each other out.
```

---

## 5. Manhattan Distance — Advantage in High Dimensions

```
CURSE OF DIMENSIONALITY:

In high-dimensional spaces (100+ features),
Euclidean distance can become unreliable.
All points start to look "equidistant."

Manhattan distance is often MORE ROBUST
in high-dimensional settings because:

  - Uses absolute differences (not squares)
  - Large differences in ONE feature don't
    disproportionately dominate the total
  - More resistant to outlier features
    that have extreme values in one dimension

Example — 100D space:
  Feature 1: enormous difference = 1000
  Features 2-100: tiny differences ≈ 0.1 each

  Euclidean: 1000² dominates everything → misleading
  Manhattan: 1000 + 9.9 = 1009.9 → feature 1 important
                                    but doesn't obliterate rest
```

---

## 6. Python Implementation

```python
import numpy as np
from scipy.spatial.distance import cityblock
from sklearn.metrics import pairwise_distances

# ─────────────────────────────────────────────
# Manual Manhattan Distance
# ─────────────────────────────────────────────
def manhattan_distance(p, q):
    """
    Compute Manhattan distance between two n-dimensional points.
    p, q: array-like of shape (n,)
    """
    return np.sum(np.abs(np.array(q) - np.array(p)))

# 2D Example
A = [1, 2]
B = [4, 6]
d = manhattan_distance(A, B)
print(f"2D Manhattan Distance: {d:.4f}")   # Output: 7.0

# 3D Example
P1 = [1, 2, 3]
P2 = [4, 6, 1]
d3 = manhattan_distance(P1, P2)
print(f"3D Manhattan Distance: {d3:.4f}")  # Output: 9.0

# nD Example
P = [1, 2, 3, 4, 5]
Q = [6, 7, 8, 9, 10]
dn = manhattan_distance(P, Q)
print(f"5D Manhattan Distance: {dn:.4f}")  # Output: 25.0

# ─────────────────────────────────────────────
# Using scipy (recommended)
# ─────────────────────────────────────────────
d_scipy = cityblock([1, 2], [4, 6])
print(f"Scipy Manhattan:       {d_scipy:.4f}")

# ─────────────────────────────────────────────
# Distance Matrix for multiple points
# ─────────────────────────────────────────────
points = np.array([[1, 2],
                   [4, 6],
                   [7, 8],
                   [2, 3]])

dist_matrix = pairwise_distances(points, metric='manhattan')
print("\nManhattan Distance Matrix:")
print(dist_matrix.round(2))

# ─────────────────────────────────────────────
# KNN with Manhattan distance
# ─────────────────────────────────────────────
from sklearn.neighbors import KNeighborsClassifier

knn_euclidean = KNeighborsClassifier(n_neighbors=5,
                                      metric='euclidean')
knn_manhattan = KNeighborsClassifier(n_neighbors=5,
                                      metric='manhattan')

knn_euclidean.fit(X_train, y_train)
knn_manhattan.fit(X_train, y_train)

print(f"KNN Euclidean Accuracy: {knn_euclidean.score(X_test, y_test):.4f}")
print(f"KNN Manhattan Accuracy: {knn_manhattan.score(X_test, y_test):.4f}")
```

---

---

# PART 3: Euclidean vs Manhattan — Deep Comparison

---

## 1. Geometric Comparison Visual

```
SAME TWO POINTS: A=(1,2) and B=(4,6)

  Y
  │
6 ┤        B(4,6)
  │       /│
  │      / │
  │     /  │
  │    /   │  Manhattan paths:
  │   /    │  ─── ─── ─── (any grid path = 7)
  │  / ←Eu │
  │ /  clid│
  │/  =5.0 │
2 ┤ A(1,2)─┘
  │
  └─────────────── X

  Euclidean = 5.0  (straight diagonal line)
  Manhattan = 7.0  (sum of axis-aligned steps)

  Euclidean ≤ Manhattan  ← ALWAYS TRUE
```

---

## 2. Multiple Manhattan Paths — Same Total

```
A = (0,0)  to  B = (3,3)

  (0,3)─────(3,3)=B        Path options (all = 6 total):
    │
    │  ↑↑↑→→→  = 6         →→→↑↑↑    (right then up)
    │  →↑→↑→↑  = 6         ↑↑↑→→→    (up then right)
    │  →→↑↑→↑  = 6         →↑→↑→↑    (alternating)
    │
  (0,0)=A                  ALL paths cost exactly 6
```

---

## 3. Side-by-Side Formula Comparison

| Dimension | Euclidean | Manhattan |
|---|---|---|
| **1D** | $\|x_2 - x_1\|$ | $\|x_2 - x_1\|$ |
| **2D** | $\sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$ | $\|x_2-x_1\| + \|y_2-y_1\|$ |
| **3D** | $\sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2}$ | $\|\Delta x\| + \|\Delta y\| + \|\Delta z\|$ |
| **nD** | $\sqrt{\sum(q_i - p_i)^2}$ | $\sum\|q_i - p_i\|$ |

---

## 4. Comprehensive Comparison Table

| Property | Euclidean | Manhattan |
|---|---|---|
| **Movement type** | Straight diagonal line | Grid-aligned steps only |
| **Math operation** | Square + Square root | Absolute difference + Sum |
| **Sensitivity to outliers** | High (squares amplify) | Lower (linear effect) |
| **High-dimensional data** | Can be unreliable | More robust |
| **Computation speed** | Slightly slower | Faster (no square root) |
| **Geometric shape** | Circle (equal-distance zone) | Diamond (rotated square) |
| **Best for** | Continuous, low-dim data | High-dim, grid-like data |
| **Common alias** | L2 distance | L1 distance / City Block |

---

## 5. Equal Distance Shapes (Unit Balls)

```
EUCLIDEAN (L2):              MANHATTAN (L1):

All points at distance        All points at distance
d = 1 form a CIRCLE:         d = 1 form a DIAMOND:

      *  *  *                      *
    *        *                   *   *
   *          *                *       *
    *        *                   *   *
      *  *  *                      *

  Circular boundary             Diamond boundary
  (rotationally symmetric)      (axis-aligned symmetric)
```

---

## 6. Worked Numerical Example

**Two data points:**
- P = (2, 4, 1, 6)
- Q = (5, 1, 3, 2)

### Euclidean Calculation

```
Step 1: Compute squared differences
  (5-2)² = 3² = 9
  (1-4)² = (-3)² = 9
  (3-1)² = 2² = 4
  (2-6)² = (-4)² = 16

Step 2: Sum them
  9 + 9 + 4 + 16 = 38

Step 3: Square root
  d = √38 ≈ 6.164
```

### Manhattan Calculation

```
Step 1: Compute absolute differences
  |5-2| = 3
  |1-4| = 3
  |3-1| = 2
  |2-6| = 4

Step 2: Sum them
  d = 3 + 3 + 2 + 4 = 12
```

```
Comparison:
  Euclidean: 6.164 (shorter, straight-line)
  Manhattan: 12.0  (longer, grid-path)

  Ratio = 12.0 / 6.164 ≈ 1.95
  Manhattan is ~2x the Euclidean in this case.
```

---

## 7. When to Use Which

```
CHOOSING YOUR DISTANCE METRIC
──────────────────────────────

START
  │
  ├── Is your data low-dimensional (< 20 features)?
  │         └─ YES → Euclidean distance (L2)
  │                  Works well in compact spaces
  │
  ├── Is your data high-dimensional (20+ features)?
  │         └─ YES → Manhattan distance (L1)
  │                  More robust to curse of dimensionality
  │
  ├── Does your data have many outlier features?
  │         └─ YES → Manhattan distance (L1)
  │                  Absolute values don't amplify outliers
  │
  ├── Is your data on a grid or coordinate system?
  │         └─ YES → Manhattan distance
  │                  Naturally represents grid movement
  │
  ├── Are all features equally scaled/normalized?
  │         └─ YES → Either works well
  │                  Euclidean is the default choice
  │
  └── Are features on very different scales?
            └─ Scale FIRST with StandardScaler
               THEN apply either metric
```

---

## 8. Impact of Feature Scaling on Distances

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# Raw (unscaled) data
P_raw = np.array([[25,  50000]])   # Age=25, Salary=50,000
Q_raw = np.array([[30,  80000]])   # Age=30, Salary=80,000

# Euclidean on raw data
d_raw = pairwise_distances(P_raw, Q_raw, metric='euclidean')
print(f"Euclidean (raw):    {d_raw[0][0]:.2f}")
# Output: 30000.00 (salary completely dominates!)

# Standardize first
data = np.array([[25, 50000], [30, 80000]])
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

P_scaled = data_scaled[[0]]
Q_scaled = data_scaled[[1]]

d_scaled = pairwise_distances(P_scaled, Q_scaled, metric='euclidean')
print(f"Euclidean (scaled): {d_scaled[0][0]:.4f}")
# Output: balanced contribution from both features

# Always scale before computing distances!
```

---

---

# PART 4: Beyond L1 and L2 — The Minkowski Family

---

## 1. Minkowski Distance — The General Formula

Both Euclidean and Manhattan are **special cases** of the more general **Minkowski distance**:

$$d = \left(\sum_{i=1}^{n}|q_i - p_i|^p\right)^{\frac{1}{p}}$$

| Value of p | Distance Metric | Formula Reduces To |
|---|---|---|
| **p = 1** | Manhattan (L1) | $\sum\|q_i - p_i\|$ |
| **p = 2** | Euclidean (L2) | $\sqrt{\sum(q_i-p_i)^2}$ |
| **p = ∞** | Chebyshev | $\max\|q_i - p_i\|$ |

```
MINKOWSKI FAMILY VISUALIZATION:
(Equal-distance boundaries for different p values)

p = 1 (Manhattan):    p = 2 (Euclidean):    p = ∞ (Chebyshev):

       *                    * * *                 * * * * *
      * *                  *     *               *         *
     *   *                *       *              *         *
      * *                  *     *               *         *
       *                    * * *                 * * * * *

   Diamond shape          Circle shape          Square shape
```

---

## 2. Python — Minkowski with Different p Values

```python
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

P = np.array([[1, 2]])
Q = np.array([[4, 6]])

for p in [1, 2, 3, 5, 10, np.inf]:
    if p == np.inf:
        d = np.max(np.abs(Q - P))
        print(f"p = ∞ (Chebyshev):   {d:.4f}")
    else:
        d = np.sum(np.abs(Q - P)**p)**(1/p)
        name = {1: "Manhattan", 2: "Euclidean"}.get(p, f"Minkowski")
        print(f"p = {p} ({name}): {d:.4f}")

# Output:
# p = 1 (Manhattan):  7.0000
# p = 2 (Euclidean):  5.0000
# p = 3 (Minkowski):  4.4979
# p = 5 (Minkowski):  4.1753
# p = ∞ (Chebyshev):  4.0000
```

---

---

# PART 5: Complete Implementation Pipeline

---

## Distance Metrics in KNN — Full Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────────
# STEP 1: Load and prepare data
# ─────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# STEP 2: Scale features (CRITICAL for distances)
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# STEP 3: Compare metrics across K values
# ─────────────────────────────────────────────
metrics   = ['euclidean', 'manhattan', 'minkowski']
k_values  = range(1, 21)
results   = {m: [] for m in metrics}

for metric in metrics:
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test_scaled))
        results[metric].append(acc)

# ─────────────────────────────────────────────
# STEP 4: Plot comparison
# ─────────────────────────────────────────────
plt.figure(figsize=(12, 5))

for metric, scores in results.items():
    plt.plot(k_values, scores, marker='o',
             linewidth=2, label=metric.capitalize())

plt.title('KNN Accuracy — Euclidean vs Manhattan vs Minkowski')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.xticks(k_values)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# STEP 5: Find best metric + K combination
# ─────────────────────────────────────────────
best_metric = max(results, key=lambda m: max(results[m]))
best_k      = k_values[results[best_metric].index(max(results[best_metric]))]
best_score  = max(results[best_metric])

print(f"\nBest Metric: {best_metric}")
print(f"Best K:      {best_k}")
print(f"Best Score:  {best_score:.4f}")
```

---

---

# Quick Reference — All Key Formulas

| Metric | 2D Formula | nD Formula | Alias |
|---|---|---|---|
| **Euclidean** | $\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}$ | $\sqrt{\sum(q_i-p_i)^2}$ | L2 Distance |
| **Manhattan** | $\|x_2-x_1\|+\|y_2-y_1\|$ | $\sum\|q_i-p_i\|$ | L1 / City Block |
| **Minkowski** | General case | $(\sum\|q_i-p_i\|^p)^{1/p}$ | Lp Distance |
| **Chebyshev** | $\max(\|\Delta x\|, \|\Delta y\|)$ | $\max\|q_i-p_i\|$ | L∞ Distance |

---

# Interview & Exam Traps

| Trap | Correct Answer |
|---|---|
| "Manhattan is always larger than Euclidean" | **Yes** — Euclidean ≤ Manhattan always holds (straight line is shortest) |
| "Distance metrics don't need scaling" | **Wrong** — always scale features before computing distances to prevent dominant features |
| "Manhattan and Euclidean give the same result in 1D" | **Yes** — in 1D both reduce to $\|x_2 - x_1\|$ |
| "Euclidean is always the best choice for KNN" | Not always — Manhattan can outperform in high dimensions or with outliers |
| "Manhattan distance has only one valid path" | **No** — infinitely many valid grid paths exist, all with the same total length |
| "Euclidean is L1, Manhattan is L2" | **Wrong** — Euclidean = **L2**, Manhattan = **L1** |
| "You don't need to scale before clustering" | **Wrong** — K-Means and KNN both use distance; unscaled features distort results |

---

> **Core philosophy:** Distance metrics are the backbone of similarity-based ML algorithms. Choosing the right metric and scaling your features properly can be the difference between a great model and a misleading one. When in doubt — scale first, then experiment with both Euclidean and Manhattan to see which performs better on your specific dataset.