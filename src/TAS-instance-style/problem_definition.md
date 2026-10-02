# Instance-Level Temporal Action Segmentation

## 1. Problem Definition

The goal is to segment a video into a sequence of **action instances**, where each instance has:

- a temporal start point
- a temporal end point
- an action/operation class

Formally, given a video:

$$
V = \{x_1, x_2, ..., x_T\}
$$

the model predicts a set of temporal action instances:

$$
\hat{Y} = \{(\hat{s}_i, \hat{e}_i, \hat{c}_i)\}_{i=1}^{N}
$$

where:

- $\hat{s}_i$: start time/frame of instance $i$
- $\hat{e}_i$: end time/frame of instance $i$
- $\hat{c}_i$: action/operation class of instance $i$
- $N$: number of detected action instances

The key requirement is that **different temporal instances must remain separate even when they belong to the same action class**.

---

## 2. Key Difference from Standard Temporal Action Segmentation

In conventional Temporal Action Segmentation (TAS), the model usually predicts a class for every frame:

$$
\hat{c}_t \in \{1,\ldots,C\}
$$

For example:

```text
Time:     0---------1-------2-------3----------4

Class:    AAAAAAAAAAAAAAAAA BBBBBBBB CCCCCCCCC
          |--- A ---|--- A ---|--- B ---|--- C ---|
```

If two consecutive segments have the same class, a conventional frame-wise representation may simply produce:

```text
AAAAAAAAAAAAAAAAA BBBBBBBB CCCCCCCC
```

and lose the fact that there are actually **two separate occurrences of action A**.

### Our task must preserve this distinction.

The desired output is:

```text
Instance 1: (0, 1, A)
Instance 2: (1, 2, A)
Instance 3: (2, 3, B)
Instance 4: (3, 4, C)
```

The two `A` instances must **not be merged**.

---

## 3. Example

Consider a worker performing operations on a sewing machine.

The video may contain:

```text
0---------1-------2-------3----------4
|         |       |       |          |
| Sew #1  | Sew#2 |Adjust | Sew #3   |
|         |       |       |          |
```

where:

```text
Sew #1      = Sewing
Sew #2      = Sewing
Adjust      = Fabric Adjustment
Sew #3      = Sewing
```

The desired prediction is:

```text
[
    (0.0, 1.0, Sewing),
    (1.0, 2.0, Sewing),
    (2.0, 3.0, Fabric_Adjustment),
    (3.0, 4.0, Sewing)
]
```

The two first `Sewing` segments must **not be merged**.

---

## 4. Core Requirements

The model must solve two coupled problems.

### 4.1 Temporal Boundary Detection

Detect the temporal boundaries between action instances:

$$
B = \{b_1, b_2, ..., b_K\}
$$

A boundary may occur even when the action class does not change.

For example:

```text
Sewing #1       Sewing #2
     A              A
     |              |
-----|--------------|-----
     ↑
  boundary
```

Therefore, boundary detection should answer:

> "Did one action instance end and another action instance begin?"

rather than simply:

> "Did the action class change?"

---

### 4.2 Action Instance Classification

For every detected temporal instance, predict its action class:

$$
c_i = f(x_{s_i:e_i})
$$

For example:

```text
Instance       Temporal Range       Class
------------------------------------------------
Instance 1     0 → 1                Sewing
Instance 2     1 → 2                Sewing
Instance 3     2 → 3                Adjustment
Instance 4     3 → 4                Sewing
```

Two instances can therefore have:

$$
c_i = c_j
$$

while:

$$
(s_i,e_i) \neq (s_j,e_j)
$$

---

# 5. Formal Task Definition

Given a video $V$ consisting of $T$ frames, predict an ordered set of action instances:

$$
Y = \{(s_i,e_i,c_i)\}_{i=1}^{N}
$$

subject to:

$$
0 \leq s_i < e_i \leq T
$$

and:

$$
s_{i+1} \geq e_i
$$

Each action instance is independently represented by:

1. **Temporal extent**
   - start: $s_i$
   - end: $e_i$

2. **Action identity**
   - class: $c_i$

The important property is:

$$
c_i = c_{i+1}
$$

is allowed.

Therefore, the task is **not equivalent to ordinary frame-wise classification**.

---

# 6. Desired Model Output

Instead of producing only:

```text
Frame → Action Class
```

the model should produce:

```text
Action Query / Instance
        ↓
Temporal Mask
        ↓
Action Class
```

For example:

```text
Query 1
├── Temporal mask: [111111000000000]
└── Class: Sewing

Query 2
├── Temporal mask: [000000111110000]
└── Class: Sewing

Query 3
├── Temporal mask: [000000000001111]
└── Class: Adjustment
```

This representation naturally allows:

```text
Query 1 = Sewing
Query 2 = Sewing
```

while keeping them as **different temporal instances**.

---

# 7. Possible Mask Representation

For a video with $T$ temporal units, each query predicts:

$$
M_i \in [0,1]^T
$$

where:

$$
M_i(t)
$$

indicates whether temporal position $t$ belongs to action instance $i$.

For example:

```text
Temporal position:

0 1 2 3 4 5 6 7 8 9 10 11

Query 1:
1 1 1 1 0 0 0 0 0 0  0  0
└─────── Instance A #1

Query 2:
0 0 0 0 1 1 1 0 0 0  0  0
        └─── Instance A #2

Query 3:
0 0 0 0 0 0 0 1 1 1  0  0
                └── Instance B
```

This makes the distinction between:

```text
A #1
A #2
```

explicit.

---

# 8. Boundary Definition

A boundary should represent a transition between **action instances**, not necessarily a transition between classes.

Therefore there are two types of boundaries:

### Type 1 — Different-Class Boundary

```text
A → B
```

Example:

```text
Sewing → Adjustment
```

### Type 2 — Same-Class Boundary

```text
A → A
```

Example:

```text
Sewing #1 → Sewing #2
```

The second type is particularly important for this task.

---

# 9. Why This Is Different from GEBD

Generic Event Boundary Detection (GEBD) asks:

> "Where does one generic event end and another event begin?"

It does not necessarily require assigning an action class to each resulting segment.

Our task requires both:

```text
Boundary Detection
        +
Action Classification
        +
Instance Separation
```

Therefore:

```text
GEBD
  ↓
Boundary
```

while our task is:

```text
Video
 ↓
Temporal Instances
 ↓
┌──────────────┬──────────────┬──────────────┐
│ Start        │ End          │ Action Class │
└──────────────┴──────────────┴──────────────┘
```

---

# 10. Why This Is Different from Conventional TAS

Conventional TAS:

```text
Video
 ↓
Frame-wise predictions
 ↓
[A A A A A B B B C C C]
```

Our task:

```text
Video
 ↓
Instance-level predictions
 ↓
[
  (A, start1, end1),
  (A, start2, end2),
  (B, start3, end3),
  (C, start4, end4)
]
```

The critical difference is:

```text
Conventional TAS:

A A A A A A
└──────────┘
   one segment


Our task:

A A | A A
  ↑
boundary

└───┘ └───┘
 A#1    A#2
```

Thus, **same-class consecutive instances must remain separate**.

---

# 11. Evaluation

Evaluation should measure both temporal localization and action classification.

Potential metrics include:

### Boundary Detection

- Boundary Precision
- Boundary Recall
- Boundary F1

### Action Classification

- Instance classification accuracy
- Macro F1
- Per-class F1

### Temporal Instance Detection

Evaluate predicted:

$$
(\hat{s}_i,\hat{e}_i,\hat{c}_i)
$$

against ground-truth:

$$
(s_j,e_j,c_j)
$$

using temporal IoU:

$$
tIoU =
\frac{
\min(e_i,e_j)-\max(s_i,s_j)
}{
\max(e_i,e_j)-\min(s_i,s_j)
}
$$

A prediction can be considered a correct instance when:

1. temporal IoU is above a threshold;
2. predicted class is correct;
3. the instance is correctly separated from neighboring instances.

This last condition is particularly important for:

```text
A #1 | A #2
```

because merging them into:

```text
A #1+#2
```

should count as an error.

---

# 12. Possible Architecture Direction

A natural architecture for this task is a **query-based temporal instance segmentation model**.

```text
                    Video
                      │
                      ▼
              Video Backbone
                      │
                      ▼
             Temporal Features
                      │
                      ▼
          Temporal Transformer
                      │
                      ▼
                Action Queries
                      │
              ┌───────┴───────┐
              ▼               ▼
       Temporal Mask      Class Head
              │               │
              ▼               ▼
       Action Instance + Action Class
```

A Mask2Former-inspired version could be:

```text
Video
  ↓
Temporal Feature Encoder
  ↓
Action Queries
  ↓
Temporal Cross-Attention
  ↓
Temporal Mask Prediction
  ↓
Masked Temporal Cross-Attention
  ↓
Refined Temporal Mask
  ↓
┌──────────────────────────┐
│ Temporal Mask + Class    │
└──────────────────────────┘
```

The spatial mask in Mask2Former becomes a **1D temporal mask**.

---

# 13. Training Objective

The model can be trained as a set prediction problem.

Ground truth:

$$
Y = \{(M_i,c_i)\}_{i=1}^{N}
$$

Predictions:

$$
\hat{Y} = \{(\hat{M}_q,\hat{c}_q)\}_{q=1}^{Q}
$$

Use bipartite matching to associate predicted queries with ground-truth instances.

A possible matching cost is:

$$
\mathcal{L}_{match}
=
\lambda_{cls}\mathcal{L}_{cls}
+
\lambda_{mask}\mathcal{L}_{mask}
+
\lambda_{boundary}\mathcal{L}_{boundary}
$$

Then optimize:

$$
\mathcal{L}
=
\lambda_{cls}\mathcal{L}_{cls}
+
\lambda_{mask}\mathcal{L}_{mask}
+
\lambda_{boundary}\mathcal{L}_{boundary}
+
\lambda_{aux}\mathcal{L}_{aux}
$$

The Hungarian matching is particularly useful because it treats:

```text
A #1
A #2
```

as two different ground-truth instances, even though:

```text
class(A#1) = class(A#2)
```

---

# 14. One-Sentence Definition

A concise definition of the task is:

> **Instance-Level Temporal Action Segmentation is the task of detecting, temporally localizing, and classifying individual action instances in a video, while preserving separate instances even when consecutive instances belong to the same action class.**

---

# 15. Short Name

Possible terminology:

### Recommended

**Instance-Level Temporal Action Segmentation (ITAS)**

or

**Temporal Instance Action Segmentation (TIAS)**

The first is clearer because it emphasizes that the segmentation is performed at the **instance level**.

A more specific formulation for this research could be:

> **Boundary-Aware Instance-Level Temporal Action Segmentation**

because the central challenge is detecting boundaries between individual action occurrences, including **same-class consecutive instances**.

---

# 16. Core Research Question

The central research question can therefore be stated as:

> **How can we detect and classify individual action instances in a video while distinguishing temporal boundaries between consecutive instances of the same action class?**

This formulation separates the problem from:

```text
GEBD
   → only event boundaries

Standard TAS
   → frame-wise action labels

Your task
   → temporal instance boundaries
   + instance-level action classification
   + same-class instance separation
```

This is also why a **Mask2Former/DETR-style query → temporal mask + class** formulation is a particularly natural architectural direction for this problem.

---

# 17. Recommended Dataset Format

For this task, the dataset should use **instance-level annotations** as the primary ground truth rather than only frame-wise class labels.

## 17.1 Dataset Structure

```text
dataset/
├── videos/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
│
├── annotations/
│   ├── video_001.json
│   ├── video_002.json
│   └── ...
│
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
│
└── classes.json
```

Each video has a corresponding annotation file containing its temporal action instances.

---

## 17.2 Primary Annotation Unit

The fundamental annotation unit should be:

$$
(s_i, e_i, c_i)
$$

where:

- $s_i$: start frame/time
- $e_i$: end frame/time
- $c_i$: action class

For example:

```json
{
  "video_id": "video_001",
  "fps": 30,
  "num_frames": 900,
  "duration": 30.0,

  "instances": [
    {
      "id": 0,
      "start_frame": 0,
      "end_frame": 120,
      "start_time": 0.0,
      "end_time": 4.0,
      "class_id": 0,
      "class_name": "sewing"
    },
    {
      "id": 1,
      "start_frame": 121,
      "end_frame": 240,
      "start_time": 4.03,
      "end_time": 8.0,
      "class_id": 0,
      "class_name": "sewing"
    },
    {
      "id": 2,
      "start_frame": 241,
      "end_frame": 330,
      "start_time": 8.03,
      "end_time": 11.0,
      "class_id": 1,
      "class_name": "adjust_fabric"
    },
    {
      "id": 3,
      "start_frame": 331,
      "end_frame": 450,
      "start_time": 11.03,
      "end_time": 15.0,
      "class_id": 0,
      "class_name": "sewing"
    }
  ]
}
```

The critical property is:

```text
Instance 0 → sewing
Instance 1 → sewing
```

They have the same class but remain **two separate instances**.

---

## 17.3 Same-Class Consecutive Instances

The dataset must explicitly support:

```text
A #1 | A #2 | B | C
```

rather than collapsing it into:

```text
A | B | C
```

For example:

```json
{
  "instances": [
    {
      "id": 0,
      "start_frame": 0,
      "end_frame": 100,
      "class_id": 0,
      "class_name": "A"
    },
    {
      "id": 1,
      "start_frame": 101,
      "end_frame": 200,
      "class_id": 0,
      "class_name": "A"
    },
    {
      "id": 2,
      "start_frame": 201,
      "end_frame": 300,
      "class_id": 1,
      "class_name": "B"
    },
    {
      "id": 3,
      "start_frame": 301,
      "end_frame": 400,
      "class_id": 2,
      "class_name": "C"
    }
  ]
}
```

Here:

$$
class(A_1) = class(A_2)
$$

but:

$$
(start_1,end_1) \neq (start_2,end_2)
$$

Therefore they are two independent action instances.

---

## 17.4 Explicit Boundary Annotations

Because boundary detection is a core part of the task, boundaries can also be stored explicitly.

```json
{
  "boundaries": [
    {
      "frame": 100,
      "type": "same_class",
      "left_instance": 0,
      "right_instance": 1
    },
    {
      "frame": 200,
      "type": "class_change",
      "left_instance": 1,
      "right_instance": 2
    }
  ]
}
```

Recommended boundary types:

```text
same_class
class_change
transition
```

### Same-class boundary

```text
A #1 ─────────|──────── A #2
              ↑
         same-class boundary
```

### Class-change boundary

```text
A ────────────|──────── B
              ↑
       class-change boundary
```

This distinction is useful for analyzing whether the model can actually solve the hardest part of the task.

---

## 17.5 Handling Transition / Background Periods

Manufacturing videos may contain periods where the worker is not performing a clearly defined operation:

```text
Sewing #1
    ↓
stop / move hands / reposition fabric
    ↓
Sewing #2
```

There are two possible annotation strategies.

### Strategy A — Explicit transition class

```json
{
  "id": 1,
  "start_frame": 120,
  "end_frame": 135,
  "class_id": -1,
  "class_name": "transition"
}
```

### Strategy B — Boundary without an action instance

The transition is not treated as an action:

```text
Sewing #1
────────────|
            | boundary
            |
       transition
            |
            |────────────
              Sewing #2
```

For manufacturing data, an explicit `transition` / `background` label can be useful if these periods are visually meaningful and consistently annotatable.

---

## 17.6 Do Not Make Frame-Wise Labels the Primary Ground Truth

The primary annotation should be:

```text
Instance-level:
(start, end, class)
```

rather than:

```text
Frame-level:
frame → class
```

Frame-wise labels can be derived from the instance annotations when needed.

This allows the same dataset to support both:

```text
                    Ground Truth
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       Frame-wise TAS        Instance-level model
              │                     │
       MS-TCN / ASFormer       Query / Mask model
       ASRF / etc.             DETR / MaskFormer
```

This is useful for comparing conventional TAS baselines against the proposed instance-level approach.

---

## 17.7 Recommended Final Schema

The recommended per-video annotation schema is:

```text
Video
│
├── metadata
│   ├── video_id
│   ├── fps
│   ├── num_frames
│   └── duration
│
├── instances
│   └── [
│        {
│          instance_id
│          start_frame
│          end_frame
│          start_time
│          end_time
│          class_id
│          class_name
│        }
│      ]
│
└── boundaries
    └── [
         {
           frame
           type
           left_instance
           right_instance
         }
       ]
```

with:

```text
boundary.type ∈ {
    "same_class",
    "class_change",
    "transition"
}
```

The **fundamental ground-truth unit is `(start, end, class)`**.

This preserves the information that conventional frame-wise TAS can lose when two adjacent action instances have the same class.

---

## 17.8 Mapping to the Proposed Query-Based Model

The dataset format maps naturally to a query-based temporal instance segmentation architecture.

```text
GROUND TRUTH
────────────────────────────

Instance #1
(start, end, class)
        │
        ▼
Temporal Mask #1 + Class


Instance #2
(start, end, class)
        │
        ▼
Temporal Mask #2 + Class


Instance #3
(start, end, class)
        │
        ▼
Temporal Mask #3 + Class
```

The model can then use bipartite/Hungarian matching:

```text
Predicted Query #1  ←→  GT Instance #2
Predicted Query #2  ←→  GT Instance #1
Predicted Query #3  ←→  GT Instance #3
```

This treats:

```text
A #1
A #2
```

as two independent ground-truth instances even though:

```text
class(A #1) = class(A #2)
```

and therefore directly supports the central requirement of this task.
