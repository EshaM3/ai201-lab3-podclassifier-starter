# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
 After pairing up each of the predicted labels with ground truth labels, the number of pairs where both labels are the same divided by the total number of ground truth labels gives you the accuracy value.
```

---

**Step-by-step logic:**

```
 1. Set an index variable to 0. Set a correct variable to 0.
 2. Go through both the ground truth labels and predicted labels at the same time, comparing each value at the current index value.
 3. If they are equal, increment the correct variable by 1.
 4. After reaching the end of at least one of the rows, divide the correct variable by the length of ground truth labels (as long as that value is not 0. Otherwise, return 0.0).
```

---

**Edge case — what if both lists are empty?**

```
1.0 if both are empty. If there are no ground truth labels, then there should not be any predicted labels. If both are empty, then the correct labels were generated (none in this case).

If only one is empty, it should be 0.0 - especially if ground_truth is empty as you can not divide by 0.
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

0.5
interview = interview (index 0)
solo = solo (index 1)
panel != solo (index 2)
interview != narrative (index 3)
2 correct values / 4 ground truth values = 0.5

```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
 When the exact string label in the ground_truth list at a certain index value is the same string label value as one in the predictions list at the same index value.
```

---

**What does "total" mean for a given class?**

```
"total" should be the total number of ground_truth values or len(ground_truth), as this is the expected number of labels from which accuracy should be calculated. If this value is 0, however, we should not divide by this value. Instead, return an accuracy value of 0.0.
```

---

**Step-by-step logic:**

```
1. Initialize a result dict: for each label in VALID_LABELS, a sub-dict
   {"correct": 0, "total": 0, "accuracy": 0.0}.
2. Loop over predictions and ground_truth together (same index), e.g. zip.
3. For each pair (predicted, truth):
     - Increment result[truth]["total"] by 1 (this episode belongs to the
       truth class). Skip if truth isn't in VALID_LABELS.
     - If predicted == truth, also increment result[truth]["correct"] by 1.
4. After the loop, for each label compute accuracy = correct / total,
   but only if total > 0; otherwise leave accuracy at 0.0.
5. Return the result dict.

```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
accuracy should be set to 0.0
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

label       correct  total  accuracy
----------  -------  -----  --------
interview     1.0     1.0      1.0
solo          1.0     2.0      0.5
panel         1.0     1.0      1.0
narrative     0.0     1.0      0.0
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes?
