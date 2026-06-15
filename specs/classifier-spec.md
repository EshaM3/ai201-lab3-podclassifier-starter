# Classifier Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 2.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `build_few_shot_prompt()` and
`classify_episode()` in `classifier.py`.

---

## build_few_shot_prompt(labeled_examples, description)

### What it does
Constructs a prompt string for the LLM that includes the task instructions,
all labeled training examples, and the new episode description to classify.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `labeled_examples` | `list[dict]` | Each dict has `"title"`, `"description"`, `"label"` (and others). These are the examples you labeled in Milestone 1. |
| `description` | `str` | The episode description to classify. |

### Output

| Return value | Type | Description |
|---|---|---|
| prompt | `str` | A complete prompt string ready to send to the LLM. |

---

### Spec fields — fill these in before writing code

**Task instruction (what should the LLM know about the task?):**

```
You are classifying podcast episodes by their format. Classify the episode
into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests,
  no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or
  discussing a topic together
- narrative: a story assembled from external sources — interviews, archival
  audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy.
```

---

**How should labeled examples be formatted in the prompt?**

```
Each example should include the episode title, a brief excerpt or the full
description, and the correct label. Separate examples with a blank line or
a delimiter like "---". Include all fields that help the model see why the
label was applied — title and description are both useful; other fields
(like episode ID) are not needed.
```

---

**Example block sketch (write one concrete example):**

```
Title: {title}
Description: {description}
Label: {label}
```

---

**How should the new episode (to be classified) be presented?**

```
Present it in the same format as the labeled examples, but omit the Label
line and replace it with an instruction to classify. For example:

Title: {title}
Description: {description}
Label: ?

Then add a line like: "Classify the episode above. Return your answer in
the format below:" followed by the output format you chose.
```

---

**What output format should you request from the LLM?**

```
Request JSON: {"label": "<one of interview|solo|panel|narrative>",
"reasoning": "<one or two sentences>"}. JSON maps directly to the required
return dict and is robust to colons/commas/newlines in the reasoning that
would break a "Label: X / Reasoning: Y" split. Tradeoff: the model may wrap
it in code fences or add stray text, so parsing must strip fences and use
try/except, falling back to label="unknown" on any failure.
```

---

**Edge cases to handle in the prompt:**

```
If labeled_examples is empty, rather than quietly running a zero-shot prompt, I would raise an actual error so that the user is not classifying the topics in a likely inaccurate way without knowing unless they were looking at the logs.
raise ValueError("labeled_examples is empty")

If the description is empty, return "unknown" as the label for classify_episode() and show the error details in the returned description.

If the description is only up to a few words, still try to pass it and classify it.
```

---

## classify_episode(description, labeled_examples)

### What it does
Classifies a single podcast episode description using the few-shot LLM classifier.
Returns a dict with a label and reasoning.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `description` | `str` | The episode description to classify. |
| `labeled_examples` | `list[dict]` | Labeled training examples from `load_labeled_examples()`. |

### Output

| Return value | Type | Description |
|---|---|---|
| result | `dict` | Must have keys `"label"` and `"reasoning"`. `"label"` must be one of `VALID_LABELS` or `"unknown"`. |

---

### Spec fields — fill these in before writing code

**Step 1 — Build the prompt:**

```
Call build_few_shot_prompt(labeled_examples, description) and store the
returned string in a variable (e.g., prompt). Pass through both arguments
exactly as received — no modification needed before calling.
```

---

**Step 2 — Send to the LLM:**

```
Call _client.chat.completions.create() with:
  - model: the model name from config (MODEL_NAME)
  - messages: a list with one dict — {"role": "user", "content": prompt}
  - max_tokens: a reasonable limit (e.g., 200–300) to keep responses concise

Extract the response text from:
  response.choices[0].message.content
```

---

**Step 3 — Parse the response:**

```
The output format is JSON, but LLMs sometimes wrap it in markdown fences
(```json ... ```) or add a sentence before it. So parse defensively:
strip whitespace and any code fences, then json.loads(). Pull "label" and
"reasoning" out of the parsed dict, and normalize the label with
.strip().lower() before validating it (Step 4). Wrap the whole thing in
try/except so a bad response falls back to "unknown" instead of crashing
(Step 5).

    import json, re

    raw = response.choices[0].message.content or ""
    text = raw.strip()

    # Strip markdown code fences if present, e.g. ```json ... ```
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        parsed = json.loads(text)
        label = str(parsed.get("label", "")).strip().lower()
        reasoning = str(parsed.get("reasoning", "")).strip()
    except (json.JSONDecodeError, AttributeError, TypeError):
        # Unparseable response — see Step 5
        label = "unknown"
        reasoning = f"Could not parse response: {raw!r}"
```

---

**Step 4 — Validate the label:**

```
After normalizing the label, if the label could not be validated (not in VALID_LABELS), send "unknown" as the label and the error details in the reasoning.
```

---

**Step 5 — Handle errors gracefully:**

```
For any of these errors: json.JSONDecodeError, AttributeError, and TypeError, we want to send the "unknown" label with the according error details in the reasoning. We do this by wrapping the parsing process with a try-catch statement.
```

---

### Return value structure

```python
{
    "label": str,      # one of VALID_LABELS, or "unknown" if invalid/error
    "reasoning": str,  # brief explanation from the LLM
}
```

---

## Notes on label quality

The classifier is only as good as your labels. If your training examples have
inconsistent or ambiguous labels, the LLM will learn the wrong pattern.

Before implementing the classifier, re-read `data/taxonomy.md` and double-check
any labels you're unsure about. Annotation quality is part of the lab.

---

## Implementation Notes

*Fill this in after implementing and testing both functions.*

**Test: what does the raw LLM response look like for one episode?**

```
Episode tested: The Case for Four-Day Workweeks
Raw response text: panel; The episode features the host and their friends discussing a topic, and they will also be joined by a relative of Michael Jackson's to share their perspective, indicating a discussion with multiple guests with roughly equal speaking time. This format is typical of a panel discussion.
```

**How did you parse the label out of the response?**

```
The requested output format was in JSON from the LLM. So, aside from removing code fencing with regex and extra whitespace with strip(), I mainly just used `parsed = json.loads(text)`
Then, I would pull the individual label with this line: `str(parsed.get("label", "")).strip().lower()`
After that, a try/catch statement was used to weed out parsing issues so that they would not result in an immediate error.
```

**Did any episodes return `"unknown"`? If so, why?**

```
No
```

**One thing about the output format that surprised you:**

```
The reasoning was clean, succinct, and stayed within a couple sentences.
```
