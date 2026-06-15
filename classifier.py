import json
import os
import re
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    TODO — Milestone 2:

    Your prompt needs to:
      1. Describe the task and the four valid labels
      2. Show the labeled training examples so the LLM can learn the pattern
      3. Present the new description and ask for a classification

    The LLM should return a single label from VALID_LABELS (exactly as written)
    plus a brief explanation of its reasoning. Think carefully about the output
    format you request — you'll need to parse it in classify_episode().

    Before writing code, complete specs/classifier-spec.md.
    """
    # Edge case: no examples means this would silently become a zero-shot
    # prompt and likely classify inaccurately. Fail loudly instead.
    if not labeled_examples:
        raise ValueError("labeled_examples is empty")

    # 1. Task instruction + the four valid labels (from the spec).
    task_instruction = (
        "You are classifying podcast episodes by their format. Classify the "
        "episode into exactly one of these four labels:\n\n"
        "- interview: a conversation between a host and one or more guests\n"
        "- solo: a single host speaking from memory, experience, or opinion — "
        "no guests, no assembled external sources\n"
        "- panel: multiple guests with roughly equal speaking time, often "
        "debating or discussing a topic together\n"
        "- narrative: a story assembled from external sources — interviews, "
        "archival audio, reporting — with a clear narrative arc\n\n"
        "Return only the label and your reasoning. Do not explain the taxonomy."
    )

    # 2. Labeled examples, each as Title/Description/Label, separated by "---".
    example_blocks = []
    for ex in labeled_examples:
        title = ex.get("title", "")
        ex_description = ex.get("description", "")
        label = ex.get("label", "")
        example_blocks.append(
            f"Title: {title}\n"
            f"Description: {ex_description}\n"
            f"Label: {label}"
        )
    examples_section = "\n\n---\n\n".join(example_blocks)

    # 3. The new episode, same format but with the label withheld.
    new_episode = (
        f"Title: (unknown)\n"
        f"Description: {description}\n"
        f"Label: ?"
    )

    # Output format instruction — request JSON so it maps directly to the
    # {"label", "reasoning"} dict and parses reliably in classify_episode().
    output_instruction = (
        "Classify the episode above. Return your answer as a single JSON "
        "object in this exact format, with no extra text:\n"
        '{"label": "<one of interview|solo|panel|narrative>", '
        '"reasoning": "<one or two sentences>"}'
    )

    prompt = (
        f"{task_instruction}\n\n"
        f"Here are labeled examples:\n\n"
        f"{examples_section}\n\n"
        f"---\n\n"
        f"Now classify this episode:\n\n"
        f"{new_episode}\n\n"
        f"{output_instruction}"
    )

    return prompt


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    TODO — Milestone 2 (complete after build_few_shot_prompt):

    Steps:
      1. Call build_few_shot_prompt() to construct the prompt
      2. Send it to the LLM via _client.chat.completions.create()
      3. Parse the response to extract a label and reasoning
      4. Validate the label — if it's not in VALID_LABELS, set it to "unknown"
      5. Return a dict with "label" and "reasoning" keys

    Handle the case where the LLM returns something unparseable gracefully —
    don't let a bad response crash the whole evaluation.

    Before writing code, complete specs/classifier-spec.md.
    """
    # Edge case: an empty/whitespace-only description can't be classified.
    if not description or not description.strip():
        return {
            "label": "unknown",
            "reasoning": "Empty description — nothing to classify.",
        }

    # Step 1 — Build the prompt (pass both args through unchanged).
    prompt = build_few_shot_prompt(labeled_examples, description)

    # Step 2 — Send to the LLM. Wrap the call so a network/API error returns
    # "unknown" instead of crashing the 20-call evaluation loop (Step 5).
    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
    except Exception as e:
        return {
            "label": "unknown",
            "reasoning": f"LLM request failed: {e}",
        }

    # Step 3 — Parse the response. JSON is requested, but the model may wrap
    # it in markdown fences or add stray text, so parse defensively.
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
        # Unparseable response (Step 5).
        return {
            "label": "unknown",
            "reasoning": f"Could not parse response: {raw!r}",
        }

    # Step 4 — Validate the label against VALID_LABELS.
    if label not in VALID_LABELS:
        return {
            "label": "unknown",
            "reasoning": f"Returned label not in VALID_LABELS. Raw response: {raw!r}",
        }

    # Step 5 — Return the validated result.
    return {
        "label": label,
        "reasoning": reasoning,
    }
