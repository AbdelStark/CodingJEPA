"""Deterministic demo example (RFC-0006 §D5).

Run with: python examples/demo-cpython-extract-helper.py

Produces a reproducible diff using a hand-crafted before/after pair
without requiring a trained checkpoint or FAISS index.
"""

from codingjepa.demo.diff import render_diff_terminal

BEFORE = """
def process_data(records, threshold=0.5):
    results = []
    for record in records:
        score = record["value"] / record["total"]
        if score > threshold:
            results.append({"id": record["id"], "score": score})
    return results
"""

AFTER = """
def _compute_score(record):
    return record["value"] / record["total"]


def process_data(records, threshold=0.5):
    return [
        {"id": r["id"], "score": _compute_score(r)}
        for r in records
        if _compute_score(r) > threshold
    ]
"""

if __name__ == "__main__":
    diff = render_diff_terminal(BEFORE.strip(), AFTER.strip())
    print("extract-helper demo — before/after diff:")
    print(diff)
