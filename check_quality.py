"""
Project: CSV Data Quality Checker powered by Claude
Level: Beginner-friendly, production-minded

WHAT IT DOES
1. Reads the first N rows of a CSV (keeps token cost low and bounded).
2. Sends them to Claude with a strict system prompt.
3. Asks Claude to return a JSON quality report.
4. Validates the JSON, then saves it to disk.

WHY THIS DESIGN
- Sampling caps cost. Sending a 10M-row file to an LLM is an anti-pattern.
- A system prompt sets the role once, so each call stays consistent.
- We demand JSON and still validate it, because models can drift.
"""

import csv
import io
import json
import os
import sys

from dotenv import load_dotenv
from anthropic import Anthropic

# ---------- CONFIGURATION ----------
load_dotenv()                          # Reads ANTHROPIC_API_KEY from .env
MODEL = "claude-sonnet-5-5"            # Balanced choice for data tasks (verify in docs)
MAX_SAMPLE_ROWS = 200                  # Hard cap on rows sent to the model
MAX_OUTPUT_TOKENS = 1500               # Enough for a structured report
REPORT_PATH = "quality_report.json"

client = Anthropic()                   # Uses ANTHROPIC_API_KEY automatically

# ---------- PROMPT ----------
SYSTEM_PROMPT = """You are a senior data engineer reviewing CSV data quality.
Check for: missing values, duplicates, invalid dates, negative or non-numeric
numbers, and inconsistent categorical values (e.g. casing).

Respond with ONLY valid JSON, no markdown, no commentary, using this schema:
{
  "issues": [
    {"column": "string", "row_hint": "string", "problem": "string",
     "severity": "high | medium | low"}
  ],
  "summary": "one-sentence overall assessment",
  "suggested_fixes": ["string"]
}"""


def load_sample(path: str, limit: int):
    """Read the header and the first `limit` data rows."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)          # First line = column names
        rows = []
        for i, row in enumerate(reader):
            if i >= limit:             # Stop early; never load the full file
                break
            rows.append(row)
    return header, rows


def to_csv_text(header, rows) -> str:
    """Re-serialize rows with csv.writer so commas inside values stay safe."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def check_quality(csv_text: str) -> dict:
    """Ask Claude for a quality report and return it as a Python dict."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=SYSTEM_PROMPT,          # Role and rules live here, not in the user turn
        messages=[
            {"role": "user",
             "content": f"Review this CSV sample:\n\n{csv_text}"}
        ],
    )

    raw = response.content[0].text.strip()

    # Defensive parsing: models sometimes wrap JSON in code fences.
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Never crash the pipeline on bad output; keep the raw text for debugging.
        return {"error": "Model did not return valid JSON", "raw": raw}


def main(path: str) -> None:
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")

    header, rows = load_sample(path, MAX_SAMPLE_ROWS)
    print(f"Loaded {len(rows)} rows, {len(header)} columns from {path}")

    report = check_quality(to_csv_text(header, rows))

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to {REPORT_PATH}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main("sales_messy.csv")