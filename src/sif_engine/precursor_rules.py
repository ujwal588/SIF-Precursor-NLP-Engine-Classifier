from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES = ROOT / "configs" / "precursor_rules.json"


def _matches(text: str, phrases: list[str]) -> list[str]:
    found = []
    for phrase in phrases:
        pattern = r"(?<![\w])" + re.escape(phrase) + r"(?![\w])"
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(phrase)
    return found


def load_rules(path: Path = DEFAULT_RULES) -> dict:
    with path.open(encoding="utf-8-sig") as file:
        return json.load(file)


def analyze_precursor_cues(text: str, rules: dict | None = None) -> list[dict]:
    """Return explicit phrase matches; never infer that unmatched controls exist."""
    if not text or not text.strip():
        return []
    rules = rules or load_rules()
    normalized = re.sub(r"\s+", " ", text).strip()
    results = []
    for category in rules["categories"]:
        exposures = _matches(normalized, category["exposure_terms"])
        gaps = _matches(normalized, category["control_gap_terms"])
        if exposures or gaps:
            results.append({
                "category": category["name"],
                "exposure_matches": exposures,
                "control_gap_matches": gaps,
                "evidence_level": "Exposure and explicit control-gap language" if exposures and gaps else (
                    "Exposure language only" if exposures else "Control-gap language only"
                ),
            })
    return results
