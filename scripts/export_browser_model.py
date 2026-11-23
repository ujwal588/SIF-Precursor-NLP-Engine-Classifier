from __future__ import annotations
import json
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
PUBLIC = SITE / "public"
PUBLIC.mkdir(parents=True, exist_ok=True)
payload = joblib.load(ROOT / "models" / "msha_severity_proxy.joblib")
vectorizer = payload["vectorizer"]
classifier = payload["classifier"]
terms = [None] * len(vectorizer.vocabulary_)
for term, index in vectorizer.vocabulary_.items():
    terms[index] = term
browser_model = {
    "format": "tfidf-logreg-v1",
    "terms": terms,
    "idf": vectorizer.idf_.tolist(),
    "coef": classifier.coef_[0].tolist(),
    "intercept": float(classifier.intercept_[0]),
    "threshold": float(payload["threshold"]),
    "stopWords": sorted(vectorizer.get_stop_words()),
    "tokenPattern": r"[A-Za-z][A-Za-z0-9'-]+",
    "fields": [
        ["ACTIVITY", "Activity"], ["CLASSIFICATION", "Classification"],
        ["ACCIDENT_TYPE", "Accident type"], ["MINING_EQUIP", "Equipment"],
        ["SUBUNIT", "Work area"], ["OCCUPATION", "Occupation"]
    ],
    "outcomePattern": r"\b(?:fatal(?:ity|ities)?|deaths?|died|killed|deceased|permanent(?:ly)?\s+(?:total\s+|partial\s+)?disabilit\w*|amputat\w*|loss\s+of\s+(?:an?\s+)?eye|days?\s+away\s+from\s+work|days?\s+lost)\b",
    "modelName": payload["model_name"],
    "featureSet": payload["feature_set"],
    "scoreSemantics": payload["score_semantics"]
}
with (PUBLIC / "model.json").open("w", encoding="utf-8") as file:
    json.dump(browser_model, file, ensure_ascii=False, separators=(",", ":"))

profile = json.loads((ROOT / "reports" / "dataset_profile.json").read_text(encoding="utf-8"))
comparisons = json.loads((ROOT / "reports" / "model_comparison.json").read_text(encoding="utf-8"))
metrics = {
    "eligible": profile["eligible_rows"],
    "sourceRows": profile["source_rows"],
    "positives": profile["positive_rows"],
    "positiveRate": profile["positive_rate"],
    "splits": profile["splits"],
    "selected": profile["deployed_candidate"],
    "baseline": profile["majority_baseline"],
    "comparisons": [{
        "label": f"{item['feature_set'].replace('_', ' ')} · {item['model']}",
        "featureSet": item["feature_set"],
        "model": item["model"],
        "validationAP": item["validation_average_precision"],
        "testAP": item["test"]["average_precision"],
        "testF2": item["test"]["f2"],
        "testRecall": item["test"]["recall"],
        "testPrecision": item["test"]["precision"]
    } for item in comparisons]
}
(PUBLIC / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
(PUBLIC / "precursor_rules.json").write_bytes((ROOT / "configs" / "precursor_rules.json").read_bytes())
print(f"Wrote {PUBLIC / 'model.json'} ({(PUBLIC / 'model.json').stat().st_size:,} bytes)")
print(f"Wrote {PUBLIC / 'metrics.json'} ({(PUBLIC / 'metrics.json').stat().st_size:,} bytes)")
