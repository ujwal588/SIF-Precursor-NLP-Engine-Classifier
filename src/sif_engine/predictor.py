from __future__ import annotations
import re
from pathlib import Path
import joblib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
DEFAULT_MODEL=ROOT/'models'/'msha_severity_proxy.joblib'
OUTCOME=re.compile(r'\b(?:fatal(?:ity|ities)?|deaths?|died|killed|deceased|permanent(?:ly)?\s+(?:total\s+|partial\s+)?disabilit\w*|amputat\w*|loss\s+of\s+(?:an?\s+)?eye|days?\s+away\s+from\s+work|days?\s+lost)\b',re.I)
FIELDS={'ACTIVITY':'Activity','CLASSIFICATION':'Classification','ACCIDENT_TYPE':'Accident type','MINING_EQUIP':'Equipment','SUBUNIT':'Work area','OCCUPATION':'Occupation'}
def load_model(path: Path=DEFAULT_MODEL):
    if not path.exists(): raise FileNotFoundError(f'Model not found at {path}; run scripts/train_msha_model.py first.')
    return joblib.load(path)
def _context_text(context):
    context=context or {}
    return ' '.join(f'{label.lower()} {str(context.get(key) or "").strip()}' for key,label in FIELDS.items() if str(context.get(key) or '').strip())
def explain(payload,text,limit=8):
    estimator=payload['classifier']
    if not hasattr(estimator,'coef_'): return []
    vector=payload['vectorizer'].transform([text]); idx=vector.indices; vals=vector.data
    contrib=vals*estimator.coef_[0][idx]; terms=payload['vectorizer'].get_feature_names_out()
    order=np.argsort(np.abs(contrib))[::-1][:limit]
    return [{'term':str(terms[idx[i]]),'direction':'raises' if contrib[i]>0 else 'lowers','contribution':float(contrib[i])} for i in order if abs(contrib[i])>1e-9]
def predict_report(narrative,context=None,model_payload=None):
    if not narrative or not narrative.strip(): raise ValueError('Enter a report narrative first.')
    payload=model_payload or load_model()
    masked=OUTCOME.sub(' ',re.sub(r'\s+',' ',narrative).strip())
    context_text=_context_text(context); feature=payload['feature_set']
    if feature=='context_only':
        text=context_text
        if not text: raise ValueError('Add at least one structured context field for this model.')
    elif feature=='masked_narrative_plus_context': text=f'{masked} {context_text}'.strip()
    else: text=masked
    matrix=payload['vectorizer'].transform([text]); score=float(payload['classifier'].decision_function(matrix)[0])
    return {'flag_for_review':score>=float(payload['threshold']),'screening_score':score,'threshold':float(payload['threshold']),
        'feature_set':feature,'model_name':payload['model_name'],'explanation':explain(payload,text),'score_semantics':payload['score_semantics']}

