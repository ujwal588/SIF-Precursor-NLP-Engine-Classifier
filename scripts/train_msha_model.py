from __future__ import annotations
import argparse, gc, json, re
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / 'data' / 'external' / 'msha' / 'extracted' / 'Accidents.txt'
MODEL_DIR, REPORT_DIR = ROOT / 'models', ROOT / 'reports'
SEED = 2601
TRAIN_END, VALID_END, TEST_END = 2018, 2021, 2025
POSITIVE, NEGATIVE = {'01', '02'}, {'00', '03', '04', '05', '06', '10'}
CONTEXT = ['ACTIVITY', 'CLASSIFICATION', 'ACCIDENT_TYPE', 'MINING_EQUIP', 'SUBUNIT', 'OCCUPATION']
REQUIRED = ['MINE_ID', 'DOCUMENT_NO', 'CAL_YR', 'DEGREE_INJURY_CD', 'NARRATIVE', *CONTEXT]
OUTCOME = re.compile(r'\b(?:fatal(?:ity|ities)?|deaths?|died|killed|deceased|permanent(?:ly)?\s+(?:total\s+|partial\s+)?disabilit\w*|amputat\w*|loss\s+of\s+(?:an?\s+)?eye|days?\s+away\s+from\s+work|days?\s+lost)\b', re.I)

def clean(x) -> str:
    if x is None or pd.isna(x): return ''
    return re.sub(r'\s+', ' ', str(x)).strip()

def load_data(path: Path):
    df = pd.read_csv(path, sep='|', usecols=lambda c: c in REQUIRED, dtype=str,
        encoding='utf-8-sig', encoding_errors='replace', low_memory=False)
    missing = [c for c in REQUIRED if c not in df]
    if missing: raise ValueError(f'MSHA file is missing fields: {missing}')
    raw_rows = len(df)
    df['year'] = pd.to_numeric(df.CAL_YR, errors='coerce')
    df['degree_code'] = df.DEGREE_INJURY_CD.map(lambda x: clean(x).zfill(2) if clean(x).isdigit() else clean(x))
    duplicate_keys = int(df.duplicated(['MINE_ID', 'DOCUMENT_NO']).sum())
    df = df.drop_duplicates(['MINE_ID', 'DOCUMENT_NO'], keep='first')
    df = df[df.year.between(2000, TEST_END)].copy()
    df = df[df.degree_code.isin(POSITIVE | NEGATIVE)].copy()
    df['label'] = df.degree_code.isin(POSITIVE).astype('int8')
    df['narrative'] = df.NARRATIVE.map(clean)
    df = df[df.narrative.str.len().ge(12)].copy()
    df['masked_narrative'] = df.narrative.map(lambda s: OUTCOME.sub(' ', s))
    df['context'] = df[CONTEXT].apply(lambda row: ' '.join(
        f'{col.lower()} {clean(value)}' for col, value in row.items() if clean(value)), axis=1)
    df['masked_plus_context'] = (df.masked_narrative + ' ' + df.context).str.strip()
    df['incident_group'] = df.MINE_ID.fillna('') + '|' + df.DOCUMENT_NO.fillna('')
    excluded = df.groupby('degree_code').size().to_dict()
    profile = {
        'source_file': path.name, 'source_rows': raw_rows,
        'duplicate_mine_document_keys_removed': duplicate_keys,
        'eligible_rows': len(df), 'years_included': [2000, TEST_END],
        'positive_rows': int(df.label.sum()), 'negative_rows': int((df.label == 0).sum()),
        'positive_rate': float(df.label.mean()),
        'fatality_rows': int(df.degree_code.eq('01').sum()),
        'permanent_disability_rows': int(df.degree_code.eq('02').sum()),
        'included_degree_code_counts': {str(k): int(v) for k, v in excluded.items()},
        'target_definition': 'Positive = MSHA codes 01 (Fatality) or 02 (Permanent total/partial disability). Negative = 00, 03, 04, 05, 06, 10. Other/unknown outcomes excluded.',
        'interpretation': 'Retrospective injury-outcome proxy, not a label for hypothetical SIF potential and not a prospective risk estimate.',
    }
    return df, profile

def split(df):
    train = df[df.year <= TRAIN_END]
    valid = df[df.year.between(TRAIN_END + 1, VALID_END)]
    test = df[df.year.between(VALID_END + 1, TEST_END)]
    if any(len(s) == 0 or s.label.sum() == 0 for s in (train, valid, test)):
        raise ValueError('A temporal split has no records or positive cases.')
    return train, valid, test

def threshold_f2(y, score):
    p, r, t = precision_recall_curve(y, score)
    if not len(t): return float('inf'), 0.0
    f2 = np.divide(5*p[:-1]*r[:-1], 4*p[:-1]+r[:-1], out=np.zeros_like(p[:-1]), where=(4*p[:-1]+r[:-1]) > 0)
    i = int(np.nanargmax(f2))
    return float(t[i]), float(f2[i])

def scores(estimator, x):
    return estimator.decision_function(x)

def metrics(y, score, threshold):
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    p, r = precision_score(y, pred, zero_division=0), recall_score(y, pred, zero_division=0)
    f2 = 5*p*r/(4*p+r) if 4*p+r else 0.0
    return {'n': int(len(y)), 'positives': int(np.sum(y)), 'positive_rate': float(np.mean(y)),
        'threshold': float(threshold), 'accuracy': float(accuracy_score(y,pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y,pred)), 'precision': float(p), 'recall': float(r),
        'f1': float(f1_score(y,pred,zero_division=0)), 'f2': float(f2),
        'specificity': float(tn/(tn+fp)) if tn+fp else 0.0,
        'average_precision': float(average_precision_score(y,score)), 'roc_auc': float(roc_auc_score(y,score)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

def ap_ci(y, s, iterations=400):
    rng, out = np.random.default_rng(SEED), []
    for _ in range(iterations):
        ix = rng.integers(0, len(y), len(y))
        if y[ix].min() != y[ix].max(): out.append(average_precision_score(y[ix],s[ix]))
    return [float(x) for x in np.percentile(out,[2.5,97.5])] if out else [None,None]

def variants(df):
    return {'narrative_raw_diagnostic': df.narrative,
        'narrative_outcome_terms_masked': df.masked_narrative,
        'context_only': df.context,
        'masked_narrative_plus_context': df.masked_plus_context}

def top_terms(vectorizer, estimator, limit=30):
    names = np.asarray(vectorizer.get_feature_names_out()); coef = estimator.coef_[0]
    pos, neg = np.argsort(coef)[-limit:][::-1], np.argsort(coef)[:limit]
    return ([{'direction':'higher-outcome class','term':str(names[i]),'weight':float(coef[i])} for i in pos] +
            [{'direction':'lower-outcome class','term':str(names[i]),'weight':float(coef[i])} for i in neg])

def main():
    ap = argparse.ArgumentParser(description='Compare real-data text baselines for a retrospective severity proxy.')
    ap.add_argument('--data', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--max-features', type=int, default=70000)
    args = ap.parse_args()
    if not args.data.exists(): raise FileNotFoundError(f'No MSHA file at {args.data}; run python scripts/download_msha.py')
    df, profile = load_data(args.data); train, valid, test = split(df)
    profile['splits'] = {}
    for name, part in [('train',train),('validation',valid),('test',test)]:
        profile['splits'][name] = {'years':[int(part.year.min()),int(part.year.max())],
            'records':len(part),'positive_records':int(part.label.sum()),'positive_rate':float(part.label.mean()),
            'incident_groups':int(part.incident_group.nunique())}
    comparisons=[]; best=None
    for name, texts in variants(df).items():
        vectorizer = TfidfVectorizer(lowercase=True,strip_accents='unicode',stop_words='english',
            ngram_range=(1,2),min_df=3,max_features=args.max_features,sublinear_tf=True,dtype=np.float32,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9'-]+\b")
        xtr=vectorizer.fit_transform(texts.loc[train.index].fillna('').astype(str))
        xv=vectorizer.transform(texts.loc[valid.index].fillna('').astype(str))
        xt=vectorizer.transform(texts.loc[test.index].fillna('').astype(str))
        for kind in ['LogisticRegression','LinearSVC']:
            model = LogisticRegression(class_weight='balanced',max_iter=500,solver='liblinear',random_state=SEED) if kind=='LogisticRegression' else LinearSVC(class_weight='balanced',random_state=SEED,max_iter=5000)
            model.fit(xtr,train.label.to_numpy())
            sv=scores(model,xv); threshold,val_f2=threshold_f2(valid.label.to_numpy(),sv)
            st=scores(model,xt); tm=metrics(test.label.to_numpy(),st,threshold)
            val_ap=float(average_precision_score(valid.label,sv))
            comparisons.append({'feature_set':name,'model':kind,'validation_average_precision':val_ap,
                'validation_f2':val_f2,'test':tm})
            # Raw post-event text is a leakage diagnostic only, never a deployable candidate.
            if name!='narrative_raw_diagnostic' and (best is None or val_ap>best['validation_ap']):
                best={'feature_set':name,'model_name':kind,'validation_ap':val_ap,'threshold':threshold,
                    'model':model,'vectorizer':vectorizer,'test_scores':st,'test_metrics':tm}
        del xtr,xv,xt,vectorizer; gc.collect()
    best['test_metrics']['average_precision_95ci']=ap_ci(test.label.to_numpy(),best['test_scores'])
    baseline=metrics(test.label.to_numpy(),np.zeros(len(test),dtype=float),0.5)
    comparisons.insert(0,{'feature_set':'majority_baseline','model':'always_negative',
        'validation_average_precision':float(valid.label.mean()),'validation_f2':0.0,'test':baseline})
    profile['majority_baseline']=baseline
    payload={'vectorizer':best['vectorizer'],'classifier':best['model'],'feature_set':best['feature_set'],
        'model_name':best['model_name'],'threshold':best['threshold'],
        'labels':{0:'no fatal/permanent-disability outcome proxy',1:'fatal/permanent-disability outcome proxy'},
        'train_years':[2000,TRAIN_END],'validation_years':[TRAIN_END+1,VALID_END],
        'source':'Official MSHA public Accident/Injury dataset','score_semantics':'Uncalibrated decision score; not a probability or prospective SIF-risk estimate.',
        'direct_outcome_terms_masked':True,'context_fields':CONTEXT,'test_metrics':best['test_metrics'],'seed':SEED}
    MODEL_DIR.mkdir(exist_ok=True); REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(payload,MODEL_DIR/'msha_severity_proxy.joblib',compress=3)
    profile['deployed_candidate']={'feature_set':best['feature_set'],'model':best['model_name'],
        'validation_average_precision':best['validation_ap'],'threshold':best['threshold'],
        'test_metrics':best['test_metrics'],'top_terms':top_terms(best['vectorizer'],best['model'])}
    (REPORT_DIR/'dataset_profile.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
    (REPORT_DIR/'model_comparison.json').write_text(json.dumps(comparisons,indent=2),encoding='utf-8')
    rows=[{'feature_set':c['feature_set'],'model':c['model'],'validation_average_precision':c['validation_average_precision'],**c['test']} for c in comparisons]
    pd.DataFrame(rows).to_csv(REPORT_DIR/'model_comparison.csv',index=False)
    m=best['test_metrics']; ci=m['average_precision_95ci']
    card=f'''# MSHA severity-proxy model card

## Task and label

This model distinguishes records associated with a recorded fatality or permanent total/partial disability (MSHA degree codes 01/02) from selected lower-outcome categories (00, 03–06, 10). It predicts a retrospective administrative outcome label. It does **not** predict hypothetical SIF potential, establish causation, or forecast future injuries.

## Data and evaluation

- Official MSHA accident/injury source; real records only, no synthetic training examples.
- Eligible records: {profile['eligible_rows']:,} (from {profile['source_rows']:,} source rows); positives: {profile['positive_rows']:,}; prevalence: {profile['positive_rate']:.2%}.
- Train: {profile['splits']['train']['years']} ({profile['splits']['train']['records']:,} records; {profile['splits']['train']['positive_records']:,} positives).
- Validation: {profile['splits']['validation']['years']} ({profile['splits']['validation']['records']:,} records; {profile['splits']['validation']['positive_records']:,} positives).
- Locked temporal test: {profile['splits']['test']['years']} ({profile['splits']['test']['records']:,} records; {profile['splits']['test']['positive_records']:,} positives).
- Identifiers, operator names, and location IDs are not features. Threshold was selected on validation to maximize F2; test set was not used for model selection.
- Raw post-event text is tested only as a leakage diagnostic. Final candidate selection excludes it; direct outcome words are masked, though residual outcome leakage may remain.

## Locked test results

| Metric | Result |
|---|---:|
| Average precision | {m['average_precision']:.4f} (bootstrap 95% CI {ci[0]:.4f}–{ci[1]:.4f}) |
| ROC-AUC | {m['roc_auc']:.4f} |
| Precision | {m['precision']:.4f} |
| Recall | {m['recall']:.4f} |
| F1 / F2 | {m['f1']:.4f} / {m['f2']:.4f} |
| Specificity | {m['specificity']:.4f} |
| Confusion counts | TP {m['tp']}, FP {m['fp']}, TN {m['tn']}, FN {m['fn']} |

- Always-negative baseline: AP {baseline['average_precision']:.4f}, recall {baseline['recall']:.4f}, accuracy {baseline['accuracy']:.4f}.

## Limitations

The data cover mining, not Oil India HSSE observations, and do not represent all hazards or near misses. Reports are retrospective and vary in detail; term masking cannot guarantee that every injury-outcome clue was removed. A flagged record should be read as a text match to the defined outcome classes in this source, never as a probability or an operational safety verdict. Domain expert validation and independently labeled near-miss/unsafe-condition data are required before making SIF-potential claims.
'''
    (REPORT_DIR/'model_card.md').write_text(card,encoding='utf-8')
    print(f"Candidates: {len(comparisons)}; selected {best['model_name']} / {best['feature_set']}")
    print(json.dumps(best['test_metrics'],indent=2))
    print(f'Model saved: {MODEL_DIR / "msha_severity_proxy.joblib"}')
    print(f'Results saved: {REPORT_DIR / "model_card.md"}')

if __name__=='__main__': main()


