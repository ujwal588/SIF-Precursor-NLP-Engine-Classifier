# MSHA severity-proxy model card

## Task and label

This model distinguishes records associated with a recorded fatality or permanent total/partial disability (MSHA degree codes 01/02) from selected lower-outcome categories (00, 03–06, 10). It predicts a retrospective administrative outcome label. It does **not** predict hypothetical SIF potential, establish causation, or forecast future injuries.

## Data and evaluation

- Official MSHA accident/injury source; real records only, no synthetic training examples.
- Eligible records: 257,509 (from 276,013 source rows); positives: 3,633; prevalence: 1.41%.
- Train: [2000, 2018] (218,092 records; 3,010 positives).
- Validation: [2019, 2021] (16,954 records; 269 positives).
- Locked temporal test: [2022, 2025] (22,463 records; 354 positives).
- Identifiers, operator names, and location IDs are not features. Threshold was selected on validation to maximize F2; test set was not used for model selection.
- Raw post-event text is tested only as a leakage diagnostic. Final candidate selection excludes it; direct outcome words are masked, though residual outcome leakage may remain.

## Locked test results

| Metric | Result |
|---|---:|
| Average precision | 0.3995 (bootstrap 95% CI 0.3472–0.4601) |
| ROC-AUC | 0.9523 |
| Precision | 0.3177 |
| Recall | 0.5565 |
| F1 / F2 | 0.4045 / 0.4838 |
| Specificity | 0.9809 |
| Confusion counts | TP 197, FP 423, TN 21686, FN 157 |

- Always-negative baseline: AP 0.0158, recall 0.0000, accuracy 0.9842.

## Limitations

The data cover mining, not Oil India HSSE observations, and do not represent all hazards or near misses. Reports are retrospective and vary in detail; term masking cannot guarantee that every injury-outcome clue was removed. A flagged record should be read as a text match to the defined outcome classes in this source, never as a probability or an operational safety verdict. Domain expert validation and independently labeled near-miss/unsafe-condition data are required before making SIF-potential claims.
