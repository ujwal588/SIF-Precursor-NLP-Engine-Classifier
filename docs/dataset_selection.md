# Dataset choice and scope

## Selection decision

The primary model uses the official Mine Safety and Health Administration (MSHA) Accident/Injury data. It is the most useful public source found for a supervised severity analogue because a record includes an incident narrative and an explicit degree-of-injury code. The full download is reproducible through `scripts/download_msha.py`; source archives and extracted narratives are excluded from Git.

The task is deliberately narrow: distinguish recorded fatalities (code 01) and permanent total/partial disabilities (code 02) from selected recorded lower-outcome classes (codes 00, 03, 04, 05, 06, 10). Occupational illness (07), natural causes (08), non-employees (09), unknown codes, and years outside 2000–2025 are excluded. These are retrospective outcomes, not expert labels of potential SIF exposure.

## Data audit and split

| Partition | Years | Records | Positive records |
|---|---:|---:|---:|
| Train | 2000–2018 | 218,092 | 3,010 |
| Validation | 2019–2021 | 16,954 | 269 |
| Locked test | 2022–2025 | 22,463 | 354 |
| Eligible total | 2000–2025 | 257,509 | 3,633 |

The test prevalence is 1.576%. The chronological split avoids training on later years than the evaluation period. Incident, employer, and location identifiers are not model features. Near-duplicate incident grouping is also considered during data preparation so records from the same MSHA document do not cross partitions.

## Why not train on the OSHA SIR file?

The local OSHA Severe Injury Report snapshot is valuable for descriptive exploration, but OSHA describes it as reports involving hospitalization, amputation, or loss of an eye; fatalities are excluded. Therefore it does not contain the desired fatality-vs-lower-outcome contrast and is not used as the model training set. It remains a separate descriptive dashboard source.

## Why not use OSHA ITA as the primary corpus?

OSHA ITA case-detail records are real, but the selected 2025-through-2026 extract examined for this project has only 229 death-coded records overall and 37 records with NAICS beginning `21` (mining, quarrying, and oil/gas extraction). The limited positive/domain slice makes it a poor primary corpus for this experiment. ITA coverage is also bounded by OSHA electronic-submission requirements and the published records contain redacted narrative fields. Keep it as a possible future cross-source check after a more deliberate, reproducible cohort definition.

## Why MSHA, and what it cannot establish

MSHA gives us an auditable, large, real corpus with explicit outcome codes, allowing a meaningful temporal evaluation without fabricating examples. Its limitation is domain mismatch: mining injury reports are not Oil India near misses, unsafe-condition reports, or SIF-potential expert labels. The model is consequently named a *severity-proxy* model. It must not be presented as a SIF precursor detector, causal model, or operational risk score. The separate rule-based cue panel only surfaces literal phrases from a user report and reports the exact evidence it matched.

## Data handling

The downloaded MSHA archive, extracted MSHA narratives, processed MSHA data, and trained binary model are not checked into Git. An OSHA SIR CSV was already present in the repository before this work; it remains a separate descriptive input. Aggregate counts and evaluation summaries are stored under `reports/`. The downloader writes a retrieval timestamp and SHA-256 digest next to the local archive for reproducibility.

## Primary source links

- MSHA Open Government Data: https://arlweb.msha.gov/opengovernmentdata/ogimsha.asp
- OSHA Severe Injury Reports: https://www.osha.gov/severe-injury-reports
- OSHA Injury Tracking Application data: https://www.osha.gov/itadata

