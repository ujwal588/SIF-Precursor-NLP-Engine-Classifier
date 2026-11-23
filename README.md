# SIF Precursor NLP Engine Classifier

A research prototype for mining public workplace incident records to surface text patterns associated with severe outcomes. It includes a single-report screen, model explanations, and a descriptive OSHA SIR explorer.

## Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_msha.py
python scripts/train_msha_model.py
streamlit run app.py
```

The downloader retrieves the current official MSHA Accident/Injury archive into `data/external/` (ignored by Git). The raw file is large and is not committed. It writes a source URL, retrieval timestamp, and SHA-256 manifest beside the local copy. The dashboard also reads the existing OSHA Severe Injury Report CSV from `data/raw/`; set `SIF_DATA_PATH` to select a different OSHA CSV.

## Data and target

The primary model corpus is the real MSHA Accident/Injury file. A binary **retrospective outcome proxy** is defined from recorded injury degree: code 01 (Fatality) or 02 (Permanent total/partial disability) is positive; codes 00, 03–06, and 10 are negative. Occupational illness, natural-cause, non-employee, unknown, and out-of-window records are excluded. We use a temporal split (train through 2018, validation 2019–2021, locked test 2022–2025). IDs, employers, and location identifiers are excluded as features. The final model is selected by validation average precision; the operating threshold is set on validation to maximize F2.

OSHA SIR is shown separately for descriptive exploration. It contains only severe injury reports and excludes fatalities, so it cannot supply a non-severe comparison class or train the primary model.

## Scope and safety

This dataset does not contain Oil India HSSE near-miss/unsafe-condition labels. The classifier estimates similarity to historical MSHA reports linked to specified actual injury outcomes; it does **not** identify ground-truth SIF potential, prove causation, or forecast future risk. Direct outcome terms are masked, but residual outcome leakage may remain in retrospective narratives. Use the display only for research and human review. Do not use it to rank workers, employers, or sites or make operational safety decisions.

The screen also includes a separate exact-phrase cue panel for high-energy exposures and explicit control-gap language; it never treats missing text as evidence of safety. See `docs/dataset_selection.md` for the source decision, `docs/research_landscape.md` for SIH 2026 and research comparisons, and `docs/sif_precursor_engine_research_paper.md` for the IEEE-style draft. `reports/` contains aggregate evaluation outputs. The downloaded MSHA archive stays local and is excluded from Git. A compact trained model artifact is included so the public demo can run inference immediately; the pre-existing OSHA SIR CSV remains a separate descriptive input.

## Live deployment

The Streamlit app entrypoint is `app.py`. The repository includes `models/msha_severity_proxy.joblib`, so report screening works after deployment without a training step. Deploy this repository, branch `main`, and file path `app.py` on [Streamlit Community Cloud](https://share.streamlit.io/). This is a research demo; do not paste confidential or identifiable incident reports into the public deployment.
