from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.sif_engine.predictor import load_model, predict_report
from src.sif_engine.precursor_rules import analyze_precursor_cues

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "models" / "msha_severity_proxy.joblib"
PROFILE_PATH = ROOT / "reports" / "dataset_profile.json"
COMPARISON_PATH = ROOT / "reports" / "model_comparison.csv"

st.set_page_config(page_title="SIF Lens | Safety Research", page_icon="◈", layout="wide")
st.markdown("""
<style>
.stApp {
  background: radial-gradient(ellipse at 8% 4%, #eef7d6 0, transparent 36%),
              radial-gradient(ellipse at 93% 72%, #e5f1d5 0, transparent 34%),
              linear-gradient(135deg, #f6f9f0 0%, #eef4e7 100%);
  color: #26352e;
}
.stApp [data-testid="stAppViewContainer"] { position: relative; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1220px; padding-top: 1.15rem; padding-bottom: 3rem; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none!important; }
.hero {
  position: relative; overflow: hidden; background: linear-gradient(115deg,#fff 42%,#f3f8e9);
  border: 1px solid #dce7ce; border-left: 7px solid #9bc63b; border-radius: 20px;
  padding: 1.45rem 1.8rem; margin: .25rem 0 .85rem; box-shadow: 0 12px 30px #344b3210;
}
.hero:after { content:""; position:absolute; width:170px;height:170px;border:18px solid #edf4df;
  border-radius:50%; right:58px;top:-85px;pointer-events:none; }
.eyebrow { color:#6d872b; text-transform:uppercase; letter-spacing:.13em; font-weight:750; font-size:.72rem; }
.hero h1 { color:#26352e; margin:.4rem 0 .25rem; font-size:2rem; }
.hero p { color:#68746b; margin:0; max-width:830px; }
.notice { background:#f6f9eb; border:1px solid #e1e9ce; border-left:4px solid #9bc63b;
  border-radius:10px; padding:.75rem 1rem; color:#465247; margin:.65rem 0 1rem; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#fff 52%,#f8faef); border:1px solid #e0e8d4;
  padding:.85rem 1rem; border-radius:15px; box-shadow:0 7px 22px #344b320b; }
[data-testid="stMetricLabel"] p { color:#6d786f!important; font-size:.82rem; }
[data-testid="stMetricValue"] { color:#26352e; font-weight:750; }
.stRadio > div { gap:.55rem; }
.stRadio [role="radiogroup"] label { background:#fff; border:1px solid #dfe7d5; border-radius:999px;
  padding:.4rem .95rem; box-shadow:0 3px 12px #344b3208; }
.stRadio [role="radiogroup"] label:has(input:checked) { background:#e8f2d2; border-color:#b9d27e; }
.stButton > button, [data-testid="stFormSubmitButton"] button { border-radius:10px; font-weight:700;
  min-height:2.75rem; border:1px solid #8cac39; background:#9bc63b; color:#243326; }
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover { background:#add64b; border-color:#8cac39; color:#243326; }
[data-testid="stVerticalBlockBorderWrapper"] { background:rgba(255,255,255,.86); border:1px solid #e0e8d4;
  border-radius:16px; box-shadow:0 7px 22px #344b3209; }
[data-testid="stDataFrame"] { border:1px solid #e0e8d4; border-radius:12px; overflow:hidden; }
[data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input { border-radius:10px; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_json(path: str):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_resource(show_spinner=False)
def get_model(path: str, modified: int):
    return load_model(Path(path))


@st.cache_data(show_spinner="Loading the OSHA SIR explorer…")
def load_osha(path: str, modified: int):
    frame = pd.read_csv(path, low_memory=False)
    frame.columns = [str(name).strip() for name in frame.columns]
    return frame


def pick_column(frame: pd.DataFrame, *names: str):
    lookup = {str(name).casefold(): name for name in frame.columns}
    return next((lookup[name.casefold()] for name in names if name.casefold() in lookup), None)


def theme_matches(text: str, taxonomy: dict):
    value = str(text).casefold()
    return [name for name, phrases in taxonomy.items() if any(str(term).casefold() in value for term in phrases)]


def render_overview(model, profile):
    st.markdown('<div class="eyebrow">Measured public-data baseline</div>', unsafe_allow_html=True)
    st.subheader("A transparent starting point for incident review")
    if profile:
        splits = profile.get("splits", {})
        test = splits.get("test", {})
        metrics = profile.get("deployed_candidate", {}).get("test_metrics", {})
        cards = st.columns(5)
        cards[0].metric("Eligible reports", f"{profile.get('eligible_rows', 0):,}")
        cards[1].metric("Test reports", f"{test.get('records', 0):,}")
        cards[2].metric("Test average precision", f"{metrics.get('average_precision', 0):.3f}")
        cards[3].metric("Recall", f"{metrics.get('recall', 0):.1%}")
        cards[4].metric("Precision", f"{metrics.get('precision', 0):.1%}")
        left, right = st.columns([1.05, .95], gap="large")
        with left:
            st.markdown("#### Evaluation timeline")
            timeline = pd.DataFrame({
                "Reports": [splits.get("train", {}).get("records", 0),
                            splits.get("validation", {}).get("records", 0),
                            test.get("records", 0)]
            }, index=["Train · 2000–2018", "Validation · 2019–2021", "Test · 2022–2025"])
            st.bar_chart(timeline, height=245, color="#9bc63b")
        with right:
            st.markdown("#### What the model learns")
            st.markdown("<div class='notice'><b>Retrospective outcome proxy.</b> MSHA degree codes 01/02 (fatality or permanent disability) versus selected lower-outcome codes.</div>", unsafe_allow_html=True)
            st.write("The classifier looks for narrative patterns associated with recorded outcomes. It does not determine whether an incident had SIF potential or predict future risk.")
            st.caption("Real public records · no synthetic training examples · time-ordered holdout")
    else:
        st.warning("Evaluation summary is missing. Run `python scripts/train_msha_model.py`.")
    if model is None:
        st.info("The local trained model is not present. Download the data and run the training command in the README to enable report review.")


def render_screen(model):
    st.markdown('<div class="eyebrow">Report review · one narrative at a time</div>', unsafe_allow_html=True)
    st.subheader("Explore the wording and the model response")
    st.write("Submit a report to see literal hazard/control-gap phrases beside the historical severity comparison.")
    st.markdown("<div class='notice'><b>Privacy:</b> This public demo processes report text in its hosted session. Do not enter worker names, employer-confidential details, or real sensitive incident reports.</div>", unsafe_allow_html=True)
    with st.form("report_review"):
        narrative = st.text_area("Incident narrative", height=150,
                                 placeholder="Describe the task, energy sources, safeguards, and what happened…")
        with st.expander("Optional incident context"):
            a, b, c = st.columns(3)
            context = {"ACTIVITY": a.text_input("Activity"),
                       "CLASSIFICATION": b.text_input("Classification"),
                       "ACCIDENT_TYPE": c.text_input("Accident type")}
            d, e, f = st.columns(3)
            context.update({"MINING_EQUIP": d.text_input("Equipment"),
                            "SUBUNIT": e.text_input("Work area"),
                            "OCCUPATION": f.text_input("Occupation")})
        submitted = st.form_submit_button("Review report", type="primary", use_container_width=True)
    if submitted:
        if not narrative.strip():
            st.warning("Enter a narrative first.")
            st.session_state.pop("last_screen", None)
        else:
            payload = {"narrative": narrative, "context": context, "cues": analyze_precursor_cues(narrative)}
            if model is not None:
                try:
                    payload["prediction"] = predict_report(narrative, context, model)
                    payload["prediction_error"] = None
                except Exception as exc:
                    payload["prediction_error"] = str(exc)
            st.session_state["last_screen"] = payload
    result = st.session_state.get("last_screen")
    if not result:
        st.markdown("<div class='notice'>Cue matching and model scoring are separate outputs. Missing phrases never mean a hazard or control is absent.</div>", unsafe_allow_html=True)
        return
    left, right = st.columns([1, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### Explicit language cues")
            cues = result["cues"]
            if cues:
                display = [{"Category": cue["category"], "Evidence": cue["evidence_level"],
                            "Exposure phrases": ", ".join(cue["exposure_matches"]) or "—",
                            "Control-gap phrases": ", ".join(cue["control_gap_matches"]) or "—"} for cue in cues]
                st.dataframe(pd.DataFrame(display), hide_index=True, use_container_width=True)
            else:
                st.info("No configured phrases matched. This does not establish that the situation is safe.")
            st.caption("Deterministic phrase matches · transparent, limited coverage · not a trained risk score")
    with right:
        with st.container(border=True):
            st.markdown("#### Historical severity comparison")
            prediction = result.get("prediction")
            if prediction:
                if prediction["flag_for_review"]:
                    st.warning("Above the review threshold · inspect the report and its context")
                else:
                    st.success("Below the review threshold · not a finding of safety")
                x, y = st.columns(2)
                x.metric("Uncalibrated score", f"{prediction['screening_score']:.3f}")
                y.metric("Review threshold", f"{prediction['threshold']:.3f}")
                st.caption(prediction["score_semantics"])
                explanation = prediction.get("explanation", [])
                if explanation:
                    st.markdown("**Text features influencing this score**")
                    st.dataframe(pd.DataFrame(explanation).rename(columns={
                        "term": "Text feature", "direction": "Score effect", "contribution": "Contribution"
                    }), hide_index=True, use_container_width=True)
                st.caption("Feature contributions explain model behavior, not verified hazards or causes.")
            else:
                st.info(result.get("prediction_error") or "Train the local MSHA model to enable this comparison.")
    st.markdown("<div class='notice'><b>Scope:</b> trained on mining incident outcomes. Not validated for Oil India, near misses, unsafe conditions, or potential SIF classification.</div>", unsafe_allow_html=True)


def render_performance(model):
    st.markdown('<div class="eyebrow">Model evaluation · chronological holdout</div>', unsafe_allow_html=True)
    st.subheader("How the selected baseline performed")
    if model:
        metrics = model["test_metrics"]
        cards = st.columns(4)
        cards[0].metric("Average precision", f"{metrics['average_precision']:.3f}")
        cards[1].metric("Recall", f"{metrics['recall']:.1%}")
        cards[2].metric("Precision", f"{metrics['precision']:.1%}")
        cards[3].metric("Specificity", f"{metrics['specificity']:.1%}")
        st.caption("Locked test period: 2022–2025 · 354 positive records · threshold selected on validation, not test")
        left, right = st.columns([1.1, .9], gap="large")
        with left:
            st.markdown("#### Baseline comparison · test average precision")
            if COMPARISON_PATH.exists():
                comparison = pd.read_csv(COMPARISON_PATH)
                comparison["Candidate"] = comparison["feature_set"].str.replace("_", " ") + " · " + comparison["model"]
                chart = comparison.set_index("Candidate")["average_precision"].sort_values()
                st.bar_chart(chart, horizontal=True, height=360, color="#9bc63b")
                st.caption("The raw narrative row is a leakage diagnostic only; it is not eligible for selection.")
        with right:
            st.markdown("#### Review threshold outcomes")
            cm = pd.DataFrame({"Predicted review": [metrics["tp"], metrics["fp"]],
                               "Below threshold": [metrics["fn"], metrics["tn"]]},
                              index=["Higher-outcome records", "Lower-outcome records"])
            st.dataframe(cm, use_container_width=True)
            st.markdown("<div class='notice'>False alerts and missed positives remain material. The interface is for human-reviewed research only.</div>", unsafe_allow_html=True)
        with st.expander("Full model comparison and model card"):
            if COMPARISON_PATH.exists():
                st.dataframe(pd.read_csv(COMPARISON_PATH), hide_index=True, use_container_width=True)
            card = ROOT / "reports" / "model_card.md"
            if card.exists():
                st.markdown(card.read_text(encoding="utf-8"))
    else:
        st.warning("Train the model before viewing measured performance.")


def render_data_explorer():
    st.markdown('<div class="eyebrow">Separate descriptive corpus · OSHA Severe Injury Reports</div>', unsafe_allow_html=True)
    st.subheader("Browse public severe-injury reports")
    configured = os.getenv("SIF_DATA_PATH")
    candidates = [Path(configured)] if configured else sorted(RAW_DIR.glob("*.csv"))
    csv_path = next((path for path in candidates if path and path.is_file()), None)
    if csv_path is None:
        st.info("No OSHA SIR CSV found in `data/raw/`. The report screen and model metrics remain available.")
        return
    try:
        frame = load_osha(str(csv_path), csv_path.stat().st_mtime_ns)
    except Exception as exc:
        st.error(f"Unable to load the OSHA CSV: {exc}")
        return
    narrative_col = pick_column(frame, "Final Narrative", "Narrative", "Description")
    date_col = pick_column(frame, "EventDate", "Event Date")
    state_col = pick_column(frame, "State")
    if narrative_col is None:
        st.error("Could not find a narrative column in this OSHA CSV.")
        return
    year = pd.to_datetime(frame[date_col], errors="coerce").dt.year if date_col else pd.Series(pd.NA, index=frame.index)
    filtered = frame.copy()
    filtered["Reporting year"] = year
    col_year, col_state, col_query = st.columns([1, 1, 1.4])
    if date_col and year.notna().any():
        years = sorted(year.dropna().astype(int).unique())
        chosen_years = col_year.multiselect("Reporting year", years, default=years)
        filtered = filtered[filtered["Reporting year"].isin(chosen_years)]
    if state_col:
        states = sorted(filtered[state_col].dropna().astype(str).unique())
        chosen_states = col_state.multiselect("State", states)
        if chosen_states:
            filtered = filtered[filtered[state_col].astype(str).isin(chosen_states)]
    query = col_query.text_input("Narrative contains")
    if query:
        filtered = filtered[filtered[narrative_col].fillna("").astype(str).str.contains(query, case=False, regex=False)]
    hospitalized = pick_column(frame, "Hospitalized")
    amputation = pick_column(frame, "Amputation")
    eye_loss = pick_column(frame, "Loss of Eye", "LossOfEye")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Reports in view", f"{len(filtered):,}")
    metric_cols[1].metric("Hospitalization flags", f"{pd.to_numeric(filtered[hospitalized], errors='coerce').fillna(0).gt(0).sum():,}" if hospitalized else "—")
    metric_cols[2].metric("Amputation flags", f"{pd.to_numeric(filtered[amputation], errors='coerce').fillna(0).gt(0).sum():,}" if amputation else "—")
    st.markdown("<div class='notice'>OSHA SIR describes reported hospitalizations, amputations, or eye-loss cases and excludes fatalities. It is separate from the MSHA training cohort.</div>", unsafe_allow_html=True)
    if date_col and filtered["Reporting year"].notna().any():
        left, right = st.columns(2)
        with left:
            st.markdown("#### Records by year")
            st.bar_chart(filtered["Reporting year"].dropna().astype(int).value_counts().sort_index(), color="#9bc63b")
        with right:
            taxonomy_path = ROOT / "configs" / "precursor_keywords.json"
            taxonomy = load_json(str(taxonomy_path)) or {}
            counts = {}
            for text in filtered[narrative_col].fillna("").astype(str):
                for category in theme_matches(text, taxonomy):
                    counts[category] = counts.get(category, 0) + 1
            st.markdown("#### Suggested narrative themes")
            if counts:
                st.bar_chart(pd.Series(counts).sort_values(ascending=False).head(10), horizontal=True, color="#738b48")
            else:
                st.caption("No configured theme phrases matched this view.")
    st.markdown("#### Sample records")
    display = pd.DataFrame(index=filtered.index)
    display["Year"] = filtered["Reporting year"]
    for label, column in [("State", state_col), ("Event", pick_column(frame, "EventTitle", "Event")),
                          ("Nature", pick_column(frame, "NatureTitle", "Nature"))]:
        if column:
            display[label] = filtered[column]
    display["Narrative"] = filtered[narrative_col].fillna("").astype(str).str.slice(0, 500)
    st.dataframe(display.head(100), use_container_width=True, hide_index=True)
    st.caption("Showing up to 100 matching rows. Phrase categories are suggestions, not confirmed causes.")


model = None
if MODEL_PATH.is_file():
    try:
        model = get_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
    except Exception as exc:
        st.warning(f"Local model could not be loaded: {exc}")

profile = load_json(str(PROFILE_PATH))
st.markdown("""
<div class="hero">
  <div class="eyebrow">Public data · incident text · human review</div>
  <h1>SIF Lens</h1>
  <p>Explore incident wording, inspect a real-data severity baseline, and keep its evidence and limitations visible.</p>
</div>
<div class="notice"><b>Research prototype.</b> Not validated for operational safety decisions or prospective SIF screening.</div>
""", unsafe_allow_html=True)

section = st.radio("Project sections", ["Overview", "Review a report", "Performance", "OSHA explorer", "About & data"],
                   horizontal=True, label_visibility="collapsed")
if section == "Overview":
    render_overview(model, profile)
elif section == "Review a report":
    render_screen(model)
elif section == "Performance":
    render_performance(model)
elif section == "OSHA explorer":
    render_data_explorer()
else:
    st.markdown('<div class="eyebrow">Method and scope</div>', unsafe_allow_html=True)
    st.subheader("What this prototype can and cannot say")
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### Training source")
            st.write("Official MSHA Accident/Injury reports with narrative text and injury-degree codes. The trained label distinguishes codes 01/02 from selected lower-outcome codes.")
            st.write("The downloaded archive and model binary are built locally using the scripts in this repository and are excluded from Git.")
    with right:
        with st.container(border=True):
            st.markdown("#### Intended interpretation")
            st.write("A retrospective text-severity analogue for research. It is not a ground-truth SIF-potential label, future-risk probability, causal explanation, or worksite decision tool.")
    st.markdown("#### Project materials")
    st.markdown("- Dataset choice: `docs/dataset_selection.md`\n- Research landscape: `docs/research_landscape.md`\n- IEEE-style paper draft: `docs/sif_precursor_engine_research_paper.md`\n- Model card: `reports/model_card.md`")
