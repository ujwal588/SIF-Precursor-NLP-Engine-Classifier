from __future__ import annotations
import json, os
from pathlib import Path
import pandas as pd
import streamlit as st
from src.sif_engine.predictor import load_model, predict_report
from src.sif_engine.precursor_rules import analyze_precursor_cues

ROOT=Path(__file__).resolve().parent
RAW_DIR=ROOT/'data'/'raw'
MODEL_PATH=ROOT/'models'/'msha_severity_proxy.joblib'
st.set_page_config(page_title='SIF Precursor Explorer',page_icon='🦺',layout='wide')

@st.cache_data(show_spinner=False)
def load_keywords(name):
    with (ROOT/'configs'/name).open(encoding='utf-8-sig') as f: return json.load(f)
@st.cache_data(show_spinner='Loading public OSHA records…')
def load_data(path,mtime):
    frame=pd.read_csv(path,low_memory=False); frame.columns=[str(c).strip() for c in frame.columns]; return frame
@st.cache_resource(show_spinner=False)
def get_model(path): return load_model(Path(path))
def col(frame,*names):
    lookup={str(x).casefold():x for x in frame.columns}
    return next((lookup[x.casefold()] for x in names if x.casefold() in lookup),None)
def truthy(series): return pd.to_numeric(series,errors='coerce').fillna(0).gt(0)
def match_themes(text,taxonomy):
    normalized=str(text).casefold(); return [name for name,terms in taxonomy.items() if any(t in normalized for t in terms)]
def get_data_file():
    configured=os.getenv('SIF_DATA_PATH')
    candidates=[Path(configured)] if configured else sorted(RAW_DIR.glob('*.csv'))
    return next((p for p in candidates if p and p.is_file()),None)

st.title('SIF Precursor Explorer')
st.caption('Research prototype · Real public incident data · Human review required')

model=None
if MODEL_PATH.exists():
    try: model=get_model(str(MODEL_PATH))
    except Exception as exc: st.warning(f'Model artifact could not be loaded: {exc}')
else:
    st.warning('Train the MSHA baseline first: `python scripts/train_msha_model.py`.')

file=get_data_file()
data=None; source_name='not loaded'; narrative=None; flags=None; filtered=None
if file:
    try:
        data=load_data(str(file),file.stat().st_mtime); source_name=file.name
        narrative_col=col(data,'Final Narrative','Narrative','Description')
        date_col=col(data,'EventDate','Event Date'); state_col=col(data,'State')
        if narrative_col:
            narrative=data[narrative_col].fillna('').astype(str)
            flags=pd.DataFrame(index=data.index)
            for label,field in [('Hospitalized',col(data,'Hospitalized')),('Amputation',col(data,'Amputation')),('Loss of eye',col(data,'Loss of Eye','LossOfEye'))]:
                flags[label]=truthy(data[field]) if field else False
            data['Reporting year']=pd.to_datetime(data[date_col],errors='coerce').dt.year if date_col else pd.NA
            data['Outcome proxy']=flags.any(axis=1).map({True:'Severe outcome flag present',False:'No listed outcome flag'})
            precursor=load_keywords('precursor_keywords.json'); lsr=load_keywords('life_saving_rule_keywords.json')
            data['Narrative themes']=narrative.map(lambda x:match_themes(x,precursor))
            data['Possible LSR topics']=narrative.map(lambda x:match_themes(x,lsr))
            filtered=data.copy()
            with st.sidebar:
                st.header('OSHA SIR filters')
                if date_col and filtered['Reporting year'].notna().any():
                    years=sorted(filtered['Reporting year'].dropna().astype(int).unique())
                    selected=st.multiselect('Reporting year',years,default=years)
                    filtered=filtered[filtered['Reporting year'].isin(selected)]
                if state_col:
                    states=sorted(filtered[state_col].dropna().astype(str).unique())
                    selected_states=st.multiselect('State',states)
                    if selected_states: filtered=filtered[filtered[state_col].astype(str).isin(selected_states)]
                query=st.text_input('Narrative contains')
                if query: filtered=filtered[narrative.loc[filtered.index].str.contains(query,case=False,regex=False)]
    except Exception as exc: st.error(f'Could not load OSHA SIR file: {exc}')
else:
    st.info('Optional OSHA SIR file not found in data/raw/. The trained MSHA model can still be used.')

screen,performance,overview,themes,records=st.tabs(['Screen a report','Model performance','OSHA data overview','Patterns','Records'])
with screen:
    st.subheader('Single-report research screen')
    st.caption('The phrase panel surfaces literal hazard/control-gap language. The classifier compares the report with historical MSHA severity-coded narratives. Neither is a SIF-potential determination or future-risk probability.')
    report=st.text_area('Paste a safety report',height=170,placeholder='Describe the task, energy sources, controls, and what happened…')
    with st.expander('Optional incident context'):
        a,b,c=st.columns(3)
        context={'ACTIVITY':a.text_input('Activity'),'CLASSIFICATION':b.text_input('Classification'),'ACCIDENT_TYPE':c.text_input('Accident type')}
        d,e,f=st.columns(3)
        context.update({'MINING_EQUIP':d.text_input('Equipment'),'SUBUNIT':e.text_input('Work area'),'OCCUPATION':f.text_input('Occupation')})
    if st.button('Review report',type='primary'):
        if not report.strip():
            st.warning('Enter a report narrative first.')
        else:
            cues=analyze_precursor_cues(report)
            st.markdown('#### Explicit precursor-language cues')
            if cues:
                rows=[]
                for cue in cues:
                    rows.append({'Category':cue['category'],'Evidence':cue['evidence_level'],'Exposure phrases':', '.join(cue['exposure_matches']) or '—','Control-gap phrases':', '.join(cue['control_gap_matches']) or '—'})
                st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
            else:
                st.caption('No configured phrases matched. This does not mean the report is safe or that controls are present.')
            st.caption('These are deterministic phrase matches, not model predictions. A missing phrase is not evidence that a hazard or control gap is absent.')
            if model is not None:
                try:
                    result=predict_report(report,context,model)
                    st.markdown('#### Historical severity-proxy comparison')
                    if result['flag_for_review']: st.error('Flag for human review · resembles higher-outcome MSHA records')
                    else: st.info('Below the review threshold · this does not establish that the situation is safe')
                    score_col,threshold_col=st.columns(2)
                    score_col.metric('Uncalibrated model score',f"{result['screening_score']:.3f}")
                    threshold_col.metric('Validation-selected threshold',f"{result['threshold']:.3f}")
                    st.caption(result['score_semantics'])
                    st.markdown('##### Text cues that influenced the model score')
                    if result['explanation']:
                        st.dataframe(pd.DataFrame(result['explanation']).rename(columns={'term':'Text cue','direction':'Score effect','contribution':'Contribution'}),hide_index=True,use_container_width=True)
                    else: st.caption('No token-level explanation is available for this model.')
                    st.caption('Token contributions describe model behavior, not verified causes or hazards.')
                except Exception as exc: st.error(str(exc))
            else:
                st.info('Train the MSHA severity-proxy model to enable the historical comparison. The explicit phrase panel works without the model artifact.')
    st.info('Scope: the trained model uses real MSHA mining injury reports. It is not trained on Oil India near-miss/unsafe-condition labels and cannot establish SIF potential.',icon='ℹ️')
with performance:
    st.subheader('Held-out performance')
    if model:
        m=model['test_metrics']; c1,c2,c3,c4=st.columns(4)
        c1.metric('Test average precision',f"{m['average_precision']:.3f}")
        c2.metric('Recall',f"{m['recall']:.1%}")
        c3.metric('Precision',f"{m['precision']:.1%}")
        c4.metric('Specificity',f"{m['specificity']:.1%}")
        st.caption(f"Temporal test: {model['train_years'][0]}–{model['train_years'][1]} train; {model['validation_years'][0]}–{model['validation_years'][1]} validation; {2022}–{2025} held out. This is a retrospective outcome proxy, not a validated risk score.")
        st.write(f"Selected: **{model['model_name']}** with `{model['feature_set']}`. Confusion counts: TP {m['tp']}, FP {m['fp']}, TN {m['tn']}, FN {m['fn']}.")
        comparison=ROOT/'reports'/'model_comparison.csv'
        if comparison.exists(): st.dataframe(pd.read_csv(comparison),hide_index=True,use_container_width=True)
        with st.expander('Model card and limits'):
            card=ROOT/'reports'/'model_card.md'
            if card.exists(): st.markdown(card.read_text(encoding='utf-8'))
    else: st.warning('Train the model to populate measured performance.')
with overview:
    st.subheader('OSHA Severe Injury Report snapshot')
    if data is None or filtered is None: st.info('Put the OSHA CSV in data/raw/ to use this descriptive view.')
    else:
        st.caption(f'{source_name} · {len(filtered):,} of {len(data):,} records shown. This source is separate from the MSHA model-training corpus.')
        x,y,z=st.columns(3); x.metric('Records in view',f'{len(filtered):,}')
        y.metric('Hospitalizations',f"{flags.loc[filtered.index,'Hospitalized'].sum():,}")
        z.metric('Amputations',f"{flags.loc[filtered.index,'Amputation'].sum():,}")
        st.info('OSHA SIR includes only severe injury reports (hospitalization, amputation, or eye loss) and excludes fatalities; it is not the training set for the severity classifier.',icon='ℹ️')
        left,right=st.columns(2)
        with left:
            st.markdown('**Records by year**'); st.bar_chart(filtered['Reporting year'].dropna().astype(int).value_counts().sort_index().rename('Records'))
        with right:
            st.markdown('**Reported outcome flags**'); st.bar_chart(flags.loc[filtered.index].sum().rename('Records'))
with themes:
    if filtered is None: st.info('OSHA SIR file is needed for this descriptive view.')
    else:
        precursor_counts={}; lsr_counts={}
        for vals in filtered['Narrative themes']:
            for value in vals: precursor_counts[value]=precursor_counts.get(value,0)+1
        for vals in filtered['Possible LSR topics']:
            for value in vals: lsr_counts[value]=lsr_counts.get(value,0)+1
        left,right=st.columns(2)
        with left: st.markdown('**Narrative themes (keyword suggestions)**'); st.bar_chart(pd.Series(precursor_counts,dtype='int64').sort_values(ascending=False).head(12).rename('Records'))
        with right: st.markdown('**Possible Life-Saving Rule matches**'); st.bar_chart(pd.Series(lsr_counts,dtype='int64').sort_values(ascending=False).head(12).rename('Records'))
        st.caption('Keyword matches can miss context and produce false matches. Review original records; do not interpret these as confirmed causes.')
with records:
    if filtered is None: st.info('OSHA SIR file is needed for this descriptive view.')
    else:
        st.caption('Employer names, addresses, and coordinates are omitted from this table.')
        display=pd.DataFrame(index=filtered.index); display['Year']=filtered['Reporting year']
        for label,field in [('State',col(data,'State')),('Event',col(data,'EventTitle','Event')),('Nature',col(data,'NatureTitle','Nature'))]:
            if field: display[label]=filtered[field]
        display['Outcome proxy']=filtered['Outcome proxy']; display['Narrative']=narrative.loc[filtered.index].str.slice(0,700)
        display['Narrative themes']=filtered['Narrative themes'].map(lambda x:', '.join(x) or '—')
        display['Possible LSR topics']=filtered['Possible LSR topics'].map(lambda x:', '.join(x) or '—')
        st.dataframe(display,use_container_width=True,hide_index=True)


