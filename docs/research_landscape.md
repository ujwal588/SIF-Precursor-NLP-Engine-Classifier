# Research landscape and comparison

## Public SIH 2026 implementations reviewed

The repositories reviewed are implementation references, not independent benchmark studies. README claims and demo scores are not comparable unless the dataset, label process, split unit, date window, and evaluation threshold match.

| Public project | Data approach described by repository | Useful idea | Limitation for comparison |
|---|---|---|---|
| [Ppoloju/SIH-SIF-Precursor-Detection](https://github.com/Ppoloju/SIH-SIF-Precursor-Detection) | Synthetic demo path plus scripts for assembling public MSHA, OSHA, and BSEE sources | Preserve source provenance; show narrative evidence; emphasize recall | Rule-derived labels and heterogeneous source cohorts are not expert PSIF labels; repository demo metrics are not comparable to this locked split |
| [MnSykia/Sentinel_SIF](https://github.com/MnSykia/Sentinel_SIF) | Synthetic development examples; high-energy exposure plus direct-control failure framing | Separate hazard energy from barrier/control status | Synthetic examples help demonstrate workflow but do not establish real-world accuracy |
| [prasanna456186/sif-sentinel](https://github.com/prasanna456186/sif-sentinel) | Data-ingestion/handoff work; modeling described as deferred | Modular ingestion and source normalization | No trained classifier benchmark in the reviewed state |
| [shlok926/oil-safety-intelligence-platform](https://github.com/shlok926/oil-safety-intelligence-platform) | Documentation/placeholder platform | Clear SIH problem framing and workflow | Not an implemented, evaluated model in the reviewed state |
| [Predictive-Safety-Intelligence/sih26165-sif-precursor-engine](https://github.com/Predictive-Safety-Intelligence/sih26165-sif-precursor-engine) | Small reported TF-IDF/One-vs-Rest experiment with case-grouped evaluation | Group related reports to reduce leakage | The repository reports a small case count; data/labels differ from the MSHA outcome proxy here |
| [Kirthik-Roshan/SIF-Guard](https://github.com/Kirthik-Roshan/SIF-Guard) | Synthetic demo records | Demonstration UX and transparent workflow | Synthetic test data cannot establish performance on operational incident narratives |
| [ujjvaltiwari-arch/sif-precursor-detection](https://github.com/ujjvaltiwari-arch/sif-precursor-detection) | Synthetic examples and baseline architecture | Simple end-to-end architecture | Synthetic labels and no matched external benchmark |

Repository descriptions can change; this summary records the contents reviewed on 2026-11-23. See each linked repository for the current implementation.

## Published research context

Prior work includes TF-IDF/SVM, CNN, hierarchical attention networks, word embeddings/BERT, transformer-plus-tree models, and keyword/rule systems. Published results differ in task and label semantics. For example, Baker et al. study injury outcomes and precursor extraction from a private oil-and-gas incident corpus; Parikh et al. study PSIF labels in a private corporate dataset with expert-informed weak labeling. Their reported scores cannot serve as direct baselines for an MSHA fatality/permanent-disability outcome proxy.

## This project’s measured baseline

The selected model is TF-IDF plus Logistic Regression over a combination of masked MSHA narrative text and selected structured context fields. The test period is 2022–2025 and is untouched for model selection. Outcome vocabulary is removed before inference and training for the selected feature view; this reduces direct leakage but cannot remove all clues that a report was written after an injury.

| Test statistic | Result |
|---|---:|
| Average precision | 0.3995 (95% bootstrap CI 0.3472–0.4601) |
| Positive prevalence | 0.0158 |
| Recall | 0.5565 |
| Precision | 0.3177 |
| F1 / F2 | 0.4045 / 0.4838 |
| Specificity | 0.9809 |

The always-negative average-precision baseline is 0.0158. This result is useful as a reproducible retrospective severity-text experiment; it is not evidence of performance on Oil India, near misses, or future SIF risk. The validation-selected F2 threshold produces false positives, so the interface requires human review and displays the uncalibrated decision score rather than calling it a probability.

## Improvement opportunities and project contribution

1. **Real public corpus instead of synthetic-only training.** Use a reproducible official MSHA download and preserve raw data outside Git.
2. **Chronological holdout.** Train through 2018, validate 2019–2021, and reserve 2022–2025 as the test period rather than randomly mixing time.
3. **Transparent scope and negative class.** Document precisely which injury codes form the retrospective binary target and report class prevalence.
4. **Leakage-aware experiments.** Compare raw narrative only as a diagnostic, mask direct outcome terms for the selected model, and explicitly state residual leakage.
5. **Rare-class evaluation.** Report average precision, recall, precision, F2, specificity, confusion counts, prevalence, and bootstrap interval rather than accuracy alone.
6. **Two outputs kept separate.** The learned model estimates similarity to historical severity-coded MSHA reports. A separately labeled, deterministic phrase matcher shows explicit hazard exposure and direct-control-gap language without pretending the phrases are learned causal explanations.
7. **Human-review interface.** Show matched terms and model token contributions, keep review thresholds explicit, and avoid operational safety recommendations.

These are engineering and evaluation improvements for a university prototype, not a claim of a new state-of-the-art SIF model. The next scientifically meaningful improvement is a sufficiently large, independently expert-labeled near-miss/unsafe-condition set from the intended industry, plus agreement measurements between reviewers.

## References (IEEE style)

[1] H. Baker, M. R. Hallowell, and A. J.-P. Tixier, “Automatically learning construction injury precursors from text,” *Automation in Construction*, vol. 118, Art. no. 103145, 2020, doi: 10.1016/j.autcon.2020.103145.

[2] P. Parikh, J. Penfield, and M. Juaire, “Automatic identification of incidents involving potential serious injuries and fatalities (PSIF),” *Scientific Reports*, vol. 14, Art. no. 8091, 2024, doi: 10.1038/s41598-024-58824-y.

[3] D. M. Goldberg, “Characterizing accident narratives with word embeddings: Improving accuracy, richness, and generalizability,” *Journal of Safety Research*, vol. 80, pp. 441–455, 2022, doi: 10.1016/j.jsr.2021.12.024.

[4] M. Z. F. Khairuddin *et al*., “Predicting occupational injury causal factors using text-based analytics: A systematic review,” *Frontiers in Public Health*, vol. 10, Art. no. 984099, 2022, doi: 10.3389/fpubh.2022.984099.

[5] A. Bayona, M. R. Hallowell, and S. Bhandari, “The things that hurt people are not the same as the things that kill people: Key differences in the proximal causes of low- and high-severity construction injuries,” *Journal of Construction Engineering and Management*, vol. 150, no. 8, Art. no. 04024089, 2024, doi: 10.1061/JCEMD4.COENG-14545.

[6] R. Liu, H.-C. Liu, H. Shi, and X. Gu, “Occupational health and safety risk assessment: A systematic literature review of models, methods, and applications,” *Safety Science*, vol. 160, Art. no. 106050, 2023, doi: 10.1016/j.ssci.2022.106050.

[7] U.S. Mine Safety and Health Administration, “MSHA open government data: Accident/injury data sets.” [Online]. Available: https://arlweb.msha.gov/opengovernmentdata/ogimsha.asp. Accessed: Nov. 23, 2026.

[8] U.S. Occupational Safety and Health Administration, “Severe injury reports.” [Online]. Available: https://www.osha.gov/severe-injury-reports. Accessed: Nov. 23, 2026.

[9] U.S. Occupational Safety and Health Administration, “Injury Tracking Application data.” [Online]. Available: https://www.osha.gov/itadata. Accessed: Nov. 23, 2026.

[10] Ppoloju, “SIH-SIF-Precursor-Detection,” GitHub repository. [Online]. Available: https://github.com/Ppoloju/SIH-SIF-Precursor-Detection. Accessed: Nov. 23, 2026.

