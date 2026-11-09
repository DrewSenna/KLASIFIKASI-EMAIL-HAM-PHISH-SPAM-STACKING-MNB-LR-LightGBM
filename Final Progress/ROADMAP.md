# Roadmap — M03 to UTS (Milestone 1)

**Today:** Mon 2 Nov 2026. **UTS window:** 9–14 Nov 2026. **Final defense:** 6 Jan 2027.
Writing rules: `WRITING_STYLE.md`. Defense notes: `M0x/DECISIONS_M0x.md` (why each choice, with sources) and `HOW_TO_READ_RESULTS.md`. Each meeting's checklist comes from the "OUTPUT PROJECT" slide at the end of its weekly deck.

## Fixed protocol (all meetings)

- Same cleaning, giving 278,941 messages, and the same 80/20 stratified split with `random_state=42`.
- **The test set is used only twice:** for the M03 baseline and for the final evaluation of the chosen model. All comparisons use the training data only, via a validation split (M04) or stratified 5-fold CV (M05 onward).
- Primary metric: **Macro F1**. Business metric: **Phish Recall**. Also report Phish Precision, Ham → Phish false alarms, and training time.
- Selection rule: highest Macro F1; if the gap is < 0.001, pick the simpler option (fewer features, or faster).
- Every resampling step and every vectorizer sits inside a `Pipeline`, so it is fit only on the training part of each fold.

## Plan per meeting

| Meeting | Slide checklist | What we run | Decision it produces | Status |
|---|---|---|---|---|
| **M03** (W03) | Problem, dataset, preprocessing, baseline, initial evaluation, findings | Baseline TF-IDF (full vocabulary) + MNB; EDA | Baseline numbers; hypotheses H1–H3 | **Run 2 Nov.** Test: Acc 0.8881, Macro F1 0.8377, Phish Recall 0.5316 (Phish → Spam 3,475) |
| **M04** (W04) | Baseline + ≥2 candidates, comparison table, 3–5 findings | Exp 1: MNB representation (TF-IDF vs count → vocabulary cap → bigram). Exp 2: best MNB vs Logistic Regression vs Decision Tree | Shortlist of the 2 best algorithms for M05; verdict on H2 | **Done (re-run 3 Nov).** Validation: LR + TF-IDF Macro F1 **0.9647** (Phish R 0.932 / P 0.967); MNB + count uni+bigram 0.9443 (Phish R **0.960** / P 0.882); DT 0.9022; baseline 0.8286. Repeated configs reproduced run 1 exactly |
| **M05** (W05) | Validation strategy + no-leakage proof; ≥1 ensemble; imbalance (ratio, metrics, techniques); results as mean ± std; error analysis | 5-fold CV on shortlist; ensemble (LightGBM on sparse TF-IDF, or Random Forest on chi²-reduced features); imbalance table (format: `Class Materials/M05 Data Imbalance Table Reference.jpeg`): none / class weight or `fit_prior=False` / random oversampling / **SMOTE** / undersampling / threshold; tune `alpha` or `C` | Final algorithm + imbalance technique; verdict on H3 and on SMOTE | **Done 3 Nov** (Colab ~3.5 h; LightGBM alone 119 min). 5-fold CV: MNB α=1.0 0.9441; LR C=64 0.9687; LightGBM 0.9773; Stack MNB+LR+LGBM **0.9807**. No imbalance technique kept (SMOTE −0.007). At Ham Recall 99.5%: **Stacking MNB+LR+LightGBM** wins (Macro F1 0.9784, b=1.514; 511 Ham lost of 102,192, Phish→Ham 86, Spam→Ham 2,146) vs MNB alone 0.9292 (Spam→Ham 10,202) |
| **M06** (W06) | Clustering (elbow/silhouette, K, sizes, profiles); dimensionality reduction (scree, 2D plot, with vs without) | TruncatedSVD (PCA does not work on sparse text) → K-Means on a sample; profile clusters by top words + label mix; LR with vs without SVD under CV | Whether SVD helps (expected: no, but we test it); whether clusters show source/label structure (supports Limitations) | **Done 3 Nov.** Silhouette best 0.091 (K=4): no strong structure. Clusters follow sources (Enron Ham; a pure-Phish cluster of 17,281 emails ≈ half of Phish, built on header timestamps `2008 aug 06 wed`). SVD hurts LR (−0.0213 Macro F1), so it is not used. H4 stated |
| **M07** (W07) | Feature selection; bias-variance; leakage check; error analysis on individual samples; reproducibility | chi² SelectKBest inside the pipeline; learning curve; near-duplicate check between train and validation; read individual false negatives/positives; model card, experiment log, `requirements.txt` | Final feature set; documented limitations | V2-only run done 4 Nov (`V2 ONLY - RUN FINISHED`). **V3 run done 4 Nov (`V3 - RUN FINISHED R1`)**: H4 and H5 supported (V2→V3 MNB −0.088, LR −0.077). Tie-break changed to business priorities → final model **Stacking MNB + LR + LightGBM, b = 1.777**, Macro F1 0.8855 at Ham Recall 99.5% (511 Ham lost, 582 Phish → Ham). **R2 done 4 Nov (`V3 - RUN FINISHED R2`): final notebook, model B saved to Drive** |
| **Final evaluation** | Guideline §13–14: final selection on evidence, one-time test evaluation, interpretation | Loads M07's `final_config.json` and `oof_log_proba.npz` from Drive (CV not recomputed): meta-model on M07 OOF, b from M07, final base models trained once on all training data, one test evaluation vs the M03 baseline, bootstrap CIs, meta weights, LR top words, examples (~1 h) | Test-set performance of the chosen model; final conclusions and limitations | **Done 5 Nov.** Test: Macro F1 **0.8913** (95% CI 0.8878–0.8946) vs baseline 0.8377 (0.8341–0.8416); Ham Recall 0.9963; Ham lost 94/25,548 (baseline 174); Phish → Ham 107 (baseline 538); Spam → Ham 781 (baseline 1,991). CV vs test +0.0058. See `Final Evaluation/DECISIONS_FINAL.md` |
| **UTS** | Guideline §8: 12 minimum items | Consolidated results + report (CRISP-DM chapters that cover all 17 sections of guideline §17) | Milestone 1 submission | Plan |

## Hypotheses (from M03 F.3)

| ID | Hypothesis | Tested in | Result |
|---|---|---|---|
| H1 | Phish and Spam share much vocabulary (offers, links) | M04 error table, M07 error analysis | Partly consistent: LR (learns per-word weights) cuts Phish → Spam from 2,858 to 458 on validation |
| H2 | Unigrams miss phrases such as *verify your account* | M04 D.3 (bigram) | **Supported** (M04 D.4, no cap): uni+bigram counts +0.0101 Macro F1 over unigram (0.9443 vs 0.9342); Phish Precision +0.024, Phish Recall −0.023 |
| H3 | The smaller Phish class pulls predictions toward larger classes | M05 imbalance table | Baseline signature fits (Phish P 0.98 / R 0.51), but the main cause for TF-IDF+MNB is smoothing: Phish's total TF-IDF weight is only ~0.1–0.2× α·\|V\| (M04 D.3, full data: Phish/smoothing ratio 0.31; α=0.1 keeping all words +0.0619 Macro F1 vs cap +0.0675). Equalising the prior (`fit_prior=False`) helped only slightly (+0.006 Macro F1, local sample). M05: no balancing technique raises Phish Recall by >0.01 (class weight +0.0044, ROS −0.0027) → **not supported**; SMOTE lowers MNB Macro F1 by 0.007 |
| H4 | The model recognises Phish by its source's number format rather than by content (Phish source kept real digits; other sources replaced numbers with `escapenumber`) | Found in M06 (clusters + source table); tested in M07 by normalising digits and `escape…` tokens to one token | 15% local sample: `escape…` in 0.3% Phish vs 63–71% Ham/Spam; digits in 93% Phish vs 14–17%; normalising numbers drops MNB Macro F1 0.939 → 0.914. **Full data (M06):** `escape…` Ham 75.1% / Phish 0.3% / Spam 67.0%; digits 12.2% / 93.4% / 10.2%. Half of Phish sits in one timestamp-header cluster, so date words (`aug`, `wed`) are part of the shortcut too. Test pending (M07) |
| H5 | The model recognises Phish by its source's other formatting: stopwords removed ("the" in 0% of Phish vs 71% Ham / 57% Spam) and punctuation removed without spaces (glued links in 38% of Phish, 0% others) | Found 4 Nov while reading M07's error examples (a stopword-free Ham email predicted Phish; chi²/LightGBM top features = the, to, of, and); tested in M07 (V3 run) | **Supported:** V2 → V3 MNB −0.088, LR −0.077 Macro F1; Phish Recall drops to 0.73 / 0.68 |

## Decision log

| # | Decision | Reason | Where |
|---|---|---|---|
| 1 | TF-IDF without a vocabulary cap | Don't discard information without evidence; MNB fits in memory | M03 D.1 |
| 2 | Test set held out; M04 compares on a validation split | Avoid choosing models on the test set | M04 B.1 |
| 3 | Logistic Regression is a candidate; if it wins, it replaces MNB | Taught in W04; standard text classifier | M04 E.1 |
| 4 | KNN excluded | Prediction cost over 178k × ~700k dims; distances lose meaning in very high dimensions | M04 E.1 |
| 5 | Decision Tree: `max_depth=20`, `min_df=5` | Training cost grows with vocabulary size × tree nodes | M04 E.2 |
| 18 | Manual NB challenge (M04) re-verified on the M03 split; test set used only to score it, nothing selected from it | Same split as the baseline makes the manual score comparable; 9-step equivalence check vs scikit-learn `MultinomialNB` proves correctness | `Manual Naive Bayes/`, run 8 Nov: 9/9 checks pass, 0/55,789 predictions differ; baseline reproduced (split identical); library count-MNB = manual in every metric; test Macro F1 0.9337 vs library TF-IDF baseline 0.8377 (gap = representation), Ham lost 430 vs 174. Report §7.5 |
| 17 | Test evaluation of model B accepted as the final result; no setting changed after the test | Better than the M03 baseline on all three business priorities; CIs do not overlap; test ≈ CV | Final Evaluation, 5 Nov |
| 16 | Final Evaluation loads M07's saved settings and OOF predictions from Drive instead of recomputing them | Same numbers as M07 by construction; ~1 h instead of ~4.5 h; H5 already tested in M07 | User decision 4 Nov |
| 15 | Final model = **Stacking MNB + LR + LightGBM**, b = 1.777, Macro F1 0.8855 at Ham Recall 99.5% (CV); 511 Ham lost, 582 Phish → Ham, 3,089 Spam → Ham. MNB α = 2.0, LightGBM 125 trees | Macro F1 tie with LR + LightGBM (0.8847, Δ 0.0008). Tie-break changed from 'fewer base models' to the business priorities: B lets 164 fewer phishing (bootstrap 129–198) and 456 fewer spam (394–524) into the inbox, with the same 511 Ham lost | M07 F.3; detailed A/B comparison in DECISIONS_M07 2b |
| 14 | Final text = V3: V2 + NLTK stopword list (179 words) removed from every class + link unification (the Phish source contains 0.00% NLTK stopwords but still has longer-list words such as `please`, so NLTK is the list the source used) | The Phish source had stopwords and punctuation stripped; leaving them lets the model detect format, not content | Final Evaluation A.3 |
| 13 | Final model = Stacking MNB + LR + LightGBM on V2 text, b = 1.625 (Ham Recall 99.5%), Macro F1 0.9551 (CV) | Highest Macro F1 at the Ham target; MNB adds +0.0018 and 81 fewer phishing emails reaching the inbox vs LR + LightGBM | M07 F.3 |
| 12 | Final preprocessing = V2: punctuation → space, remove email-header timestamps, numbers/`escape…` → `numtoken`, month/weekday names → `datetoken` (not `may`) | The source-format shortcut (H4) would not exist on real email; rule fixed before seeing results. The first design (no timestamp removal) still leaked via `datetoken` | M07 B.1 |
| 11 | No dimensionality reduction in the final model | LR with SVD 200: −0.0213 Macro F1 vs no reduction (M06 F) | M06 F |
| 10 | Final candidate: Stacking (meta-LR) over MNB count uni+bi (α=1) + LR TF-IDF (C=64) + LightGBM, Ham bias b=1.514 → Ham Recall 99.5% on OOF | Highest Macro F1 at the Ham target; least phishing/spam leakage at every Ham-loss level (M05 G.2 chart). Pending: H4 re-test in M07, MNB-ablation of the stack, final test-set check | M05 G.3 |
| 9 | SMOTE and other imbalance techniques dropped | None improves Macro F1 without losing more Ham (M05 E.3) | M05 E.4 |
| 8 | Business priority: no legitimate email lost. Accepted error ("poison"): Spam → Ham. Final model chosen at Ham Recall ≥ 99.5% (target proposed by Claude; user ran M05 with it) | User decision 3 Nov; matches M02 success criterion (very low false positives) | M05 B |
| 7 | LR + TF-IDF and MNB + count uni+bigram go to M05 | Top-2 Macro F1 on validation. LR wins overall (fewer Spam → Ham / Spam → Phish errors); MNB has higher Phish Recall and loses fewer Ham emails (Ham → Spam 174 vs 421) | M04 G.1 |
| 6 | SMOTE is tested, not assumed | W05: compare against cheaper class weighting; resample only inside CV folds | M05 |

## Open items for the team

- Exact UTS submission date and format (notebook only, or notebook + report PDF?).
- Final direction: started as MNB + TF-IDF + SMOTE, and is now decided by M04/M05 evidence.
- **How `max_features=50_000` picks words:** sklearn ranks words by total count across all training messages (raw counts, even inside TfidfVectorizer) and keeps the top 50,000. It ranks by total occurrences, not by how many messages contain the word.
- Expected Colab runtime: M03 ≈ 10–15 min; M04 ≈ 30–45 min (bigram and Decision Tree are the slowest cells).
