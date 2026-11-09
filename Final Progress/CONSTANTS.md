# Every Numerical Constant in the Project, and Where It Comes From

This file lists every number that the final model, or a decision about it, depends on. For each one it says **where it is used**, **how it was set**, and **the evidence**. Numbers that are only results (scores, counts) are not listed, unless a decision was based on them.

**How each value was set:**

| Code | Meaning |
|---|---|
| **TUNED** | Chosen by a validation curve or grid search on cross-validation (training data only) |
| **TESTED** | Chosen by comparing a few options in an experiment |
| **DATA** | Measured from the data |
| **RULE** | A selection rule or threshold fixed before the results were seen |
| **CONV** | A standard convention or library default, not tuned |
| **PRACT** | Set for a practical reason (memory, time), not tuned |

Section 7 lists the values that were **not re-tuned** on the final V3 text, because a defense question may ask about them.

---

## 1. Data and splitting

| Constant | Value | Used in | Set by | Reason and evidence |
|---|---|---|---|---|
| Rows after cleaning | 278,941 | M03 → all | DATA | 365,448 rows → minus 3 empty texts, minus 44 rows of 14 texts with conflicting labels, minus 86,460 duplicates (M03 C.2) |
| Test share | 20% (`test_size=0.20`) | M03 → Final | CONV | Common 80/20 split. Gives 55,789 test emails, enough for tight intervals: test Macro F1 95% interval width ≈ 0.007 (Final E.3) |
| Training / test emails | 223,152 / 55,789 | all | DATA | Result of the 80/20 split |
| `stratify` | by label | M03 → all | CONV | Phish is the smallest class (15.4%), so stratifying keeps the class proportions equal in training and test (W05 p. 41) |
| `random_state` | **42** | everywhere (split, folds, models, sampling) | CONV | Any fixed number makes runs repeatable. 42 is a common convention and is used everywhere, so every meeting uses the same split and folds |
| M04 validation share | 20% of training (44,631 emails) | M04 | CONV | Hold-out validation before cross-validation was introduced in M05 |
| Number of folds | **5** (`StratifiedKFold`, shuffled) | M05 → M07 | CONV | Standard choice. Each fold has ~44,600 validation emails, so fold-to-fold SD is ≤ 0.0011 Macro F1 (M05). 10 folds would double the runtime of the multi-hour runs for little gain |

---

## 2. Text cleaning (V3, the final text)

| Constant | Value | Set by | Reason and evidence |
|---|---|---|---|
| Stopword list | **NLTK English, 179 entries = 153 unique words without apostrophes** | DATA | The Phish source contains **0.00%** NLTK stopwords, but still has words from longer lists (e.g. `please`, 825 times in a 5% sample). So NLTK is exactly the list the Phish source removed. scikit-learn's longer list would also delete content words (`please`, `call`, `system`) (M07 1.1b) |
| Web parts deleted | `www`, `com`, `net`, `org`, `edu`, `gov` | DATA | These appear as separate words only in Ham and Spam. In Phish they are glued inside links (38–39% of Phish has glued links, 0% of the others) (M07 B.1) |
| `may` excluded from date words | — | PRACT | "may" is a common English verb, so it is not replaced by `datetoken` |
| Timestamp pattern digits | day 1–2 digits, year 4, clock 3–6 | DATA | Header timestamps look like `tue 05 aug 2008 193121 0400` (M07 1.1). The `escape…` placeholder is accepted in every number slot, because Ham and Spam timestamps had their numbers replaced |
| Whitespace normalisation | all runs of whitespace → one space | CONV | Applied before deduplication (M03) |

---

## 3. The final model

### 3.1 Multinomial Naive Bayes

| Constant | Value | Set by | Reason and evidence |
|---|---|---|---|
| Representation | word **counts**, not TF-IDF | TESTED | M04 D.1: counts 0.9323 vs TF-IDF 0.8286 Macro F1. MNB models how often words occur; with TF-IDF, the smoothing total (α × 664,008 words) outweighs Phish's total TF-IDF mass (207,178) (M04 D.3) |
| n-grams | unigram + **bigram** | TESTED | M04 D.4: +0.0101 Macro F1 (0.9443 vs 0.9342). Hypothesis 2 supported |
| `min_df` | **2** | PRACT | The number of distinct word pairs is huge. Keeping pairs seen in ≥ 2 messages saves memory; both sides of the test used it (M04 1.5). Gives ~3.3 million features per fold on V3 |
| `alpha` | **2.0** | TUNED | M07 D.1 validation curve on V3, 5-fold CV: 0.25 → 0.8169, 0.5 → 0.8201, 1 → 0.8283, **2 → 0.8356**, 4 → 0.8251, 8 → 0.7811. The best value is inside the range. Rule: highest Macro F1; ties (< 0.001) go to higher Ham Recall. No tie here |
| chi-square `k` | **none (all words)** | TUNED | M07 E.1: k = 10k 0.7492, 50k 0.7912, 200k 0.8167, 1M 0.8330, all 0.8356. Rule: the smallest k within 0.001 of "all"; 1M is 0.0026 below, so all words are kept |

### 3.2 Logistic Regression

| Constant | Value | Set by | Reason and evidence |
|---|---|---|---|
| Representation | TF-IDF unigram, `sublinear_tf=True`, full vocabulary | TESTED / CONV | M04: LR + TF-IDF was the strongest candidate (0.9647). Sublinear TF stops a word repeated thousands of times from dominating: some emails are 1.5 million words long (M03 B.6) |
| `C` | **64** | TUNED (M05, V0 text) | M05 grid 1, 4, 16, 64 → best 64, at the top edge, but gains were flattening (+0.0005 from 16 to 64). **Not re-tuned on V3** (section 7) |
| `max_iter` | 1000 | PRACT | Gives the solver enough iterations to converge on hundreds of thousands of features (M04 parameter table). scikit-learn's default is 100 |

### 3.3 LightGBM

| Constant | Value | Set by | Reason and evidence |
|---|---|---|---|
| **Number of trees** | **125** | TUNED (M07, V3) | M07 D.2: one model on fold 1 (V3), scored every 25 trees. Validation Macro F1 0.8446 (25), 0.8668 (50), 0.8756 (75), 0.8816 (100), **0.8875 (125)**, 0.8868 (150). Rule: the **smallest** number of trees within 0.001 of the best validation score, because fewer trees train faster. 125 is the best itself; 150 is slightly lower, so the curve has levelled off. The train–validation gap at 125 is 0.029 (0.9161 vs 0.8875), a moderate gap |
| Maximum trees tested | 150 | PRACT | M05's value. Because 125 beat 150, the optimum is inside the range, unlike M05 (150 at the edge) |
| Curve training sample | 20,000 training emails | PRACT | Scoring the training curve on all ~178,000 emails after every 25 trees would be slow; 20,000 gives a precise estimate |
| `learning_rate` | **0.2** | TESTED (M05, 15% sample, V0) | Pairs with ~150 trees: same quality as 300 trees at 0.1 (0.9677 vs 0.9675) in half the time. Not re-tuned on V3 |
| `num_leaves` | **63** | TESTED (M05, 15% sample, V0) | 31 leaves scored lower (0.9657). Not re-tuned on V3 |
| `colsample_bytree` | **0.3** | TESTED (M05, 15% sample, V0) | Each tree sees 30% of the words: faster, and the trees are more varied. Not re-tuned on V3 |
| Features | unigram counts in **≥ 20** messages | TESTED (M05) | `min_df` = 20 and 5 gave identical results; 20 uses fewer features |
| Probability floor | 1e-15 | PRACT | LightGBM can output a probability of exactly 0, and log(0) is −∞. The floor keeps log-probabilities finite |

### 3.4 Stacking (meta-model)

| Constant | Value | Set by | Reason and evidence |
|---|---|---|---|
| Base models | MNB + LR + LightGBM | RULE (M07 F.3) | Tied on Macro F1 with LR + LightGBM (0.8855 vs 0.8847, difference < 0.001). The tie was broken by business priority: 164 fewer phishing and 456 fewer spam emails in the inbox (bootstrap 95% intervals: 129–198 and 394–524), with the same 511 Ham lost |
| Log-probability clip | **[−30, 0]** | PRACT (M05) | MNB's log-probabilities reach the thousands of negative units. Without a limit, MNB alone would dominate the meta-model's inputs. A probability of e^−30 ≈ 10^−13 already means "impossible" |
| Scaling | `StandardScaler` | CONV | Puts all nine inputs on the same scale, so the meta-model's weights can be compared (Final F.1) |
| Meta-model | Logistic Regression, default C = 1, `max_iter=1000` | CONV | A simple, well-behaved combiner with nine inputs; not tuned |

### 3.5 The Ham threshold

| Constant | Value | Set by | Reason and evidence |
|---|---|---|---|
| **Ham Recall target** | **99.5%** | RULE (M05, user decision) | Business priority 1, "no legitimate email lost". Our M02 success criterion asks for very low false positives. 99.5% = at most 1 in 200 legitimate emails lost |
| **b** | **1.777** | DATA (M07) | The smallest value that, added to log P(Ham), gives Ham Recall ≥ 99.5% on the final stack's out-of-fold predictions (M07 F.3). Meaning: an email leaves the inbox only if Spam or Phish is at least e^1.777 ≈ **5.9×** more likely than Ham. Equivalently, it counts as Ham if P(Ham) ≥ 0.169 × the largest other probability. On the test set it gave Ham Recall 0.9963 (Final E.1) |
| Earlier b values | 1.514 (M05, V0), 1.625 (M07 run 1, V2), 1.926 (stack without MNB) | DATA | The same rule on other models or text versions; not used in the final model |

---

## 4. Selection rules and hypothesis thresholds

| Constant | Value | Used in | Set by | Reason |
|---|---|---|---|---|
| **Tie tolerance** | **0.001 Macro F1** | M04 → M07 (all selections) | RULE | About one fold-to-fold standard deviation (≤ 0.0011 in M05). A smaller difference cannot be told apart from noise, so it counts as a tie and a secondary criterion decides. Written in each notebook before the results |
| Tie-breaks | M04: fewer features. M07 α: higher Ham Recall. M07 trees and chi-square k: smaller value. M07 final stack: Phish → Ham, then Spam → Ham, then fewer base models | per decision | RULE | Each tie-break prefers either the business priority or the simpler and faster option |
| **Hypothesis 4 and 5 threshold** | Macro F1 drop **> 0.005** for either model | M07 B.3 | RULE | About 5× the fold-to-fold standard deviation, so a drop this large is not noise. Results: H4 −0.044 / −0.026, H5 −0.088 / −0.077. Both are far above the threshold |
| "Use SVD?" | only if it costs < 0.001 Macro F1 | M06 F | RULE | Same tie logic. Result −0.0213, so SVD is not used |
| Phish-word ranking: minimum share | a word must appear in ≥ **1%** of Phish emails | M07 B.4 | PRACT | Without it, the top 25 was filled with words from single CNN-template emails |
| Top words shown | 25 (B.4), 15 (B.5, Final F.2), 20 (chi-square, gain) | M07, Final | CONV | Enough to see a pattern, short enough to read |
| "Template" coverage threshold | < 20% of Phish | M07 B.4 text | RULE | If the top-25 words cover under 20% of Phish (result: 11.5%), they come from a repeated template, not a general Phish pattern |

---

## 5. Analysis settings (they do not change the final model)

### 5.1 M06 clustering and dimensionality reduction

| Constant | Value | Set by | Reason |
|---|---|---|---|
| TF-IDF `min_df` | 5 | PRACT | A word in fewer than 5 messages cannot define a group; this also saves memory for SVD |
| SVD components fitted | 200 | PRACT | The slides' 90–95% variance rule is not feasible for text: it would need thousands of components. 200 explain 21.8% |
| Components used for clustering | 100 | CONV | A common LSA size; they explain 16.7% of the variance |
| K range | 2 – 12 | CONV | Covers the 3 true classes with room on both sides |
| K chosen | **4** | TUNED | Highest silhouette (0.0906) of K = 2–12 |
| K for the detailed profile | 7 | DATA | Highest NMI (0.319); used only as a check |
| Silhouette threshold | 0.25 | CONV | Kaufman & Rousseeuw (1990): ≤ 0.25 means "no substantial structure". Ours is 0.09 |
| Silhouette sample | 10,000 emails | PRACT | The full silhouette needs ~50 billion distances |
| K-Means `n_init` | 3 | PRACT | Three random starts per K, to keep 11 values of K within time |
| Ward sample | 5,000 emails | PRACT | Ward's memory grows with the square of the number of emails |
| t-SNE | 6,000 points, perplexity 30 | CONV / PRACT | 30 is the standard default; 6,000 points keeps the plot readable and fast |
| SVD sizes tested for classification | 50, 100, 200 | CONV | Three sizes from small to the largest fitted (best: SVD 200, 0.9443 vs no reduction 0.9655) |
| LR `C` in M06 | 1 | CONV | Default. Only the relative difference with and without SVD matters |

### 5.2 M07 diagnostics

| Constant | Value | Set by | Reason |
|---|---|---|---|
| Near-duplicate threshold | cosine similarity **≥ 0.9** | CONV | A common cut-off for "near-duplicate" text. The notebook shows the full similarity histogram, so other cut-offs can be read off. Result: 55.2% |
| "Nearly identical" threshold | ≥ 0.99 | CONV | A second, stricter cut-off for practically identical texts |
| Near-duplicate sample | 3,000 validation emails (fold 1) | PRACT | All pairs would be ~8 billion similarity calculations per fold; 3,000 gives the proportion within about ±2% |
| Learning-curve fractions | 10%, 25%, 50%, 100% of fold 1's training data | CONV | Roughly doubling steps that show whether the curve is still rising |
| Learning-curve training sample | 20,000 emails | PRACT | Same reason as the tree curve |
| Message-length groups | ≤ 10, 11–50, 51–200, > 200 words | CONV | Separates very short messages (few clues) from typical and long ones. Result: ≤ 10 words is where errors concentrate (Ham lost 6.3% vs 0.5% overall) |
| Validation-curve step | every 25 trees | PRACT | Six points from one trained model, at no extra training cost |

### 5.3 Bootstraps (M07 F.3, Final E.3)

| Constant | Value | Set by | Reason |
|---|---|---|---|
| Resamples | 1,000 | CONV | A standard number for stable 95% percentile intervals (Efron & Tibshirani, 1993) |
| Interval | 95% (2.5th – 97.5th percentile) | CONV | Standard confidence level |

---

## 6. Values tested earlier and not used in the final model

| Value | Meeting | Why it is not in the final model |
|---|---|---|
| Baseline: TF-IDF + MNB, α = 1.0, full vocabulary | M03 | The fixed reference. Retrained unchanged in the Final Evaluation: test Macro F1 0.8377 |
| 50,000-word vocabulary cap | Original M03, tested in M04 | Helps TF-IDF + MNB (+0.0675) only because it reduces smoothing; α = 0.1 without the cap gives +0.0619. No effect on counts (0.9324 vs 0.9323) |
| α = 0.1 check | M04 | Used only to prove the smoothing explanation |
| α grid 0.01 – 1.0 | M05 (V0) | Best was 1.0 at the edge. Extended in M07 to 0.25 – 8, where 2.0 won |
| C grid 1, 4, 16, 64 | M05 (V0) | 64 is used (section 3.2) |
| Decision Tree, `max_depth=20`, `min_df=5` | M04 | The weakest candidate (0.9022). The limits kept training at 233 s |
| SMOTE (`k_neighbors=5`, Phish raised to the size of Ham) | M05 | −0.007 Macro F1. Hypothesis 3 not supported |
| Class weights, random over- and undersampling | M05 | All lower Macro F1 and more Ham lost |
| 150 LightGBM trees | M05 | Re-tuned in M07 on V3 → 125 |
| Text versions V0, V1, V2 | M05 – M07 | Contain source-format shortcuts (H4, H5) |
| SVD 50 / 100 / 200 | M06 | All lower than no reduction |
| Manual count-MNB (no library), α = 1.0, full unigram vocabulary | M04 challenge (`Manual Naive Bayes/`) | Implementation check only: identical to scikit-learn `MultinomialNB` (0 of 55,789 test predictions differ). V0 text, so not comparable with the final model |
| KNN | M04 | Not tested: one prediction compares against ~178,000 emails in ~664,000 dimensions |

---

## 7. Values not re-tuned on the final V3 text (honest list)

These were chosen on the original text (V0), in M04–M05, and carried over unchanged:

| Value | Chosen in | Risk | Why it was acceptable |
|---|---|---|---|
| LR `C` = 64 | M05 grid, V0 | The best C on V3 could be different | It was already at the flat part of the curve (+0.0005 from 16 to 64) |
| LightGBM `learning_rate` 0.2, `num_leaves` 63, `colsample_bytree` 0.3 | M05, 15% sample, V0 | They could be sub-optimal on V3 | The number of trees, which interacts most with the learning rate, *was* re-tuned on V3 (125). Listed as limitation 9 in the Final Evaluation |
| LightGBM feature `min_df` 20, MNB `min_df` 2 | M04 – M05 | Low | Memory settings; 20 and 5 gave identical results |
| Ham Recall target 99.5% | M05 | — | A business decision, not a tuned value |

---

## 8. The demo notebook

| Constant | Value | Reason |
|---|---|---|
| Short-message warning | ≤ 10 words after cleaning | Matches the error-analysis group where mistakes concentrate (M07 G) |
| "Few known words" warning | < 50% of the email's distinct words are in LR's vocabulary | A simple sign that the email is unlike the training data, for example not in English |
| Words shown | top 10 by LR contribution | Enough to explain a decision |
| Every model setting | read from `final_config.json` | The demo uses exactly the M07 configuration and b |

---

## 9. Likely defense questions about constants

**Why 125 trees?**
It was chosen by a validation curve on the final V3 text, not by hand. One LightGBM on fold 1 was scored every 25 trees. Validation Macro F1 rose up to 125 trees (0.8875) and dipped slightly at 150 (0.8868). Our rule picks the smallest number of trees within 0.001 of the best, because fewer trees train faster. Here, 125 is the best value itself.

**Why α = 2.0 and not the default 1.0?**
The validation curve on V3 peaks at 2.0 (0.8356). Values above it smooth too much: at α = 8, Phish Recall falls to 0.40.

**Why b = 1.777?**
It is not chosen by hand. It is the smallest value that keeps 99.5% of legitimate email on the out-of-fold predictions. That 99.5% is the business target. It means an email leaves the inbox only when the model finds Spam or Phish about 5.9 times more likely than Ham.

**Why a tie tolerance of 0.001?**
It is about one fold-to-fold standard deviation. A smaller difference is within noise, so we let a secondary criterion decide instead of the noise. The rule is written in each notebook before the results.

**Why random_state = 42?**
Any fixed number works. The point is that every meeting uses the same split, the same folds and the same model seeds, so results are comparable and repeatable.

**Some settings come from M05 on the old text. Isn't that a problem?**
They are listed openly in section 7. The most important interacting setting, the number of trees, was re-tuned on V3. C was already on a flat part of its curve. Re-tuning everything on V3 would cost many more hours of Colab time, which we judged not worth it given these flat curves.
