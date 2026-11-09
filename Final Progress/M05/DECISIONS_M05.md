# M05 — Decisions and Their Reasons

**Meeting goal (W05 "Output Project", slides p. 51–52):**

- A. validation strategy with proof of no leakage;
- B. at least one ensemble model (Random Forest or Boosting), tuned and compared with the baseline;
- C. imbalance handling: class ratio, metrics, techniques and results;
- D. results as mean ± std, error analysis, weaknesses.

**Notebook:** `Group_1_Email_Spam_Detection_M05.ipynb`. **Reading guide:** `../HOW_TO_READ_RESULTS.md`.

---

## 1. Business priority: "no legitimate email lost"

**Decision.** Mistakes are ranked:

1. legitimate email lost (Ham → Spam / Phish), the worst;
2. phishing into the inbox (Phish → Ham);
3. phishing filtered without a warning (Phish → Spam);
4. spam flagged as phishing (Spam → Phish);
5. spam into the inbox (Spam → Ham), the most acceptable. This is the error we deliberately accept.

The final model must keep **at least 99.5% of legitimate email** (at most 1 in 200 lost). Among models that meet this, the highest Macro F1 wins.

**Why.**

- Our own Business Understanding (M02) set the success criterion: "tingkat *false positive* yang sangat rendah".
- A lost client email costs more than one more spam in the inbox.
- The slides say the decision threshold should be chosen on validation data, *based on the cost of FP vs FN*.
- The guideline lists "biaya kesalahan" (cost of errors) as a valid reason for choosing a final model.

**Reference.** M02 submission, p. 1 (success criteria); W05 slides p. 41; Guideline §13.

---

## 2. Validation: stratified 5-fold cross-validation

**Decision.** The 223,152 training emails are split into 5 folds with equal class proportions. Each model is trained 5 times, each time validated on a different fold, and results are reported as mean ± standard deviation.

**Why.**

- Every training email is validated exactly once.
- The ± shows how stable a result is.
- The test set stays untouched.

**Proof of no leakage.**

- The vectorizers are **re-fitted on each fold's training part**. Their matrices are saved to disk and reused, so the text is processed once per fold, not once per model.
- Resampling (SMOTE, oversampling, undersampling) is applied **only to each fold's training part**.
- Every fold prints **0** texts shared between training and validation (C.2).

**Reference.** W05 slides p. 35 (protocol), p. 47 (applying SMOTE before CV gave a fake F1 of 0.995 vs a real 0.256); W07 p. 35; Hastie, Tibshirani & Friedman, *ESL* ch. 7.

---

## 3. Tuning

| Model | Parameter | Values tried | Best | Note |
|---|---|---|---|---|
| MNB (counts, unigram + bigram) | `alpha` | 0.01, 0.03, 0.1, 0.3, 1.0 | **1.0** | Edge of the range; Macro F1 was still rising. Smaller α keeps more Ham (0.993 at α = 0.01) but finds less Phish. |
| Logistic Regression (TF-IDF) | `C` | 1, 4, 16, 64 | **64** | Edge of the range, but gains are flattening (+0.0005 from 16 to 64). |

**Why grid search with CV:** the slides require tuning with CV on training data only (W05 p. 51, step 4). The notebook flags a best value that sits at the edge of its range. Values beyond the range are checked in M07.

---

## 4. Class imbalance: tested, and none kept

**Ratio.** Ham 2.98 : Phish 1; Spam 2.53 : Phish 1.

| Technique (MNB) | Macro F1 | Phish Recall | Ham lost |
|---|---|---|---|
| None | **0.9441** | 0.960 | **1,050** |
| Class weight | 0.9353 | 0.964 | 1,201 |
| Random oversampling | 0.9350 | 0.957 | 1,150 |
| **SMOTE** | 0.9371 | 0.950 | 1,102 |
| Random undersampling | 0.9328 | 0.971 | 1,342 |

LR + class weight gained +0.0004 Macro F1 but lost 153 more legitimate emails, so it was rejected.

**Decision.** No imbalance technique and **no SMOTE** in the final model. **Hypothesis 3 is not supported.**

**Why SMOTE did not help.**

1. The imbalance is moderate (3 : 1). The slides' advice is to fix evaluation first, then class weight or threshold, and to resample *only if still needed*.
2. SMOTE creates a new point *between* two real Phish emails (x_new = x_i + λ(x_neighbour − x_i)). For text, that is a blend of two emails' word counts, a message that does not exist.
3. Resampling mostly moves the decision threshold rather than improving the model's ranking ability (PR-AUC barely changes). Our Ham threshold (§7) already controls the threshold, directly and with business meaning.

**Reference.** W05 slides p. 41 (workflow order), p. 42 (SMOTE formula), p. 46 ("PR-AUC hampir tidak membaik oleh resampling/class_weight"); Chawla et al. (2002); He & Garcia (2009).

---

## 5. Why LightGBM

1. **The checklist requires it:** W05 asks for at least one ensemble model, Random Forest *or* Boosting (p. 51).
2. **Boosting fixes what our models lack.** Boosting reduces *bias* (W05 p. 50).
   - MNB and LR are linear: each word pushes the score up or down on its own.
   - Gradient-boosted trees learn *combinations* of words (e.g., "account" **and** "verify" **and** no "meeting").
   - Each new tree corrects the errors of the previous trees (Friedman, 2001).
3. **LightGBM is built for large, sparse data:**
   - Histogram-based splits make it fast on large data (W05 p. 19: "Jauh lebih cepat pada data besar").
   - *Exclusive Feature Bundling* merges sparse features that are rarely non-zero together. Our word-count matrix is exactly that kind of data (Ke et al., 2017).
4. **Why not Random Forest:** not tested. A single depth-20 Decision Tree already took 233 s in M04, and Random Forest needs hundreds of deeper trees on the same sparse data.

**Settings and how they were chosen.** They come from timing tests on a 15% real-data sample:

| Setting | Value | Why |
|---|---|---|
| Trees | 150 | Same quality as 300 trees (0.9677 vs 0.9675 Macro F1) in about half the time. |
| `learning_rate` | 0.2 | Pairs with 150 trees; same quality as 300 trees at 0.1. |
| `num_leaves` | 63 | 31 leaves scored lower (0.9657). |
| `colsample_bytree` | 0.3 | Each tree sees 30% of the words: faster, and the trees are more varied. |
| Features | unigram counts in ≥ 20 messages | `min_df` = 20 vs 5 gave identical results with fewer features. |

Training time grows with the number of emails, not words. LightGBM took **119 min** for 5 folds on Colab.

**Weakness:** these settings were not tuned on the full data (planned for M07).

**Result.** The best single model, Macro F1 **0.9773**, but at the default threshold it loses the most legitimate email (Ham Recall 0.9815).

**Reference.** Ke, G. et al. (2017); Friedman, J. H. (2001); W05 slides p. 19, 50, 51.

---

## 6. Why stacking

**Decision.** A *meta-model* (Logistic Regression with `StandardScaler`) takes the base models' predicted log-probabilities as input and learns when to trust which model. Two versions were tested:

- MNB + LR;
- MNB + LR + LightGBM.

**Why.**

- M04 showed that MNB and LR make **different mistakes**: MNB loses fewer legitimate emails, LR lets less spam in. Stacking combines models of *different types* so each covers the other's weakness.
- Voting gives each model a fixed say, either equal or set by hand. Stacking *learns* from data how much to trust each model.

**How it avoids leakage.** The meta-model is trained on **out-of-fold** predictions: each email's base predictions come from models that never saw that email. The meta-model is then evaluated with the same 5 folds.

**Why clip log-probabilities to [−30, 0].** MNB produces log-probabilities down to the thousands of negative units. Without a limit, MNB alone would dominate the meta-model's inputs.

**Result.** MNB + LR + LightGBM reaches Macro F1 **0.9807** (best overall) and lifts Ham Recall from 0.9815 (LightGBM alone) to 0.9886.

**Reference.** Wolpert, D. H. (1992); W04 slides p. 61; W05 slides p. 9, 22, 23.

---

## 7. The Ham threshold `b`

**Decision.** Add `b` to the log-probability of Ham before choosing the class. For each model, `b` is the smallest value that gives Ham Recall ≥ 99.5% on the out-of-fold predictions.

**Meaning.** A message leaves the inbox only if P(Spam or Phish) > e^b × P(Ham). For the final model, b = 1.514, so the model must find Spam or Phish about **4.5×** more likely than Ham.

**Why this, and not resampling.** It controls exactly the error we care about (legitimate email lost) with one interpretable number. The slides recommend a threshold chosen on validation data by error cost.

**Honest note.** `b` was chosen and scored on the same out-of-fold predictions, so the result is slightly optimistic. The final test-set evaluation checks it once.

**Reference.** W05 slides p. 41, 44 (threshold moving).

---

## 8. Result and current final candidate

At Ham Recall 99.5% (511 of 102,192 legitimate emails lost):

| Model | Macro F1 | Phishing into inbox | Spam into inbox |
|---|---|---|---|
| MNB alone | 0.9292 | 488 | 10,202 |
| LR alone | 0.9612 | 184 | 4,070 |
| LightGBM | 0.9709 | 237 | 4,042 |
| Stacking MNB + LR | 0.9677 | 118 | 2,768 |
| **Stacking MNB + LR + LightGBM** | **0.9784** | **86** | **2,146** |

The trade-off chart (G.2) shows the 3-model stack lowest at *every* level of legitimate-email loss, not only at 0.5%.

**Open items for M07:**

- Is MNB needed in the stack (test LR + LightGBM without it)?
- α > 1.
- LightGBM tuning.
- Hypothesis 4 (number-format shortcut).

---

## 9. Likely defense questions

**Why didn't you use SMOTE, when your plan said MNB + TF-IDF + SMOTE?**
We tested it inside 5-fold CV, on the training part of each fold only. It lowered Macro F1 by 0.007 and lowered Phish Recall. The imbalance is moderate (3 : 1), and SMOTE's synthetic emails are blends of two real emails' word counts. Following the slides' order, we use a threshold instead of resampling.

**Why LightGBM?**
W05 requires a boosting or Random Forest ensemble. Boosting learns word *combinations* that linear models cannot, and LightGBM is designed for large, sparse data (histogram splits, feature bundling; Ke et al., 2017). It was our best single model.

**Why stacking instead of voting?**
Our models make different kinds of mistakes. Stacking *learns* how much to trust each one, using out-of-fold predictions so the meta-model never sees a base prediction made on its own training email.

**Why 99.5%?**
Our M02 success criterion asks for very low false positives. 99.5% means at most 1 in 200 legitimate emails is lost. The trade-off chart shows what a stricter or looser target would cost.

**Isn't choosing `b` on training predictions cheating?**
The predictions are out-of-fold, so no model saw the email it predicted. The value is still slightly optimistic, which is why the final model and its `b` are checked once on the untouched test set.

**Your MNB scores higher Ham Recall at α = 0.01. Why choose α = 1?**
The tuning rule uses Macro F1. Ham protection is enforced later and directly by `b`, for every model in the same way.

**Why is the standard deviation so small (≤ 0.0011)?**
Each fold has about 44,600 validation emails. With that many, one fold's result barely differs from another's.

---

## 10. References

- Chawla, N. V., Bowyer, K. W., Hall, L. O. & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
- He, H. & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*.
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259.
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, ch. 7. Springer.
- Course slides: W04 p. 61; W05 p. 9, 19, 22, 23, 35, 41, 42, 44, 46, 47, 50, 51; W07 p. 35. M02 submission p. 1. Guideline §13.

The papers by Chawla, Friedman, He & Garcia and Wolpert are listed in the W05 slides' reading list (p. 54). LightGBM is named on W05 p. 19.
