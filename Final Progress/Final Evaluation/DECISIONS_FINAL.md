# Final Evaluation — Decisions and Results

**Goal (Guideline §13–14):** evaluate the chosen model once on the test set, compare it with the M03 baseline on the same test set, and interpret it.

**Notebook:** `Group_1_Email_Spam_Detection_Final_Evaluation.ipynb` (executed on Colab on 5 Nov 2026, about 45 minutes of training). **Reading guide:** `../HOW_TO_READ_RESULTS.md`.

---

## 1. Decisions

### 1.1 Load the M07 results from Drive instead of recomputing them

**What.** The notebook reads two files from `MyDrive/Group1_ML/m07_artifacts/`:

- `final_config.json`: text patterns, hyperparameters, the chosen model and b;
- `oof_log_proba.npz`: the out-of-fold predictions of every M07 model.

It does not recompute cross-validation.

**Why.**
- The configuration, the out-of-fold predictions and b are **the same as M07 by construction**, so the notebook cannot drift out of sync with the meeting that chose the model.
- It runs in about 1 hour instead of 4.5.
- Hypothesis 5 was already tested in M07, so repeating it here would add nothing.

**Decided by:** the user, 4 Nov.

### 1.2 Everything was fixed before the test set was used

| Fixed in | What |
|---|---|
| M07 | Text version V3 and its patterns |
| M07 | MNB α = 2.0, LR C = 64, LightGBM 125 trees |
| M07 | Final model: Stacking MNB + LR + LightGBM |
| M07 | Ham bias b = 1.777, chosen for Ham Recall ≥ 99.5% on out-of-fold predictions |

The test labels are used only in section E of the notebook. Nothing was changed after the test results were seen. This is the second and last use of the test set; the first was the M03 baseline (ROADMAP protocol).

### 1.3 How the final model is built

1. **Meta-model:** trained on M07's out-of-fold log-probabilities for all 223,152 training emails (clipped to [−30, 0], standardised, Logistic Regression).
2. **Base models:** each retrained once on all 223,152 training emails (V3 text). Training time: MNB 3.4 min, LR 3.3 min, LightGBM 33 min.
3. **Test prediction:** the base models predict the test set, then the meta-model combines them, then b is added to log P(Ham).
4. **Baseline:** the M03 baseline (TF-IDF + MNB, α = 1, raw text) is retrained and scored on the same test set. It reproduces M03 exactly: Accuracy 0.8881, Macro F1 0.8377, Phish Recall 0.5316.

---

## 2. Results (test set, 55,789 emails: 25,548 Ham, 8,568 Phish, 21,673 Spam)

### 2.1 Main result

| | Baseline M03 | Final model, b = 0 | **Final model, b = 1.777** |
|---|---|---|---|
| Accuracy | 0.8881 | 0.9319 | **0.9258** |
| **Macro F1** | 0.8377 | 0.8967 | **0.8913** |
| **Ham Recall** | 0.9932 | 0.9882 | **0.9963** |
| Phish Recall | 0.5316 | 0.6932 | **0.6901** |
| Phish Precision | 0.9844 | 0.9029 | **0.9055** |
| PR-AUC Phish | 0.8863 | 0.8887 | **0.8887** |

### 2.2 The business priorities

| Error | Baseline M03 | **Final model** | Change |
|---|---|---|---|
| **1. Legitimate email lost** (Ham → Spam or Phish) | 174 (0.68%) | **94 (0.37%)** | **−46%** |
| **2. Phishing into the inbox** (Phish → Ham) | 538 (6.28%) | **107 (1.25%)** | **−80%** |
| **3. Spam into the inbox** (Spam → Ham) | 1,991 (9.19%) | **781 (3.60%)** | **−61%** |
| Phishing filed as Spam (still kept out of the inbox) | 3,475 (40.6%) | 2,548 (29.7%) | −27% |
| Spam filed as Phish | 65 | 612 | +547 |

The final model is better than the baseline on **all three business priorities at once**. The only error that grows is Spam filed as Phish. Those emails are still filtered out of the inbox; they just get the stronger warning.

**The comparison is conservative.** The baseline reads raw text, so it can still use the source-format shortcuts (H4, H5). The final model reads V3, where those shortcuts are removed. On equal terms, the gap would be larger.

**What b does on the test set** (b = 0 → b = 1.777):

| | b = 0 | b = 1.777 | Change |
|---|---|---|---|
| Legitimate email lost | 302 | 94 | 208 saved |
| Phishing into the inbox | 33 | 107 | +74 |
| Spam into the inbox | 244 | 781 | +537 |

This is the trade the business priority asks for: 208 legitimate emails saved, paid for with 74 phishing and 537 spam emails that reach the inbox.

### 2.3 Cross-validation vs test

| | CV (training, M07) | Test | Difference |
|---|---|---|---|
| Macro F1 | 0.8855 | 0.8913 | +0.0058 |
| Ham Recall | 0.9950 | 0.9963 | +0.0013 |
| Phish Recall | 0.6783 | 0.6901 | +0.0119 |
| Phish Precision | 0.8925 | 0.9055 | +0.0130 |

The test result is slightly **better** than cross-validation, so neither the model nor b overfits the training data.

Why it's a little better:
- In cross-validation each base model trained on 80% of the training data; the final models trained on 100%. The M07 learning curves were still rising at 100%.
- b was chosen on the less confident out-of-fold predictions, so on the test set it is slightly conservative: Ham Recall 0.9963 instead of exactly 0.995.

### 2.4 Confidence intervals (bootstrap, 1,000 resamples of the test set)

| Model | Metric | Value | 95% interval |
|---|---|---|---|
| Final model | Macro F1 | 0.8913 | 0.8878 – 0.8946 |
| Final model | Ham Recall | 0.9963 | 0.9956 – 0.9971 |
| Final model | Phish Recall | 0.6901 | 0.6798 – 0.6997 |
| Baseline M03 | Macro F1 | 0.8377 | 0.8341 – 0.8416 |
| Baseline M03 | Ham Recall | 0.9932 | 0.9921 – 0.9941 |
| Baseline M03 | Phish Recall | 0.5316 | 0.5215 – 0.5428 |

- The Macro F1 intervals do not overlap, so the improvement over the baseline is not luck in which emails ended up in the test set.
- The lower bound of the Ham Recall interval (0.9956) is above the 99.5% target.

### 2.5 Interpretation

**How much the meta-model uses each base model** (share of absolute weight per class):

| Class | MNB | LR | LightGBM |
|---|---|---|---|
| Ham | 11.0% | **54.6%** | 34.3% |
| Phish | 7.1% | 23.6% | **69.4%** |
| Spam | 5.8% | 39.4% | **54.9%** |

- LR decides most of Ham, and LightGBM most of Phish and Spam.
- MNB gets a small weight (6–11%). That still matters: in M07, adding it removed 164 phishing and 456 spam emails from the inbox (bootstrap-significant).

**LR's highest-weight words:**

| Class | Top words |
|---|---|
| Ham | `enron`, `wrote`, `vince`, `thanks`, `louise`, `houston`, `cannot`, `university`, `samba`, `subject`, `debian`, `yahoo` |
| Phish | `nan`, `cnncom`, `josemonkeyorg`, `fw`, `contenttype`, `fwd`, `contenttransferencoding`, `prescripiton`, `sirmadam`, `monkeyorg`, `urltoken` |
| Spam | `cannot`, `hk`, `uwaterloo`, `adf`, `paulresis`, `canadianpharmacy`, `anatrim`, `php`, `usdrugs`, `vista`, `wanna` |

Many of these words are **source fingerprints**, not phishing or spam language:

| Fingerprint | Where it comes from |
|---|---|
| `enron`, `vince`, `louise` | Enron staff (Ham) |
| `samba`, `debian` | Open-source mailing lists (Ham) |
| `josemonkeyorg`, `monkeyorg` | Recipient address of the phishing corpus |
| `contenttype`, `contenttransferencoding` | Email headers glued together by the Phish source |
| `nan` | A missing value written as text |
| `uwaterloo` | The source corpus's university (Spam) |

Even after V3, the model partly recognises the **source corpus** of an email. That is the main limitation (section 3).

**Test examples:**
- **Lost Ham:**
  - very short messages ("scanned");
  - an unsubscribe footer ("click wish receive future emailings us");
  - a 7-word personal note;
  - a spam-like text labelled Ham.
- **Missed phishing:**
  - several emails that look like spam rather than phishing: job-resume spam, "debt elimination", "pure profit software";
  - a short newsletter request.

Part of the remaining error is label noise between Phish and Spam.

---

## 3. Limitations (also in notebook G.2)

1. **Source fingerprints remain.** Each class comes from a different corpus. Formats were made uniform (H4, H5), but words that identify the corpus remain (table in 2.5). On email from another organisation, performance can be lower: the dataset-shift risk in Jáñez-Martino et al. (2023).
2. **The test set comes from the same dataset.** About half of the validation emails have a near-copy in training (M07). The test set therefore does not measure performance on truly new sources.
3. **Label definitions.** Part of the Phish class is pharmacy or job spam, and part of Ham is commercial newsletters. About 30% of Phish is classified as Spam.
4. **Short and non-ASCII messages.** Errors concentrate in messages of 10 words or fewer. Phishing with non-ASCII characters reaches the inbox more often (M07).
5. **Side effects of V3.** Negation words (`not`, `no`, `nor`) are removed with the other stopwords. Glued words in the Phish source cannot be split.
6. **Limited tuning.** For LightGBM only the number of trees was tuned.

---

## 4. Likely defense questions

**Why is the test score higher than cross-validation? Isn't that suspicious?**
No. The difference is small (+0.006 Macro F1). The final models trained on all the training data, while each cross-validation model saw only 80%, and the learning curves were still rising. Nothing was tuned on the test set: every setting and b came from M07.

**Your Macro F1 dropped from 0.978 (M05, cross-validation) to 0.891. Did the project get worse?**
No. The 0.978 was inflated by source-format shortcuts (H4, H5) that real email does not have. 0.891 is the honest number. It is also measured on the test set, which was never used for any decision.

**Is the improvement over the baseline real?**
Yes. The 95% bootstrap intervals for Macro F1 do not overlap (0.8878–0.8946 vs 0.8341–0.8416). The final model also has fewer errors on all three business priorities, even though the baseline still benefits from the format shortcuts.

**How many legitimate emails does the model lose?**
94 of 25,548 test emails (0.37%). The target was at most 0.5%, and the 95% interval for Ham Recall (0.9956–0.9971) stays above 99.5%.

**Why does MNB get such a small weight if you chose the model because of it?**
The weight shows how much MNB moves the meta-model's decision, not how useful it is. Even a small contribution changes the borderline cases at the Ham threshold. In M07 that meant 164 fewer phishing and 456 fewer spam emails in the inbox, and the bootstrap showed it is not luck.

**Why weren't the source-fingerprint words removed?**
There is no end to them: every corpus has its own names, addresses and headers. Removing them by hand would mean choosing features by looking at the label. The honest choice is to report them as a limitation and to name dataset shift as the main risk.

---

## 5. References

- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Jáñez-Martino, F. et al. (2023). A review of spam email detection: analysis of spammer strategies and the dataset shift problem. *Artificial Intelligence Review*, 56, 1145–1173.
- Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS 30*.
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259.
- Guideline Final Project §13–14.
