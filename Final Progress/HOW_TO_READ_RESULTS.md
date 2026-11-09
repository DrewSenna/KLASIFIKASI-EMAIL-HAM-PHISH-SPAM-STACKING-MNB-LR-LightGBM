# How to Read Our Results

This guide covers the terms used in every meeting's notebook. Each meeting folder also has a `DECISIONS_M0x.md` that explains *why* each choice was made.

---

## 1. Confusion matrix and TP / FP / FN

Rows are the **actual** class; columns are the **predicted** class. The diagonal holds correct predictions; every other cell is one type of mistake, written `actual → predicted`.

|  | Predicted Ham | Predicted Phish | Predicted Spam |
|---|---|---|---|
| **Actual Ham** | correct | Ham → Phish | Ham → Spam |
| **Actual Phish** | Phish → Ham | correct | Phish → Spam |
| **Actual Spam** | Spam → Ham | Spam → Phish | correct |

With three classes, TP / FP / FN are counted **per class**: that class is the "positive", and the other two together are the "negative" (*one-vs-rest*). So **every mistake cell is two things at once**:

| Mistake (actual → predicted) | Our plain-language name | For the **actual** class it is a… | For the **predicted** class it is a… |
|---|---|---|---|
| Ham → Spam, Ham → Phish | legitimate email lost | False Negative of Ham | False Positive of Spam / Phish |
| Phish → Ham | phishing into the inbox | False Negative of Phish | False Positive of Ham |
| Phish → Spam | phishing filtered without a warning | False Negative of Phish | False Positive of Spam |
| Spam → Phish | spam flagged as phishing | False Negative of Spam | False Positive of Phish |
| Spam → Ham | spam into the inbox | False Negative of Spam | False Positive of Ham |

**From the business view** (positive = "a harmful email was detected"):

- *Phishing into the inbox* is a **false negative**: a threat was missed.
- *Legitimate email lost* is a **false positive**: a safe email was treated as harmful. This is the "false positive" our M02 success criterion asks to keep very low.

**In the M05 imbalance table** (the professor's TP-vs-false-alarm format), Phish is the positive class:

- TP = Phish → Phish
- FP = Ham → Phish + Spam → Phish (false alarms)
- FN = Phish → Ham + Phish → Spam (missed phishing)

**Formulas** (for any class used as the positive):

- Recall = TP / (TP + FN): of all real members of the class, how many were found.
- Precision = TP / (TP + FP): of everything labelled as the class, how much really belongs to it.
- F1 = 2 × Precision × Recall / (Precision + Recall).

Our two key recalls:

- **Ham Recall**: the share of legitimate emails that stay in the inbox. 1 − Ham Recall is the share of legitimate email lost.
- **Phish Recall**: the share of phishing emails that are recognised as Phish.

---

## 2. Macro vs Weighted averages

- **Macro**: the plain average of the three per-class scores. Ham, Phish and Spam count equally, so weak Phish detection cannot hide behind the large Ham and Spam classes. This is our main metric.
- **Weighted**: the average weighted by the number of messages per class, so it is pulled toward the large classes (Ham, Spam).

---

## 3. How to read "naik 0.1038", "+0.104", "turun 0.007"

- The number is always an **absolute difference** between two results **on the same data**, against the reference named in the same sentence.
- Example (M04 D.1): on the same 44,631 validation emails, MNB + word counts = 0.9323 and MNB + TF-IDF = 0.8286, so "Macro F1 naik 0.1038" (+0.104).
  - This is about **10.4 percentage points**, not a 10.4% relative increase. The relative increase would be 0.1038 / 0.8286 ≈ 12.5%.
  - It does **not** mean "since M04". It compares two configurations inside the same experiment.
- "hampir tidak berubah" (almost unchanged) means the difference is smaller than 0.001.

---

## 4. Which data a number comes from

Numbers from different data cannot be compared directly, even for the same model.

| Meeting | Evaluated on | Size |
|---|---|---|
| M03 baseline | **Test set** (20% of clean data) | 55,789 emails |
| M04 | **Validation split** (20% of the training data) | 44,631 emails |
| M05, M06 | **5-fold cross-validation** on the training data | 223,152 emails, each validated once |
| Final evaluation | **Test set**, used only once for the final model | 55,789 emails |

Example: the baseline scores Macro F1 0.8377 on the test set (M03) but 0.8286 on the validation split (M04). It is the same model; the messages are different.

**Out-of-fold (OOF) prediction** (M05): in 5-fold cross-validation, each email is predicted by a model that was trained on the other four folds and never saw that email. Collecting these predictions gives one honest prediction per training email.

---

## 5. "0.9441 ± 0.0008"

This is the average over the five cross-validation folds ± the standard deviation between folds. A small ± means the result is stable and does not depend on which emails landed in which fold.

---

## 6. PR-AUC (Phish)

PR-AUC measures how well a model **ranks** Phish above non-Phish messages, across all possible thresholds. 1.0 means every Phish ranks above every non-Phish. It does not depend on a single decision threshold, so it shows the model's underlying ability. Resampling and class weights mostly move the threshold, not this ranking (W05 slides p. 46).

---

## 7. The Ham threshold `b` (M05)

The model normally picks the class with the highest probability. We add `b` to the log-probability of Ham before choosing. This means a message leaves the inbox **only if** P(Spam or Phish) > e^b × P(Ham).

- Example: `b = 1.514` gives e^1.514 ≈ 4.5. A message is removed from the inbox only when the model finds Spam or Phish at least about 4.5 times more likely than Ham.
- `b = 0` is the normal behaviour.
- A larger `b` protects legitimate email more, but lets more spam and phishing into the inbox.

---

## 8. Clustering terms (M06)

Silhouette, K, the 0.25 threshold, elbow, Davies–Bouldin, Calinski–Harabasz, purity, ARI and NMI are explained with worked examples in `M06/DECISIONS_M06.md`, section 0.
