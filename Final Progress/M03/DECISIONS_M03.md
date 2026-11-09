# M03 — Decisions and Their Reasons

**Meeting goal (W03 "Output Project Pertemuan 3"):** problem, dataset, preprocessing, baseline, initial evaluation, initial findings.
**Notebook:** `Group_1_Email_Spam_Detection_M03.ipynb`. **Reading guide:** `../HOW_TO_READ_RESULTS.md`.

---

## 1. Decisions

| # | Decision | Why | Evidence (notebook) | Reference |
|---|---|---|---|---|
| 1 | Remove the 2 empty texts and 1 whitespace-only text | There is nothing to learn from an empty message. | B.5, C.2 | — |
| 2 | Remove **every** row of a text that appears with more than one label (14 groups, 44 rows), instead of keeping one | Keeping the first row would let the row order decide the label, which adds label noise. The correct label cannot be known. | B.5 (e.g. "unsubscribe" labelled both Ham and Spam), C.2 | Guideline §2 data quality |
| 3 | Remove the 86,460 same-label duplicates **before** splitting train/test | The same text in both training and test lets the model be tested on text it has already memorised (*data leakage*), which inflates the score. | C.2: 365,448 → 278,941 | W07 slides p. 35 (leakage checklist: split before preprocessing); guideline §23 |
| 4 | Minimal text cleaning: no stemming, lemmatisation, stopword, punctuation or URL removal | Words such as `http`, `www` or `verify` may separate spam from phishing. We only remove something once an experiment shows it helps. | C.1; B.6 shows links survive as words (`http www site com`) | — |
| 5 | Stratified 80/20 split, `random_state=42` | Phish is the smallest class (15.4% after cleaning); stratifying keeps the same class proportions in train and test. The seed makes the split repeatable. | D.2 | W05 slides p. 41 (stratified split for imbalanced data) |
| 6 | TF-IDF is fitted on the training data only, inside a `Pipeline` | Fitting it on all data would let the test set shape the vocabulary and IDF weights (leakage). | D.3 | W07 slides p. 35 |
| 7 | `sublinear_tf=True` | Some emails are extremely long (up to 1,585,483 words). A log scale stops one word repeated thousands of times from dominating. | B.6.1 (30,385 IQR outliers, 8.31%) | — |
| 8 | Keep the full vocabulary (no 50,000-word cap) | Don't discard information without evidence. **M04 later tested this and found the cap helps TF-IDF + MNB, because of smoothing** (see DECISIONS_M04 §1.4). | D.1 parameter table | — |
| 9 | Baseline = TF-IDF + **Multinomial Naive Bayes**, `alpha=1.0`, untuned | A simple, fast, classic text classifier and a fair starting point. The slides give "TF-IDF + Logistic Regression" as *an example* text baseline; we added Logistic Regression as a candidate in M04 instead. | D.1 | McCallum & Nigam (1998); Manning et al. (2008) ch. 13; W03 slides p. 45 |
| 10 | Main metrics: **Macro F1** and **Phish Recall**, not accuracy | The raw class ratio is 3.93 : 1 (Ham : Phish). Accuracy would hide poor Phish detection behind the large classes. | B.4, E | W04 slides (metric choice); W05 slides p. 41 (PR, F1, recall for imbalance) |
| 11 | Audit texts that end up with no TF-IDF features, and keep them | Such texts are classified by class proportion alone. Keeping them keeps the evaluation realistic. | D.4 audit (26 train, 57 test) | — |

## 2. Results (test set, 55,789 emails)

| Accuracy | Macro F1 | Phish Recall | Phish Precision | Phish → Spam | Phish → Ham | Spam → Ham |
|---|---|---|---|---|---|---|
| 0.8881 | 0.8377 | 0.5316 | 0.9844 | 3,475 | 538 | 1,991 |

Hypotheses written in F.3 and where they were tested:

- **H1:** Phish and Spam share vocabulary. Tested in M04 and M07.
- **H2:** Unigrams miss phrases. Tested in M04; **supported**.
- **H3:** The smaller Phish class pulls predictions toward larger classes. Tested in M05; **not supported**.

## 3. Likely defense questions

**Why deduplicate before splitting, not after?**
If duplicates are split across train and test, the model is tested on emails it has already seen, so the test score is too high. Removing them first means each test email is new to the model.

**Why delete all rows of a conflicting text instead of keeping one?**
We cannot know which label is correct. Keeping the first occurrence would let the row order decide, which adds label noise to the data.

**Why isn't accuracy your main metric?**
Phish is about 12–15% of the data. A model that rarely predicts Phish can still score high accuracy. Macro F1 gives each class equal weight, and Phish Recall directly measures missed phishing.

**Why Multinomial Naive Bayes as the baseline?**
It is the classic, fast text classifier and needs no tuning to give a reasonable result. A baseline only has to be a fair reference, not the best model.

**Your baseline (0.8377) is lower than the earlier submitted version (0.8980). Why?**
That version capped the vocabulary at 50,000 words; this one keeps every word. M04 D.3 shows why the full vocabulary hurts TF-IDF + MNB: with α = 1, the smoothing added to every word (α × 664,008 words) is about 3× larger than Phish's total TF-IDF weight. This flattens Phish's word probabilities, so the model rarely predicts Phish (Precision 0.98, Recall 0.53).

**What is `escapenumber`?**
A placeholder that the dataset's creators put in place of numbers in some of their sources. M06 shows it is almost absent from Phish, which leads to Hypothesis 4.

## 4. References

- McCallum, A. & Nigam, K. (1998). A comparison of event models for naive Bayes text classification. *AAAI-98 Workshop on Learning for Text Categorization*.
- Manning, C. D., Raghavan, P. & Schütze, H. (2008). *Introduction to Information Retrieval*, ch. 13. Cambridge University Press.
- Course slides: W03 p. 45 (baseline examples); W05 p. 41 (imbalance workflow); W07 p. 35 (leakage checklist).
- Guideline Final Project, §2 and §23.
