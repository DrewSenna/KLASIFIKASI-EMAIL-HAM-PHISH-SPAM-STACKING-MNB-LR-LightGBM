# M04 — Decisions and Their Reasons

**Meeting goal (W04 "Project Checkpoint – Pertemuan 4"):** baseline plus at least two candidate models with reasons, a comparison table, and 3–5 findings. No final model yet.
**Notebook:** `Group_1_Email_Spam_Detection_M04.ipynb`. **Reading guide:** `../HOW_TO_READ_RESULTS.md`.

---

## 1. Decisions

### 1.1 Evaluate on a validation split, not the test set

The test set is thrown away inside the notebook. 20% of the training data (44,631 emails) becomes a validation set.

- **Why:** choosing models by their test score turns the test set into training data, so the final score becomes too optimistic. The slides' protocol is: set the test set aside → compare on training data → test once at the end.
- **Evidence:** B.1 prints **0** texts shared between the fit and validation parts.
- **Reference:** W05 slides p. 35 (validation protocol).

### 1.2 Change one factor at a time

Experiment 1 changes only the word weighting, then only the vocabulary size, then only the phrase length.

- **Why:** if two things change at once, we cannot tell which one caused the result.

### 1.3 Word counts instead of TF-IDF for MNB

- **Why:** Multinomial Naive Bayes models *how many times* each word appears in a message. Counts are its natural input; TF-IDF is a normalised weighting designed for other purposes.
- **Evidence:** D.1: counts 0.9323 vs TF-IDF 0.8286 Macro F1 (**+0.1038**); Phish Recall 0.9498 vs 0.5144.
- **Reference:** McCallum & Nigam (1998), the multinomial event model.

### 1.4 The 50,000-word cap, and *why* it matters

- **Evidence:** D.2: the cap raises TF-IDF Macro F1 by **+0.0675** but changes counts by about 0 (0.9324 vs 0.9323).
- **Cause (D.3):** MNB estimates P(word | class) = (weight of word in class + α) / (total weight of class + α × |V|).
  - Laplace smoothing adds α to *every* word, so each class gets α × |V| = 664,008 extra.
  - With TF-IDF, each message's weights add up to only a few units, so Phish's total weight is just **207,178**. That is **0.31×** the smoothing. Phish's word probabilities are flattened toward "all words equally likely", and its distinctive words lose their effect.
  - With counts, Phish's total is 3,249,018 (4.9× the smoothing), so the problem does not occur.
- **Proof:** α = 0.1 **keeps every word** and still gives **+0.0619**, almost as much as the cap. So the gain comes from smaller smoothing, not from dropping words.
- **How the 50,000 words are picked:** scikit-learn ranks words by their total count over all training messages and keeps the top 50,000. The rarest words are dropped.
- **Reference:** Manning et al. (2008) ch. 13 (Laplace smoothing in Naive Bayes).

### 1.5 Bigrams, tested without the cap

- **Why without the cap:** with a fixed 50,000 features, adding bigrams pushes unigrams out, so the test would change two things at once.
- **Why `min_df=2`:** the number of distinct word pairs is huge. Keeping only pairs that appear in at least 2 messages saves memory, and both configurations use the same filter.
- **Evidence:** D.4: +0.0101 Macro F1 (0.9443 vs 0.9342), so **H2 is supported**.

### 1.6 Candidate models

| Candidate | Reason | Reference |
|---|---|---|
| MNB (best representation) | Same algorithm as the baseline, so any difference comes only from the representation. | — |
| Logistic Regression + TF-IDF | Standard linear text classifier, taught in W04. It learns a weight for every word directly from its errors (it is *discriminative*), so it can down-weight words shared by Phish and Spam. | W04 slides p. 10–28, p. 112; W03 p. 45; Jáñez-Martino et al. (2023) p. 1150 |
| Decision Tree | A readable set of yes/no rules; checks whether a few word rules are enough. | W04 slides p. 35 |
| **KNN excluded** | To predict one email it must compare against all ~178,000 training emails over ~664,000 word dimensions, and distances lose meaning in very high dimensions. W04 says not every model must be tried if it does not fit the data. | W04 slides p. 91 (KNN prediction "Lambat (data besar)"), p. 112; W06 (curse of dimensionality) |

### 1.7 Decision Tree limits (`max_depth=20`, `min_df=5`)

- **Why:** a local timing test showed that training time grows with *vocabulary size × number of tree nodes*. An unlimited tree on ~664,000 words could run for hours.
- **Evidence:** the tree took 233 s with these limits.

### 1.8 Selection rule

Highest Macro F1 on validation; if two configurations are within 0.001, the one with fewer features wins. The rule is written in B.2 *before* the results, so the choice cannot be bent after seeing them.

## 2. Results (validation, 44,631 emails)

| Role | Model | Macro F1 | Phish Recall | Phish Precision |
|---|---|---|---|---|
| Baseline | MNB + TF-IDF | 0.8286 | 0.514 | 0.983 |
| Candidate 1 | MNB + counts (unigram + bigram) | 0.9443 | 0.960 | 0.882 |
| Candidate 2 | Logistic Regression + TF-IDF | **0.9647** | 0.932 | **0.967** |
| Candidate 3 | Decision Tree | 0.9022 | 0.876 | 0.928 |

The two leaders make **different** mistakes (F.3):

| Mistake | MNB | LR |
|---|---|---|
| Ham → Spam | **174** | 421 |
| Phish → Spam | **262** | 458 |
| Spam → Ham | 799 | **257** |
| Spam → Phish | 835 | **189** |

MNB loses fewer legitimate emails; LR lets less spam through. This difference is what motivated stacking in M05.

## 3. Likely defense questions

**Why not use the test set to compare models?**
Every comparison is a choice. Choosing on the test set means the test score no longer measures performance on unseen data.

**Why does TF-IDF hurt Naive Bayes so much?**
Smoothing adds α to every word in every class, 664,008 in total. With TF-IDF, Phish's total weight (207,178) is smaller than that, so its word probabilities become almost uniform and the model rarely predicts Phish. Lowering α to 0.1 without removing any word recovers most of the loss, which confirms the cause.

**Is the 50,000-word cap a good idea then?**
For TF-IDF + MNB yes, because it reduces the smoothing total. For word counts it makes no difference. Our MNB uses counts.

**Why exclude KNN?**
Prediction cost (a comparison with ~178,000 emails per message) and the curse of dimensionality. The W04 slides allow skipping models that do not fit the data.

**Why is Logistic Regression better overall?**
It learns word weights directly to separate the classes, including words shared by Phish and Spam. Naive Bayes estimates each word's probability per class independently, assuming words are unrelated.

**Why is the Decision Tree weakest?**
A single depth-20 tree can use only a limited number of word rules, and one tree is unstable (high variance). M05 uses boosting, which combines many trees, to address this.

## 4. References

- McCallum, A. & Nigam, K. (1998). A comparison of event models for naive Bayes text classification. *AAAI-98 Workshop on Learning for Text Categorization*.
- Manning, C. D., Raghavan, P. & Schütze, H. (2008). *Introduction to Information Retrieval*, ch. 13.
- Jáñez-Martino, F. et al. (2023). A review of spam email detection: analysis of spammer strategies and the dataset shift problem. *Artificial Intelligence Review*, 56, 1145–1173. p. 1150 (PDF p. 6): "NB, Logistic Regression (LR) or SVM" as the usual text classifiers.
- Course slides: W03 p. 45; W04 p. 10–28, 35, 91, 112; W05 p. 35.
