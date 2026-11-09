# Manual Naive Bayes — Decisions and Their Reasons

**Goal:** answer two questions about the Meeting 04 challenge (Multinomial Naive Bayes without a library).
1. Is the manual calculation correct?
2. How does it score on the **same** train/test split as the M03 baseline?

**Notebook:** `Group_1_Naive_Bayes_Manual_Verification.ipynb`. Ran on Colab on 8 Nov 2026 (manual training 65 s, manual prediction 24 s); no Drive is needed. **Reading guide:** `../HOW_TO_READ_RESULTS.md`.

---

## 1. Decisions

### 1.1 Use the baseline's split, not the challenge's own split

- **What:** `train_test_split(test_size=0.20, random_state=42, stratify=y)` on the same cleaned data (278,941 emails). This is the exact code from M03 C/D.
- **Why:** the submitted challenge notebook wrote its own stratified split with `random.Random(42)`. It has the same sizes (223,152 / 55,789) but puts **different emails** in the test set. So its Macro F1 (0.9346) could not be compared with the baseline.
- **Proof that the split is identical:** D.3 retrains the M03 baseline (TF-IDF + MNB, α = 1) in the same notebook. It must reproduce the recorded M03 test result: Accuracy 0.8881, Macro F1 0.8377, Phish Recall 0.5316. If any email moved between training and testing, these numbers would change. A.3 also prints a SHA-256 fingerprint of the test indices, so other notebooks can be checked against this split.

### 1.2 The manual code is copied unchanged

- **What:** `tokenize`, `word_frequencies`, `train_multinomial_nb`, `predict_multinomial_nb` and the metric code are the submitted challenge code, character for character.
- **Why:** the question is whether **our submitted** code is correct. Rewriting it would verify a different program.

### 1.3 Verify against scikit-learn step by step, not only by the final score

- **What:** the reference is `CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w\w+\b')` + `MultinomialNB(alpha=1.0)`. Same tokenizer, same counts, same smoothing.
- **Why:** if only the final accuracy matched, two errors could cancel out. Comparing each intermediate value tells us *where* an error is: the first check that fails.

| # | Check | What it proves | Must be |
|---|---|---|---|
| 1 | Vocabulary (size and contents) | Tokenisation is correct | identical |
| 2 | D_c (emails per class) | Class counting | identical |
| 3 | T_c (tokens per class) | Token totals | identical |
| 4 | F_w,c for every word × class | Word counting | identical |
| 5 | log P(c) | Prior formula | within 1e-9 |
| 6 | log P(w\|c) for every word × class | Laplace smoothing formula | within 1e-9 |
| 7 | log S_c for every test email × class | Prediction formula | within 1e-9 (relative) |
| 8 | Posterior P(c\|x) | log-sum-exp normalisation | within 1e-9 |
| 9 | Predicted class for every test email | Final decision | identical |

D.2 then feeds the **manual predictions** into scikit-learn's metric functions (accuracy, balanced accuracy, macro/weighted/micro P-R-F1, Cohen's kappa, MCC, and the confusion matrix). This checks the manual metric code separately from the model.

- **Why we use a tolerance of 1e-9:** the manual code adds `f × log(num/den)` one word at a time. scikit-learn computes `log(num) − log(den)` and multiplies a sparse matrix. Floating-point addition in a different order changes the last digits. A long email has hundreds of thousands of words, so its score can reach −10⁶. A relative error of 1e-9 is about 10 000 times larger than this rounding noise, and far smaller than any real formula error. For example, forgetting α in the denominator changes log P(w|c) by about 10⁻².

### 1.4 Hand calculation on 6 toy emails kept, shown as a table instead of `assert`

- **Why:** the defense may ask for a calculation by hand. B.2 shows the hand fractions (12/1014 and 2/1014, posterior 0.75 / 0.125) next to the code output. It uses a "sesuai" column instead of `assert`, so a mismatch is shown rather than stopping the run.

### 1.5 Using the test set here does not break the protocol

- **The protocol** (ROADMAP) says the test set is used only for the M03 baseline and the final evaluation, so it cannot steer model selection.
- **Why this is still fine:** nothing is selected here. α = 1 is the textbook default and was fixed before any result. No model, parameter or threshold in the final pipeline depends on this notebook. The final model (stacking on V3, b = 1.777) was chosen in M07 on training-data CV and is not touched.
- The challenge was already scored on a test set when it was submitted. This notebook only moves that evaluation onto the baseline's test emails.

### 1.6 Library and manual Naive Bayes are shown side by side (D.4)

- **What:** one table with three Multinomial Naive Bayes models on the same test emails: (1) library + TF-IDF (the baseline), (2) library + word counts, (3) manual + word counts. Both confusion matrices and the training and prediction times are shown next to each other.
- **Why:** each pair of columns answers one question.
  - Columns 2 vs 3 have the same algorithm, input and α, so they test the **implementation**.
  - Columns 1 vs 2 have the same algorithm with a different input, so they test the **representation**.
  - Without column 2, "manual vs baseline" would mix both effects.

### 1.7 The score is not compared with the final model

- Both the manual MNB and the baseline read the raw cleaned text. That text still contains the source-format shortcuts: H4 (Phish keeps real digits) and H5 (Phish had its stopwords removed). So the manual model's score is inflated in the same way as the V0 results in M05.
- The final model reads V3, where these shortcuts are removed. Its test Macro F1 of 0.8913 is therefore **not** worse than a raw-text manual MNB that might score above 0.9. The two numbers measure different tasks.

---

## 2. Results (Colab run, 8 Nov 2026)

**Answer 1: the manual calculation is correct.** All 9 checks passed. The metric code also matches scikit-learn, and the hand calculation matches.

| Item | Value |
|---|---|
| Cleaned data / training / testing | 278,941 / 223,152 / 55,789 |
| Test fingerprint (A.3) | `193b7f8793006d67` |
| Vocabulary size (manual = scikit-learn) | 721,764 (0 words only on one side) |
| Word counts F_w,c compared | 2,165,292, **0 different** |
| Max \|Δ log P(w\|c)\| | 3.6e-15 |
| Max \|Δ log S_c\| | 8.3e-08 absolute, **3.3e-14 relative** (rounding only) |
| Max \|Δ posterior\| | 2.5e-12 |
| Prediction differences | **0 of 55,789** |
| Metrics equal to scikit-learn (D.2) | 13 / 13; confusion matrix 0 cells different |
| Baseline reproduced (D.3)? | **Yes**: 0.8881 / 0.8377 / 0.5316, so the split is identical to M03 |

**Answer 2: library vs manual Naive Bayes on the baseline's test set (D.4)**

| Metric (test) | (1) Library, TF-IDF (baseline) | (2) Library, word counts | (3) Manual, word counts |
|---|---|---|---|
| Accuracy | 0.8881 | **0.9416** | **0.9416** |
| Macro F1 | 0.8377 | **0.9337** | **0.9337** |
| Ham Recall | **0.9932** | 0.9832 | 0.9832 |
| Phish Recall | 0.5316 | **0.9568** | **0.9568** |
| Spam Recall | **0.9051** | 0.8866 | 0.8866 |
| Ham lost (Ham → Phish/Spam) | **174** | 430 | 430 |
| Phish → Ham | 538 | **24** | **24** |
| Spam → Ham | 1,991 | **1,301** | **1,301** |
| Phish → Spam | 3,475 | **346** | **346** |
| Predictions different from (3) | — | **0** | — |
| Training + prediction time | — | 73 s | 89 s (65 + 24) |

- (2) = (3) in every cell, because the predictions are identical on all 55,789 emails. This is the implementation effect: **zero**.
- (1) → (2) is the representation effect: **+0.0961** Macro F1 and +0.4252 Phish Recall, at the cost of +256 Ham lost.
- Speed: the library is slightly faster. Both take under 1.5 minutes on 223,152 emails, because MNB training is a single counting pass.

Other manual metrics: Balanced Accuracy 0.9422, Weighted F1 0.9414, Cohen's kappa 0.9056, MCC 0.9066.

**Reading the results**
- Counts instead of TF-IDF give +0.0961 Macro F1 on the same test emails, with the same algorithm and α. This confirms M04 D.1 (validation: +0.1038) on the test set.
- The price is Ham protection: 430 legitimate emails lost vs 174 for the baseline. For comparison, the final model loses 94 (Final Evaluation).
- **The shortcut is visible inside the model (E):** `this`, `is` and `being` have F_w,Phish = **0**. So P(this|Phish) = 2.1e-7, about 22,000× smaller than for Ham. Any email containing a common stopword is pushed away from Phish. That is H5 (the Phish source had stopwords removed). Part of the 0.9568 Phish Recall comes from detecting the source format, not the content, which is why this score is not compared with the final V3 model.
- Compared with the original challenge's own split: Macro F1 is 0.9346 there and 0.9337 here. The vocabulary is 715,109 there and 721,764 here. The split changes the score by only 0.0009, so the challenge's conclusion still holds.

---

## 3. Likely defense questions

**How do you know your manual Naive Bayes is correct?**
We trained scikit-learn's `MultinomialNB` with the same tokenizer and α on the same training emails, and compared every intermediate value. The vocabulary, class counts and every word count are identical. Priors, likelihoods, scores and posteriors agree to rounding error. Every test prediction is the same. The metric code also gives the same values as scikit-learn's functions on the same predictions.

**Why does scikit-learn give the same answer when it does not compute it the same way?**
Both compute log P(c) + Σ f(w,x) log P(w|c). scikit-learn does it as one matrix product: count matrix × log-probability matrix. We do it with a loop over the words in each email. The math is the same; only the order of additions differs, and that changes only the last digits.

**Why is the manual MNB much better than the baseline if both are Multinomial Naive Bayes with α = 1?**
The only difference is the input. MNB models word **counts**. With TF-IDF, each email's weights sum to a few units, so a whole class has a small total weight compared with the smoothing α|V| added to every word. Phish's word probabilities get flattened, and its distinctive words lose their effect (M04 D.3: Phish TF-IDF mass 207,178 vs smoothing 664,008). With counts, Phish's total is several times the smoothing, so the problem does not occur. M04 D.1 found the same gap on the validation split (counts 0.9323 vs TF-IDF 0.8286).

**Why did the original challenge report 0.9346, and why is this number different?**
The challenge used its own random split. Its test set had the same size and class proportions but different emails. This notebook uses the baseline's split, so this number is the one that can be compared with the baseline.

**Why does the manual model ignore words that are not in the vocabulary?**
They have no estimate in any class. Adding them would multiply all three class scores by the same smoothed value α / (T_c + α|V|), but T_c differs per class, so it would change the ranking in favour of the class with fewer tokens. scikit-learn does the same thing: `CountVectorizer.transform` drops unknown words.

**Which is better, the library or the manual Naive Bayes?**
Neither, in accuracy: they give the same prediction for every one of the 55,789 test emails, so every metric is identical. The library is slightly faster (73 s vs 89 s) and is tested code, so the project's pipelines use scikit-learn. The manual version exists to show that we understand and can reproduce every step of the calculation.

**Then why is the "library" baseline so much worse than your manual model?**
The baseline is a library Naive Bayes, but with TF-IDF input. A library Naive Bayes with word counts gives exactly the manual result (0.9337). So the gap is caused by the input, not by library vs manual.

**Why use logarithms?**
A product of thousands of probabilities around 10⁻⁵ underflows to 0 in floating point. The logarithm turns the product into a sum and keeps the ranking unchanged, because log is monotonic.

---

## 4. References

- C. D. Manning, P. Raghavan, H. Schütze, *Introduction to Information Retrieval*, ch. 13, Cambridge University Press, 2008 (multinomial NB, Laplace smoothing, log space).
- A. McCallum, K. Nigam, "A comparison of event models for Naive Bayes text classification," AAAI-98 Workshop on Learning for Text Categorization, 1998.
- scikit-learn documentation: `MultinomialNB` (`feature_log_prob_`, `class_log_prior_`, `predict_joint_log_proba`) and `CountVectorizer` (default `token_pattern`).
- Course slides: W04 (Naive Bayes classification).
