# M07 — Decisions and Their Reasons

**Meeting goal (W07 slides p. 2–3):**

- feature selection (filter, wrapper, embedded) and model-based selection;
- fair model comparison;
- bias–variance diagnosis with learning and validation curves;
- data leakage detection (target, group, temporal);
- systematic error analysis;
- reproducibility and experiment documentation.

**Notebook:** `Group_1_Email_Spam_Detection_M07.ipynb`. **Reading guide:** `../HOW_TO_READ_RESULTS.md`.

> **Status:** V3 run finished on 4 Nov (R1). Section 2 = first run (V2 only); **section 2b = V3 run (current)**. Final model: **Stacking MNB + LR + LightGBM**, b = 1.777 (tie-break by business priority). **Final executed notebook: `V3 - RUN FINISHED R2 …`**. `final_config.json` on Drive holds model B.

The order follows the slides: **check leakage first, then diagnose, then analyse errors**. Analysing "errors" that are really leakage artifacts would be wasted effort (W07 p. 4).

---

## 1. Decisions

### 1.1 Test Hypothesis 4 by making the text format uniform (leakage detection)

#### What we suspected

The three classes come from different sources that stored email differently. A model can then learn the *format* instead of the *content*, for example "has real numbers → Phish". Real incoming email does not have these dataset-specific differences.

- This is close to *target leakage* (W07 p. 32): information that predicts the label only because of how the data was assembled, and that will not exist when the model is used.
- The review paper shows that ignoring such dataset differences can make real performance much worse than the estimate (Jáñez-Martino et al., 2023).

#### How the clues were found and pinned down

Five steps, each with its own evidence:

| Step | Evidence | Where |
|---|---|---|
| 1. Spot | Unsupervised clustering produced a 99.8%-Phish cluster (17,281 emails, about half of Phish) whose top words were dates and times (`2008 aug 06 wed 07 thu 0000 0300`), not content | M06 D.2 |
| 2. Count | `escape…` placeholder: Ham 75.1% / Phish 0.3% / Spam 67.0%; real digits: 12.2% / 93.4% / 10.2% | M06 E |
| 3. Locate | Reading the raw emails showed where the dates come from: an email-header timestamp, e.g. `indemand payperview movies sports` **`wed 15 sep 2004 11 33 14 0500`** `hi cable tv subscriber …`. Ham and Spam have header timestamps too, but their source had already replaced the digits with `escapenumber` (`sent friday june escapenumber escapenumber …`). | M07 B.1 (+ raw-data check below) |
| 4. Prove the effect | Remove the clues → the score drops (the ablation test) | M07 B.3 |
| 5. Prove *which* words | List the 25 words that most strongly point to Phish in V0, V1 and V2, and count how many are format tokens (real digits, `escape…`, month/day names). If the clues cause the skew, V0's list is full of them and V2's list is content words. | M07 B.4 |

Steps 4 and 5 together answer "how do you know these features cause the skew?". Step 4 shows removing them changes the result; step 5 shows they were what the model relied on.

**Header leftovers in the raw data.** The dataset is mostly email *bodies*, but some header lines survived. Counted over all 365,448 raw messages:

| Header leftover | Ham | Phish | Spam | What we do |
|---|---|---|---|---|
| Header timestamp with **real digits** | 4.6% | **67.8%** | 1.3% | **Removed** (V2) |
| Header timestamp with digits **masked as `escapenumber`** (5% sample) | ≈ 35% | 0% | ≈ 8% | **Removed** (V2) |
| "original message" (a forwarded/replied email inside the body) | **20.1%** | 0.4% | 0.6% | Kept; see limitations |
| "from … sent … to … subject" block | **9.6%** | 0.0% | 0.2% | Kept; see limitations |
| "received from", "message id", "x mailer", "return path" | 0.6–0.8% | 0–0.3% | 0.4–1.0% | Kept: rare and spread over all classes |

Examples:

- Phish: `indemand payperview movies sports wed 15 sep 2004 11 33 14 0500 hi cable tv subscriber …` (subject, date header, body).
- Ham: `… original message from rodriguez grace sent friday june escapenumber … am to fitzpatrick amy williams bill subject re wolf reply …` (a forwarded Enron email).
- Spam: `… below this line is a copy of the message return path received qmail …` (a bounced message).

#### How the clues are removed: three versions of the text

| Version | Change |
|---|---|
| V0 | Text as is |
| V1 | Punctuation → space; every word containing a digit, and every `escape…` placeholder, becomes `numtoken` |
| V2 | Punctuation → space; **email-header timestamps are removed entirely**; then numbers → `numtoken` as in V1; the remaining month and weekday names → `datetoken` |

V2 is produced by the function `normalize_v2` (notebook B.1). It applies four regular-expression steps, **in this order, to every email of every class**. The original data is not changed; the models are trained on this cleaned copy. The exact patterns are in the notebook and are saved to `final_config.json`.

| # | Step | Pattern (in words) | Replaced with |
|---|---|---|---|
| 1 | Punctuation | any character that is not a letter, digit, underscore or space (`[^\w\s]`) | a space |
| 2 | Header timestamp | [optional weekday] + (`dd month` or `month dd`) + year + [optional time: `hh mm [ss]` or joined `hhmmss`] + [optional `am`/`pm`] + [optional zone: 4 digits or 3 letters]. **Every number slot may be a real number or an `escape…` placeholder**, so masked timestamps are removed too. | nothing (deleted) |
| 3 | Numbers | any `escape…` placeholder (`escapenumber`, `escapelong`, `escapenumbermg`, …) or any word containing a digit (`2008`, `3pm`, `mp3`) | `numtoken` |
| 4 | Dates in the body | month and weekday names, full or short (`monday`, `mon`, `aug`, …), **except `may`**, which is usually the verb | `datetoken` |

**Why the order matters.** Step 2 needs the digits (`05`, `2008`) to recognise a timestamp, so it must run before step 3 turns them into `numtoken`.

**Worked examples** (output of the real function):

| | Phish, header date (real digits) | Ham, forwarded header (masked digits) | Ham, ordinary content |
|---|---|---|---|
| Original | `specialpricespharm tue 05 aug 2008 193121 0400 order now` | `original message from rodriguez grace sent friday june escapenumber escapenumber escapenumber escapenumber am to fitzpatrick amy` | `out of the country until wednesday afternoon , month of july` |
| After step 1 | unchanged | unchanged | `out of the country until wednesday afternoon month of july` |
| After step 2 | `specialpricespharm order now` | `original message from rodriguez grace sent to fitzpatrick amy` | unchanged (no timestamp) |
| After steps 3–4 | `specialpricespharm order now` | `original message from rodriguez grace sent to fitzpatrick amy` | `out of the country until datetoken afternoon month of datetoken` |

The header timestamps disappear from *both* classes. Real content ("meet on monday", "price … usd") keeps its meaning.

#### Why each step is needed

- **Punctuation → space.** The Ham and Spam sources keep punctuation as separate tokens (`sent : monday , october 01`); the Phish source has none. Without this step, the timestamp pattern would match only Phish. The vectorizer already ignores punctuation; we checked that the extracted word tokens are identical either way. So this step changes no model feature, only how the patterns match.
- **Delete the whole timestamp (not just its numbers), in every class.** Two earlier designs failed in sample test runs, and both failures were caught by the feature checks:
  1. *Replace digits and date words only.* `datetoken` became LightGBM's #2 feature: a Phish timestamp still turned into `datetoken numtoken datetoken numtoken …`, a sequence only one Phish source produced.
  2. *Delete only timestamps with real digits.* Phish timestamps disappeared, but Ham's masked ones (`friday june escapenumber …`) stayed. Date words then pointed to **Ham**, the same shortcut in reverse.

  A timestamp is header data, not content, so V2 deletes it in every class, whether its digits are real or masked. Checked on a 5% sample of the raw data:

| | Ham | Phish | Spam |
|---|---|---|---|
| Header timestamp, real digits only | 4.2% | **66.7%** | 1.2% |
| Header timestamp, real **or masked** digits | **39.3%** | 66.7% | 9.0% |
| Year "2008", before → after removal | 0% → 0% | **51.0% → 11.2%** | 0% → 0% |
| Month/weekday name, raw | 56.6% | 71.2% | 17.4% |
| … after removing real-digit timestamps only | 54.5% | 10.0% | 16.6% |
| … after removing **all** timestamps (V2) | **33.0%** | **10.0%** | **12.1%** |

  What is left is ordinary content ("until wednesday afternoon", "for the month of july", "this friday during your visit"). Business email talks about dates more often, which is a fair Ham signal.
- **Numbers → one token, not deleted.** One token keeps the information that *a number was there* (prices, amounts, codes), which can be a real signal, while removing the source-specific format.
- **Date words → one token.** Dates written inside the body then look the same in every class.

#### Limits of this cleaning (for the report)

- **The timestamp pattern does not catch every date format.** "2008" still remains in about 11% of Phish (down from 51%), written in other ways.
- **A full date inside the body is also removed** if it looks like a header date (e.g. `10 may 2005`). This costs very little, because a full date is rarely what makes an email phishing.
- **Other source clues are kept on purpose:**
  - Ham's forwarded "original message … from … sent … to … subject" blocks (20.1% of Ham). Reply chains are normal in work email, so they are a fair Ham signal, but here they come mostly from one company (Enron).
  - The word `enron` itself.

  Both are reported as dataset limitations: the model may treat "looks like a forwarded internal email" as Ham more easily than it should on other organisations' email.

#### Decision rule, fixed before seeing results

- The final model uses the most uniform text version (**V3** since the update in 1.1b; V2 in the first run), because source formatting is not available on real email.
- The drop V0 → V2 is reported as how much the earlier models relied on the artifact.
- Hypothesis 4 counts as supported if Macro F1 drops by more than 0.005 for either model. That is about five times the fold-to-fold standard deviation in M05.

**What the checks showed on the 12% sample (final V2 design).**

- **B.4:** among the 25 words that most strongly point to Phish, **16 are format clues in V0** (`2008`, `30303`, `358`, `125000`, URLs with dates) and **0 in V1 and V2**.
- **LightGBM gain:** `datetoken` fell from #2 (first design) to **#12** (about 1% of the total gain).
- After cleaning, the top of the B.4 list is one repeated campaign: a fake "CNN Daily Top 10" newsletter (`cnncom`, `montauk`, `caylee`, …), whose words appear in up to about 9% of Phish. Those words are content, but they reflect **one template sent hundreds of times**, so they are a near-duplicate issue (section C), not a format issue. B.4 now prints how much of Phish its top words cover, so this is visible in the notebook.

**Why choose V2 even though it scores lower?** The guideline says the best model is the one that best fits the problem's goal, *not* the one with the highest number (§10, §23). A higher score that comes from an artifact would not hold on real email. On the 12% sample, V2 lowered Macro F1 by 0.044 (MNB) and 0.032 (LR); the official numbers come from the Colab run.

**Cost-saving choice.** The Hypothesis 4 comparison uses unigram features for MNB. Building word-pair matrices for all three versions would roughly triple the preparation time, and the question is only *how much* the format matters. The final MNB on V2 still uses unigrams + bigrams.

### 1.1b Hypothesis 5: stopwords and glued links (added after the first run)

**What we found.** Reading the error examples of the first run showed a Ham email with no stopwords (`let know need help w cooking antibiotics …`) predicted as Phish. Counting on the raw data (5% sample) confirmed a second source difference:

| | Ham | Phish | Spam |
|---|---|---|---|
| Contains "the" | 71.2% | **0.0%** | 56.9% |
| Words from the NLTK stopword list (179 entries, 153 unique words) | 25.4% of all words | **0.00%** | 26.4% |
| Words only on scikit-learn's longer list (e.g. `please`) | 4.3% | 6.8% | 4.3% |
| Links glued into one word (`httpwwwsitecom`) | 0.0% | **38.1%** | 0.0% |
| Contains any punctuation | 61.7% | **0.0%** | 54.1% |

The first run already hinted at it: chi-square's top words were `the`, `to`, `of`, `and` (all Ham), and LightGBM's most-used words included `to` and `the`.

**Hypothesis 5.** The model recognises Phish by the *absence* of stopwords and by glued links, not by content.

**V3 = V2 + two steps, applied to every class:**

| Step | Change | Why |
|---|---|---|
| Links | any word starting with `http`, separate or glued → `urltoken`; standalone `www`, `com`, `net`, `org`, `edu`, `gov` deleted | Glued links exist only in Phish; separate `www`/`com` tokens exist only in Ham and Spam |
| Stopwords | delete the **NLTK** English list (179 entries, 153 unique words without apostrophes) | The Phish source contains **0.00%** NLTK stopwords but still has words from longer lists (`please` appears 825 times in a 5% sample). So NLTK is exactly the list its creators removed. Using the longer scikit-learn list would also delete content words like `please`, `call`, `bill`, `system`. |

The link step runs before the number step, because glued links contain digits.

**Verified on real messages** (`Final Evaluation/test_sanitization.py`): timestamps, glued and separate links, numbers and `net`/`com` are handled, and content words such as `please`, `call` and `system` are kept.

**How it is tested in the notebook.**
- **B.3:** MNB and LR on V0, V1, V2 and V3. Hypothesis 5 counts as supported if Macro F1 drops by more than 0.005 from V2 to V3.
- **B.4:** the top Phish words, now with a V3 column.
- **B.5 (new):** the 15 highest-weight words per class for LR on V2 and on V3 (fold 1), with stopwords in bold, counted per class. If Hypothesis 5 is right, stopwords appear in V2's list and disappear in V3's.

**Side effects (for the limitations):** negation words (`not`, `no`, `nor`) are removed with the other stopwords. Glued words from the Phish source (e.g. `fastshippinginternetpricesfdaapproved`) cannot be split.

### 1.2 Near-duplicate check (group leakage)

**What.**

- Take 3,000 random validation emails from fold 1.
- Find each one's most similar training email (TF-IDF cosine similarity, from 0 to 1).
- Count those with similarity ≥ 0.9 as near-duplicates.

**Why.** M03 removed only *identical* texts. Emails from the same template with a different name or number can still sit on both sides of the split. The slides' group-leakage case study shows near-identical samples from the same source inflating validation scores (W07 p. 34).

**Why 3,000 and one fold.** Comparing every validation email with every training email is about 8 billion similarity calculations per fold. A random sample of 3,000 gives the proportion with good precision.

**Use.** Section G compares the final model's error rate on near-duplicates vs other emails, to show how much they flatter the score.

### 1.3 Validation curve for MNB's `alpha` (0.25 – 8)

**Why.** In M05 the best α (1.0) was at the top edge of the tested range, so a larger value might be better. A validation curve sweeps one complexity parameter at fixed data size (W07 p. 28).

**Rule.** Highest mean Macro F1; ties (< 0.001) go to the higher Ham Recall.

### 1.4 Validation curve for LightGBM's number of trees

**What.** One LightGBM with 150 trees on fold 1, scored after every 25 trees, on both the validation fold and 20,000 training emails.

**Why.**

- M05's 150 trees came from a sample test, not from the full data.
- The gap between the training and validation curves shows overfitting (W07 p. 25: "lihat GAP").
- Reading the curve from one trained model costs no extra training.

**Rule.** The smallest number of trees within 0.001 of the best validation Macro F1, because fewer trees means faster training. The fold-1 model is reused in the final run.

### 1.5 Learning curves for MNB and LR (10%, 25%, 50%, 100% of fold 1's training data)

**Why.** A learning curve answers "would more data help?" (W07 p. 26–27):

- training and validation scores converge low → high bias;
- training score much higher than validation → high variance;
- validation still rising at 100% → more data would help.

### 1.6 Feature selection: chi-square filter (k = 10k, 50k, 200k, 1M, or all)

**Why chi-square.**

- It is a *filter* method: fast, and independent of the model.
- It suits non-negative features such as word counts (W07 p. 7–8).
- Wrapper methods such as RFE (recursive feature elimination) would need thousands of model fits on millions of features, which is not feasible here (W07 p. 11).

**Leakage control.** Words are selected inside each fold, on that fold's training part only.

**Rule.** The smallest k within 0.001 of "all words". Fewer features means a simpler, faster model. Values of k larger than the vocabulary are skipped, because they would select every word.

### 1.7 Model-based importance: LightGBM gain

**What.** The 20 words that reduce LightGBM's error the most when it splits on them.

**Also used as a leakage check.** This list is how we found that the first V2 design still leaked through `datetoken`. Expect source words such as `enron` here too. Enron correspondence is a large share of Ham, so the company name works as a source signal. That is a dataset limitation for the final report: a model trained on one company's email will learn that company's vocabulary.

In the 12% sample test of the final V2 design, `datetoken` was still the #2 gain feature. But chi-square shows it now occurs mostly in **Ham** (date words after timestamp removal: Ham ≈ 55%, Phish ≈ 10%), i.e. meeting days and leftovers of forwarded headers in work correspondence. It no longer marks the Phish source. We keep it, because removing every date word would also remove real content. This is listed as a dataset limitation.

**Why.** This is the embedded / model-based view (W07 p. 13). The slides warn that tree importance can be biased and recommend permutation importance to confirm it (p. 13–15). Gain only shows *which* words the model uses, not *in which direction*. Permutation importance and direction belong to the final interpretation step, after the final model is fixed.

### 1.8 Final stack on V3 (V2 in the first run), with ablation (stacking without MNB)

**What.** Four stacks were compared at Ham Recall 99.5%:

- MNB + LR + LightGBM;
- LR + LightGBM;
- MNB + LightGBM;
- MNB + LR.

The three single models are compared alongside them.

**Why the ablation.** M05 never showed that MNB adds anything to the stack. Removing it and comparing at the same Ham Recall answers that directly.

**Rule.** Highest Macro F1 at the Ham target. Ties (< 0.001) follow the business priorities from M05, in order:

1. no legitimate email lost (already equal, because every finalist is tuned to Ham Recall 99.5%);
2. fewest phishing emails in the inbox;
3. fewest spam emails in the inbox;
4. fewest base models.

Phishing filed as Spam is not a criterion, because those messages are still kept out of the inbox. A bootstrap of the training data (1,000 resamples) checks that the inbox difference between the two top stacks is not luck.

*Change from the first V3 run:* that run used "fewer base models" as the only tie-break, and selected LR + LightGBM. Comparing the two tied stacks in detail (section 2b) showed the 3-model stack removes more harmful email at no cost in legitimate email. Simplicity alone ignored the project's own priorities, so the tie-break was changed to follow them. Simplicity stays as the last criterion.

### 1.9 Save the results

**What.**

- out-of-fold predictions;
- the final configuration (`final_config.json`);
- an experiment log;
- `requirements.txt`.

On Colab these are saved to Google Drive.

**Why.**

- The W07 reproducibility checklist asks for fixed seeds, `requirements.txt`, an experiment log and a model card (p. 48).
- Saving the predictions means the final test evaluation does not need to repeat hours of cross-validation.

### 1.10 Error analysis by message type

**What.** The final model's three priority errors, broken down by:

- message length (≤ 10, 11–50, 51–200, > 200 words);
- non-ASCII characters;
- near-duplicate status.

Individual examples are also shown.

**Why.** The slides ask us to read the individual samples behind the FN/FP cells, group them into categories, measure each category's share, and turn that into a concrete improvement (W07 p. 38, 41–42).

---

## 2. Results of the first run (V2 only; Colab, 4 Nov; training data, 223,152 emails, 5-fold CV)

| Item | Result |
|---|---|
| **Hypothesis 4**: Macro F1 V0 → V2 | MNB 0.9344 → 0.8900 (**−0.0444**); LR 0.9687 → 0.9423 (**−0.0263**). **Supported.** |
| Phish Recall (LR), Phish with vs without digits | V0: 0.963 vs 0.650; V2: 0.897 vs 0.663. Phish that contain no digits (6.5% of Phish) were always hard; the drop comes from Phish *with* digits. |
| Header timestamps per class (any digits) | Ham 41.1% / **Phish 67.6%** / Spam 10.8%. Date words after removal: 33.8% / 10.6% / 12.0% |
| **B.4**: format clues in the top-25 Phish words | **V0: 17 → V1: 0 → V2: 0** |
| B.4: what is left at the top | A fake "CNN Daily Top 10" template (`cnncom` in 2,802 Phish emails). The top-25 V2 words cover only 11.5% of Phish, so it is a repeated template, not a general pattern. |
| **Near-duplicates** (cosine ≥ 0.9, 3,000 validation emails, fold 1) | **51.7%** (Ham 50.9%, Phish 35.9%, Spam 58.5%); nearly identical (≥ 0.99): 20.3% / 12.7% / 27.7% |
| Near-duplicates: are they easier? | Final model error 3.81% on near-duplicates vs 4.28% on the rest, so only slightly easier |
| α (0.25 – 8) | **1.0** (inside the range). Larger α keeps more Ham (0.995 at α = 8) but Phish Recall collapses (0.56). |
| LightGBM trees (25 – 150) | **150** (top of the range; validation still rising +0.002 per 25 trees). Train–validation gap 0.025. |
| Learning curve (50% → 100% data) | MNB +0.0065, LR +0.0078 validation Macro F1, so more data still helps. LR train ≈ 0.996 vs validation 0.943 (gap 0.053): high variance. MNB train only 0.938: limited capacity (bias). |
| Chi-square | No k kept Macro F1 (best: 1M words 0.9092 vs all 0.9127), so all words are used |
| LightGBM gain, top words | `to`, `enron`, `original`, `the`, `thanks`, `http`, `org`, `wrote`, `datetoken` (#9, points to Ham), … Many are work-correspondence / reply-chain words (dataset limitation). |
| Base models on V2 (default threshold) | MNB 0.9127, LR 0.9423, LightGBM 0.9538 |
| Stacks (default threshold) | MNB+LR+LGBM **0.9589**, LR+LGBM 0.9588, MNB+LGBM 0.9557, MNB+LR 0.9436 |
| **Final model at Ham Recall 99.5%** | **Stacking MNB + LR + LightGBM, b = 1.625, Macro F1 0.9551** (± 0.0010). Lost: 511 of 102,192 Ham. Into the inbox: 516 Phish (1.5% of Phish), 2,309 Spam (2.7% of Spam). |
| Does MNB help the stack? | Slightly. At the Ham target: +0.0018 Macro F1 vs LR + LightGBM (0.9551 vs 0.9533); Phish into inbox 597 → 516. At the default threshold the two are tied (0.9589 vs 0.9588). |
| Where errors concentrate | **≤ 10-word messages**: Ham lost 6.09% (vs 0.50% overall), Phish into inbox 8.14%, Spam into inbox 9.22%. Non-ASCII Ham lost 1.00% vs 0.47%. |

**Compared with M05 (V0 text), at the same Ham target:**

| | M05 (V0) | M07 (V2) |
|---|---|---|
| Macro F1 | 0.9784 | **0.9551** |
| Phishing into inbox | 86 | **516** |
| Spam into inbox | 2,146 | 2,309 |

The M05 numbers were inflated by the format shortcut. Without it, about 6× more phishing reaches the inbox. **0.9551 is the honest estimate.**

**What the error examples show** (G):

- Many "lost" Ham emails are commercial newsletters or registration mails (cheapflights, hotwire, sears) that look like spam, plus a few binary attachments read as text. The label itself is debatable for these.
- Many Phish that reach the inbox are pharmacy offers (`pharmacyphentrimine…`, `fastshippinginternetprices…`), i.e. spam-like content inside the Phish class. The "Phish" label in this dataset follows its source corpus, not a strict phishing definition. Both go into Limitations.

**Process notes.**

- The run took about 4.5 h; LightGBM alone took 156 min for folds 2–5 plus 39 min for the fold-1 curve.
- The artifact folder was **not** on Google Drive (`m07_artifacts` = Colab's temporary storage), so the saved files are lost unless they were downloaded before the session ended.

## 2b. Results of the V3 run (Colab, 4 Nov; computed from the saved out-of-fold predictions)

**Reproducibility:** V0, V1 and V2 give exactly the same numbers as the first run (e.g. LR V2 0.9423, MNB V2 0.8900), so V3 is the only new factor.

**Hypotheses 4 and 5** (default threshold, 5-fold mean):

| Model | V0 | V1 | V2 | V3 | H4: V0 → V2 | **H5: V2 → V3** |
|---|---|---|---|---|---|---|
| MNB (unigram) Macro F1 | 0.9344 | 0.8956 | 0.8900 | 0.8016 | −0.0444 | **−0.0884** |
| LR Macro F1 | 0.9687 | 0.9476 | 0.9423 | 0.8658 | −0.0263 | **−0.0766** |
| MNB Phish Recall | 0.984 | 0.947 | 0.937 | 0.730 | | |
| LR Phish Recall | 0.943 | 0.892 | 0.882 | 0.678 | | |

- **Both hypotheses are supported.** The stopword/link shortcut (H5) was worth even more than the number shortcut (H4).
- From V0 to V3, MNB loses 0.133 and LR 0.103 Macro F1. Most of the earlier "Phish detection" came from the Phish source's preprocessing, not from what the emails say.

**Tuning on V3:**

- Best α = **2.0** (inside the 0.25–8 range).
- LightGBM: **125** trees.
- Chi-square: no k kept Macro F1, so all words are used.

**Models at Ham Recall 99.5%** (511 of 102,192 Ham lost in every row):

| Model | b | Macro F1 | Phish Recall | Phish Precision | Phish → Ham | Phish → Spam | Spam → Ham |
|---|---|---|---|---|---|---|---|
| MNB (α = 2) | 21.49 | 0.7815 | 0.542 | 0.762 | 5,274 | 10,422 | 15,373 |
| LR | 2.84 | 0.8555 | 0.667 | 0.784 | 911 | 10,514 | 4,594 |
| LightGBM | 1.47 | 0.8684 | 0.668 | 0.905 | 1,449 | 9,943 | 7,689 |
| **Stacking MNB + LR + LightGBM (selected)** | 1.78 | **0.8855** | 0.678 | 0.893 | **582** | 10,444 | **3,089** |
| Stacking LR + LightGBM | 1.93 | **0.8847** | 0.680 | 0.894 | 746 | 10,212 | 3,545 |
| Stacking MNB + LightGBM | 1.68 | 0.8775 | 0.679 | 0.884 | 739 | 10,270 | 4,879 |
| Stacking MNB + LR | 1.86 | 0.8666 | 0.620 | 0.882 | 676 | 12,346 | 3,410 |

**Selection: Stacking MNB + LR + LightGBM, b = 1.777.** The two best stacks are tied on Macro F1 (difference 0.0008 < 0.001), so the business priorities decide (rule in 1.8):

| At Ham Recall 99.5% | A: LR + LightGBM | **B: MNB + LR + LightGBM** | B − A |
|---|---|---|---|
| Legitimate email lost | 511 (22 → Phish, 489 → Spam) | 511 (23 → Phish, 488 → Spam) | 0 (457 of the 511 are the same emails) |
| Phishing into the inbox | 746 | **582** | **−164** (95% bootstrap: 129–198 fewer) |
| Spam into the inbox | 3,545 | **3,089** | **−456** (95% bootstrap: 394–524 fewer) |
| Phishing filed as Spam | 10,212 | 10,444 | +232 (still kept out of the inbox) |
| Macro F1 | 0.8847 | 0.8855 | +0.0008 (B higher in 4 of 5 folds) |

The advantage grows with stricter Ham protection:

| Ham Recall target | Phish into inbox (A / B) | Spam into inbox (A / B) | Macro F1 (A / B) |
|---|---|---|---|
| 99.0% | 271 / **235** | 1,325 / **1,283** | **0.8919** / 0.8909 |
| 99.5% | 746 / **582** | 3,545 / **3,089** | 0.8847 / **0.8855** |
| 99.9% | 5,528 / **3,912** | 19,414 / **15,040** | 0.8140 / **0.8344** |

Cost of B: one more base model, about 50 minutes more to build MNB's word counts, and more memory. MNB alone is the weakest model (0.7815), but in the stack it adds a different view of the data that reduces harmful email in the inbox.

The first V3 run (`V3 - RUN FINISHED R1`) still shows A, because it used the old tie-break. The re-run of the updated notebook shows B.

**How the honest numbers compare:**

| At Ham Recall 99.5% | M05 (V0) | M07 run 1 (V2) | **M07 (V3)** |
|---|---|---|---|
| Macro F1 | 0.9784 | 0.9551 | **0.8855** |
| Phishing into the inbox | 86 | 516 | **582** (1.7% of Phish) |
| Phishing filed as Spam | — | 2,384 | **10,444** (30% of Phish) |
| Spam into the inbox | 2,146 | 2,309 | 3,089 (3.6% of Spam) |

**Reading.** Without the source shortcuts, about **30% of Phish is classified as Spam**: still filtered, but without a phishing warning (priority 3). Separating Phish from Spam by content alone is hard in this dataset. Part of the Phish class is pharmacy spam (see the error examples), so the two classes overlap in content. Phishing that reaches the inbox stays low (1.7%).

**Notebook outputs** (`V3 - RUN FINISHED R2 …ipynb`, the final run of the updated notebook; sections B–E are identical to R1):

- **B.5, stopwords among LR's 15 top-weighted words per class (fold 1):**

  | | Ham | Phish | Spam |
  |---|---|---|---|
  | V2 | 3 (`to`, `the`, `for`) | 0 | **8** (`and`, `no`, `your`, `to`, `you`, `of`, `our`, `here`) |
  | V3 | 0 | 0 | 0 |

  On V2 the model recognised Spam largely *by its stopwords*, because Phish had none. This is the direct evidence for Hypothesis 5. (B.5 refits LR each run, and near-equal weights can swap places: the first V3 run showed 4 and 10, R1 and R2 show 3 and 8. The conclusion is the same in every run: V2 has stopwords at the top, V3 has none.)
- **Remaining source fingerprints in V3** (same table). These are a dataset limitation, not something cleaning can remove word by word:

  | Class | Fingerprint words | Origin |
  |---|---|---|
  | Phish | `josemonkeyorg`, `monkeyorg` | Recipient address of the phishing corpus the class came from |
  | Phish | `contenttype`, `contenttransferencoding` | Email headers glued by the Phish source |
  | Phish | `nan` | A missing value written as text |
  | Ham | `enron`, `vince`, `louise`, `houston` | Enron staff and locations |
  | Spam | `uwaterloo` | The source corpus's university |

  Each class is still partly identifiable by its source corpus, which is exactly the dataset-shift risk in Jáñez-Martino et al. (2023).
- **Format clues in the top-25 Phish words (B.4):** V0 17, V1 0, V2 0, V3 0. The remaining top words come from the fake "CNN Daily Top 10" template (11.5% of Phish).
- **Per-class format table (B.1):** "the" in Ham 65.6% / Phish 0.0% / Spam 51.8% (V0), 0% in every class (V3). Glued links: Phish 39.1%, others 0%. Punctuation: Phish 0%.
- **Near-duplicates (C):** 55.2% of the fold-1 sample have a near-copy (≥ 0.9) in training. With the final model, the error rate on them is 7.37% vs 7.58% on the rest. With the shortcuts removed, near-copies **barely flatter the score**.
- **LightGBM trees (D.2):** validation 0.8446 (25) → 0.8875 (125) → 0.8868 (150), so **125** is chosen. Train–validation gap 0.029.
- **Learning curves (D.3):** 50% → 100% data adds +0.014 (MNB) and +0.008 (LR; 0.8554 → 0.8630). Gaps at 100%: MNB 0.047, LR 0.090 (0.9533 vs 0.8630), so LR overfits. More data still helps both.
- **LightGBM gain (E.2):** `subject`, `enron`, `thanks`, `datetoken`, `cnncom`, `urltoken`, `numtoken`, `wrote`, `original`, `attached`, … These are mostly correspondence and source words.
- **Error examples (G, final model B):**
  - **Lost Ham** includes 1-word and 5-word messages (`quotesen`, "when you have a minute"), a foreign-language forum digest, and a **Yahoo "verify your account, click the link" email labelled Ham**. That last one looks exactly like phishing, so some of the 511 "losses" are arguably label noise.
  - **Missed phishing** includes real attacks: "dear user due to congestion … click to activate account", "card currently suspended", and a fake "system maintenance" notice from IT.
  - **Pharmacy spam labelled Phish** also slips through (`canadianremedymeds`, `worldwidehealthfastshipping`). So do a university "mail delivery failure" notice and a CRM sales email, which look more like Ham or Spam than phishing.
- **Chi-square (E.1):** no k keeps Macro F1 (k = 1,000,000: 0.8330 vs all words 0.8356), so all words are used. The top chi-square terms are placeholders and template words: `numtoken`, `enron`, `urltoken numtoken` (Phish), `datetoken`, `cnncom`, `daily top`.
- **Error rates by group (G, final model B):**

  | Group | Messages | Ham lost | Phish into inbox | Spam into inbox |
  |---|---|---|---|---|
  | ≤ 10 words | 5,851 | **6.28%** | **15.25%** | **12.85%** |
  | 11–50 words | 51,509 | 0.68% | 2.01% | 4.29% |
  | 51–200 words | 93,312 | 0.26% | 1.47% | 2.77% |
  | > 200 words | 72,480 | 0.38% | 0.79% | 2.98% |
  | ASCII only | 204,571 | 0.48% | 1.64% | 3.94% |
  | Contains non-ASCII | 18,581 | 0.78% | **6.11%** | 1.34% |

  Very short messages are the main weakness. Phishing with non-ASCII characters (e.g. `mességéµ centéµr`) slips through 3.7 times as often.
- **Completeness of R2:** every code cell has output (the only empty cell defines helper functions and prints nothing). No errors.

---

## 3. Likely defense questions

**Your rule said ties go to the simpler model. Why did you pick the 3-model stack?**
The tie-break was changed after a detailed comparison, and we say so openly. Both stacks lose exactly 511 legitimate emails. The 3-model stack lets 164 fewer phishing and 456 fewer spam emails into the inbox. A bootstrap shows the difference is not luck (129–198 and 394–524 fewer). "Simpler" was only meant to stop picking a bigger model for a meaningless gain. Here the gain is in exactly what the project cares about, so the business priorities from M05 come first and simplicity is the last criterion.

**Doesn't B file more phishing as Spam?**
Yes, 232 more (10,444 vs 10,212). Those emails are still kept out of the inbox. They only lose the phishing warning, which is the least harmful of the errors we listed.

**Your scores dropped after normalising the numbers. Isn't that worse?**
The higher score partly came from a formatting difference between the dataset's sources, not from what the emails say. Real email does not have that difference, so the lower V2 score is the more honest estimate. The guideline values a justified model over the highest number.

**Why not remove numbers altogether?**
Replacing them with one token keeps the information that *a number was there*, which can be a real signal (prices, amounts, codes), while removing the source-specific format.

**Why remove the header timestamps instead of just the date words?**
About two-thirds of Phish contain an email-header timestamp with real digits, such as `tue 05 aug 2008 193121 0400`. Ham and Spam have header timestamps too, but their source had already replaced the digits with `escapenumber`. Our sample test runs showed two failed shortcuts:

- replacing only digits and date words still left a Phish-only sequence (`datetoken numtoken datetoken …`);
- removing only real-digit timestamps left Ham's masked ones, so dates then pointed to Ham.

A timestamp is header data, not message content, so we remove it completely in every class, whether its digits are real or masked.

**How exactly did you remove the timestamps and numbers?**
With four regular-expression steps applied to every email in the same order: punctuation → space; delete header timestamps (weekday, day, month, 4-digit year, time, zone); every number or `escape…` placeholder → `numtoken`; remaining month/weekday names → `datetoken`. The timestamp step runs before the number step because it needs the digits to recognise a date. The patterns are in the notebook (B.1) and saved in `final_config.json`.

**Did you also remove Ham's forwarded-email headers?**
No. Reply chains ("original message … from … sent … subject") are normal in real work email, so they are a fair signal. But they come mostly from one company (Enron), so we list them as a dataset limitation instead.

**Why remove stopwords? Aren't they harmless?**
In this dataset they are not harmless. The Phish source had already removed them ("the" appears in 0% of Phish vs 71% of Ham), so their presence or absence tells the model which source an email came from. Removing the same list from every class takes that clue away.

**Why the NLTK list and not scikit-learn's?**
Phish contains none of the NLTK words but still contains words from scikit-learn's longer list, such as `please`. So NLTK is the list the source used. scikit-learn's list would also delete real content words like `please`, `call` and `system`.

**What is a near-duplicate, and why does it matter?**
Two emails that are almost the same, e.g. one template with different names. If one is in training and the other in validation, the model is tested on something it has nearly seen, which inflates the score (group leakage, W07 p. 34).

**Why chi-square and not RFE?**
RFE refits the model many times while removing features. With millions of word features that is not feasible. Chi-square scores every word once, works for count data, and is applied inside each fold.

**Why is the learning curve done on only one fold?**
It needs four training runs per model. One fold shows the trend, and the 5-fold CV elsewhere covers stability.

**Does MNB actually help the final model?**
Answered by the ablation in F.3. If removing MNB changes nothing at the same Ham Recall, the simpler stack is chosen.

---

## 4. References

- Jáñez-Martino, F. et al. (2023). A review of spam email detection: analysis of spammer strategies and the dataset shift problem. *Artificial Intelligence Review*, 56, 1145–1173. Abstract and p. 1165 (PDF p. 21): error rates up to 48.81% under dataset shift.
- Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS 30*.
- Mitchell, M. et al. (2019). Model cards for model reporting. *FAT\* 2019* (W07 slides p. 47, 51).
- Course slides: W07 p. 2–4, 7–8, 11, 13–15, 25–28, 32, 34, 38, 41–42, 47–48.
- Guideline Final Project §10, §11, §15, §23.
