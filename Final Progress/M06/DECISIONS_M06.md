# M06 — Decisions and Their Reasons

**Meeting goal (W06 "Output Project", slides p. 35–36):**

- A. data preparation without leakage;
- B. clustering: algorithm, elbow/silhouette, chosen K, cluster sizes;
- C. cluster profiles and interpretation;
- D. dimensionality reduction: scree plot, number of components, 2D plot, results with vs without reduction under CV.

**Notebook:** `Group_1_Email_Spam_Detection_M06.ipynb`. **Reading guide:** `../HOW_TO_READ_RESULTS.md`.

> **Status:** run on Colab on 3 Nov; results are in section 2.

---

## 0. Key Terms Explained

### Clustering and K

**Clustering** groups emails *without* looking at their labels. **K is the number of groups we ask the algorithm to make.** K = 4 means "split all 223,152 emails into 4 groups". The right K is not known in advance, so we tried K = 2 to 12 and scored each one.

**K-Means** works like this:

1. Place K centre points.
2. Assign every email to its nearest centre.
3. Move each centre to the middle of its emails.
4. Repeat steps 2–3 until nothing changes.

### Silhouette

Silhouette measures **how clearly an email belongs to its own group rather than the next-closest group.** For one email:

- **a** = its average distance to the other emails in **its own** group;
- **b** = its average distance to the emails in the **nearest other** group;
- **silhouette = (b − a) / max(a, b)**, which always lies between −1 and 1.

| Example | a (own group) | b (next group) | Silhouette | Meaning |
|---|---|---|---|---|
| Well separated | 0.2 | 0.8 | (0.8 − 0.2) / 0.8 = **0.75** | Much closer to its own group: a clear cluster |
| **Our case** | 0.91 | 1.00 | (1.00 − 0.91) / 1.00 = **0.09** | Its own group is barely closer than the next one |
| Wrong group | 0.8 | 0.5 | (0.5 − 0.8) / 0.8 = **−0.375** | Closer to another group: probably mis-assigned |

The silhouette of a whole clustering is the **average over all emails**. Our best is **0.0906 at K = 4**: an email's own group is on average only about 9% closer than the next group, so **the groups overlap heavily**. We chose K = 4 because it had the highest silhouette of K = 2–12, even though that value is still low.

The silhouette compares every email with every other email. For 223,152 emails that is about 50 billion distances, so it is computed on 10,000 random emails.

### Why 0.25 is the threshold

It is a **rule of thumb** from the creators of the method. Rousseeuw introduced the silhouette in 1987, and in *Finding Groups in Data* (1990) Kaufman & Rousseeuw gave this guide for reading the average silhouette:

| Average silhouette | Interpretation (Kaufman & Rousseeuw, 1990) |
|---|---|
| 0.71 – 1.00 | A strong structure has been found |
| 0.51 – 0.70 | A reasonable structure has been found |
| 0.26 – 0.50 | The structure is weak and could be artificial |
| **≤ 0.25** | **No substantial structure has been found** |

It is a widely used convention, not a statistical test. Intuitively, at 0.25 an email's own group is only 25% closer than the next group, which is too small a gap to call the groups separate. Our 0.09 is far below it, so we report "no strong cluster structure".

### The other K-selection metrics (W06 checklist, slides p. 32)

| Metric | What it measures | Better when | What we saw |
|---|---|---|---|
| **Inertia (elbow)** | Sum of squared distances from each email to its group centre. It always drops as K grows, so we look for the K where the drop suddenly slows (the "elbow"). | Clear elbow | No clear elbow; the curve keeps falling steadily |
| **Davies–Bouldin** | For each group, how similar it is to its most similar neighbour (spread ÷ distance between centres), averaged | Lower | Lowest at K = 12, highest around K = 4–7 |
| **Calinski–Harabasz** | Spread *between* groups ÷ spread *within* groups | Higher | Highest at K = 2, then falls |

The four metrics **disagree** on the best K. That disagreement is itself evidence of weak structure: when real groups exist, the metrics usually point to the same K.

### Comparing clusters with the true labels (a check only)

| Term | Meaning |
|---|---|
| **Purity** | Share of the most common class inside a cluster. A 99.8% Phish cluster is almost entirely Phish. |
| **ARI** (Adjusted Rand Index) | How often pairs of emails are grouped the same way in two groupings, corrected for chance. 0 = no better than random, 1 = identical. |
| **NMI** (Normalized Mutual Information) | How much knowing an email's cluster tells you about its label. 0 = nothing, 1 = everything. |

K = 7 had the highest NMI (0.319), so we also profile K = 7 for a more detailed view. These label-based scores are **not** used to choose K, because clustering must not use the answer it is trying to discover.

---

## 1. Decisions

### 1.1 Training data only; labels not used to build clusters or plots

- **Why:** the test set must stay untouched, and an unsupervised analysis must not use the answer it is trying to discover.
- Labels are used only *afterwards*, to check how the clusters relate to the classes.

### 1.2 Features: TF-IDF unigram with `min_df=5`

- **Why:** a word that appears in fewer than 5 messages cannot define a group. Removing such words also saves memory and time in the SVD step.
- TF-IDF (not counts) because clustering compares messages by *similarity*, and TF-IDF down-weights words that appear everywhere.

### 1.3 No `StandardScaler`; normalise each message to length 1 instead

- **Why:** `StandardScaler` subtracts each column's mean. On a sparse matrix with hundreds of thousands of columns, this produces a dense matrix that does not fit in memory.
- TF-IDF is already scaled per message. After SVD we re-normalise each message to length 1 (`Normalizer`), so K-Means' Euclidean distance behaves like **cosine similarity**, the standard similarity measure for text.
- This adapts W06's "standardise first" rule to sparse text.
- **Reference:** W06 slides p. 32 (checklist: scale before K-Means / PCA); Manning et al. (2008) ch. 6 (cosine similarity).

### 1.4 TruncatedSVD instead of PCA

- **Why:** PCA centres every column (subtracts the mean), which also makes the sparse matrix dense. TruncatedSVD is PCA *without* centring and works directly on sparse data. On text it is called **Latent Semantic Analysis (LSA)**.
- **Number of components:** the slides' rule of 90–95% cumulative variance (W06 p. 31) is not feasible for text, because it would need thousands of components.
  - We fit 200 components and cluster on the first 100, a common LSA size.
  - Section F tests 50, 100 and 200 components for classification, which is W06's third option: "validasi tugas hilir" (validate on the downstream task).
- **Reference:** Deerwester et al. (1990); Manning et al. (2008) ch. 18; W06 slides p. 31.

### 1.5 t-SNE only for visualisation

- **Why:** t-SNE keeps similar messages close together, but distances between groups on a t-SNE plot have no meaning. It is never used as a model feature.
- **Reference:** van der Maaten & Hinton (2008); W06 slides p. 29, 34.

### 1.6 K-Means, with K chosen by silhouette

- **Why K-Means:** fast on 223,152 messages, and every cluster has a centre that can be translated back into its top words.
- **Choosing K:** we compute four metrics for K = 2–12, as the W06 checklist asks: inertia (elbow), silhouette, Davies–Bouldin and Calinski–Harabasz. **Silhouette** decides.
  - Silhouette compares every pair of messages, so it is computed on 10,000 random messages.
- **Reading silhouette** (Kaufman & Rousseeuw): ≤ 0.25 means no substantial structure; 0.26–0.50 weak; 0.51–0.70 reasonable; > 0.70 strong.
- **Reference:** W06 slides p. 32; Rousseeuw (1987); Kaufman & Rousseeuw (1990).

### 1.7 ARI and NMI only as a check, not to pick K

- **Why:** they compare clusters with the true labels. Using them to *pick* K would turn the analysis into a supervised one.
- We use the K with the highest NMI only as a second, more detailed profile.
- **Reference:** Hubert & Arabie (1985) for ARI; Strehl & Ghosh (2002) for NMI.

### 1.8 Agglomerative (Ward) as the second algorithm

- **Why:** W06 asks to compare with another algorithm when relevant. Ward merges the most similar groups step by step and does not depend on random starting points.
- It needs memory proportional to the *square* of the number of messages, so it runs on 5,000 random messages with the same K.
- **Reference:** Ward (1963); W06 slides p. 32.

### 1.9 Profile columns that check the *source* of the data

- **Columns:** `escape…` placeholder (%), real numbers (%), the word `enron` (%).
- **Why:** M03's sample rows showed that Ham contains Enron company correspondence, and that some sources replaced numbers with `escapenumber`. If clusters follow these clues, the clusters reflect *where the data came from*, not only what the emails say.
- **Why it matters:** a model can then learn the source instead of the content, and perform worse on real new email. This is the *dataset shift* risk described in our reference paper. If the clues differ strongly between classes, the notebook states **Hypothesis 4**, to be tested in M07.
- **Reference:** Jáñez-Martino et al. (2023), Abstract and p. 1165 (PDF p. 21): filters degraded to error rates as high as 48.81% under dataset shift.

### 1.10 Testing whether SVD helps classification: Logistic Regression, same 5 folds as M05

- **Why Logistic Regression:** SVD features can be negative, and Multinomial Naive Bayes only accepts zero or positive values.
- **Why re-fit TF-IDF and SVD inside each fold:** fitting them on all the training data would let each validation fold shape the features (leakage).
- One 200-component SVD per fold; the 50- and 100-component settings use its first components, because SVD components are ordered by variance. This keeps the run to about 1 hour.
- **Reference:** W06 slides p. 35, step 5 ("Uji manfaat reduksi … memakai CV").

---

## 2. Results (Colab run, 3 Nov; training data, 223,152 emails)

| Item | Result |
|---|---|
| TF-IDF size | 223,152 emails × 143,172 words (`min_df=5`) |
| Variance explained by 100 / 200 components | 16.7% / 21.8% |
| Best silhouette | **0.0906 at K = 4**, so no strong cluster structure (< 0.25). Elbow, Davies–Bouldin and Calinski–Harabasz do not agree on a K either. |
| K with the highest NMI | K = 7 (NMI 0.319); 2 of 7 clusters are ≥ 90% pure (49,900 emails, 22.4%) |
| Pure Phish cluster | **17,281 emails (99.8% Phish), about half of all training Phish.** Top words: `2008 aug 06 wed 07 thu 0000 0300` (email-header timestamps). |
| Pure Ham cluster | 32,619 emails (98.8% Ham); 44.3% contain `enron` |
| `escape…` placeholder: Ham / Phish / Spam | 75.1% / **0.3%** / 67.0% |
| Real numbers: Ham / Phish / Spam | 12.2% / **93.4%** / 10.2% |
| `enron`: Ham / Phish / Spam | 26.7% / 0.0% / 4.2% |
| K-Means vs Ward (5,000 emails, K = 4) | Silhouette 0.093 vs 0.066; agreement between them (ARI) 0.38, so the clusters depend on the algorithm |
| LR Macro F1: no reduction vs best SVD | **0.9655** vs 0.9443 (SVD 200), a difference of **−0.0213**. SVD is not used. |

**What this means:**

1. Email text has no clean natural groups.
2. The clearest groups follow the **data source** (Enron correspondence, one Phish source with timestamp headers), not only the content.
3. **Hypothesis 4** is confirmed as a risk: Phish almost never has the `escape…` placeholder and almost always has real numbers.
   - The timestamp cluster shows that month and weekday names (`aug`, `wed`) are part of the same source pattern.
   - M07 tests how much the model relies on it.

## 3. Likely defense questions

**Why not PCA, as in the slides?**
PCA subtracts each column's mean, which turns our sparse word matrix of hundreds of thousands of columns into a dense one that does not fit in memory. TruncatedSVD is PCA without that step. It is the standard choice for text (Latent Semantic Analysis).

**Why didn't you standardise the features?**
For the same reason: standardising destroys sparsity. TF-IDF already scales each message, and we normalise each message to length 1, so K-Means compares messages by cosine similarity.

**The silhouette is low. Did the clustering fail?**
No. It answers the question: email text does not split into cleanly separated natural groups. Reporting that honestly is part of the W06 checklist ("laporkan jujur"). The profiles still show which clusters are pure and what defines them.

**What does a silhouette of 0.09 mean, and why compare it with 0.25?**
For each email, silhouette compares its average distance to its own group (a) with its distance to the nearest other group (b): (b − a) / max(a, b). At 0.09, an email's own group is only about 9% closer than the next one, so the groups overlap. The 0.25 boundary is Kaufman & Rousseeuw's (1990) rule of thumb: an average silhouette ≤ 0.25 means no substantial structure was found. It is a widely used convention, not a statistical test.

**Why K = 4?**
Of K = 2–12, K = 4 had the highest silhouette. The other three metrics did not agree on a K, which confirms the structure is weak. We also profiled K = 7, where the clusters match the labels best (NMI), to see the groups in more detail.

**Why only 100 components when they explain so little variance?**
Text variance is spread over thousands of directions. We tested the effect directly on classification in section F instead of relying on a variance rule.

**What is Hypothesis 4, and why does it matter?**
If one class comes from a source with a different text format (e.g., numbers not replaced), a model can recognise that format instead of the content. Real incoming email would not have this difference, so real performance could be lower. M07 tests it by making the number format identical for all classes.

---

## 4. References

- Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K. & Harshman, R. (1990). Indexing by latent semantic analysis. *Journal of the American Society for Information Science*, 41(6), 391–407.
- Hubert, L. & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1), 193–218.
- Jáñez-Martino, F. et al. (2023). A review of spam email detection: analysis of spammer strategies and the dataset shift problem. *Artificial Intelligence Review*, 56, 1145–1173.
- Kaufman, L. & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis*. Wiley.
- Manning, C. D., Raghavan, P. & Schütze, H. (2008). *Introduction to Information Retrieval*, ch. 6 and 18. Cambridge University Press.
- Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.
- Strehl, A. & Ghosh, J. (2002). Cluster ensembles — a knowledge reuse framework for combining multiple partitions. *Journal of Machine Learning Research*, 3, 583–617.
- van der Maaten, L. & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605.
- Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. *Journal of the American Statistical Association*, 58(301), 236–244.
- Course slides: W06 p. 29, 31, 32, 34, 35–36.
