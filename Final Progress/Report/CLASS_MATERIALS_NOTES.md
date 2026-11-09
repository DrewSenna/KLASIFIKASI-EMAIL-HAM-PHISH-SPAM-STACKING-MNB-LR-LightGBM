# Class Materials: What the Final Report Must Show

Read on 6 Nov 2026 from `Class Materials/` (guideline, contract, W01–W07, M05 table) and `PDF REPORT FILE REFERENCE FORMAT.pdf`. Slide page numbers are PDF pages, checked with `pdftotext`.

---

## 1. Required report structure (Guideline §17)

The current draft `Laporan_UTS_Group_1.md` already follows this order: chapters 1–17 plus References.

| § | Required chapter | Required content |
|---|---|---|
| 1 | Abstract | — |
| 2 | Introduction | Latar belakang; masalah; tujuan |
| 3 | Problem Formulation | Definisi masalah; target; fitur; tujuan evaluasi |
| 4 | Dataset and Data Understanding | Tahap 2 list: sumber, unit observasi, jumlah data, fitur, target, kualitas, *missing values*, pencilan, distribusi, potensi bias/keterbatasan |
| 5 | Data Pre-processing | Including how *data leakage* is prevented |
| 6 | Feature Engineering / Feature Selection | Why each feature is used, "bukan hanya karena nilainya lebih tinggi" (§7 Tahap 4) |
| 7 | Experimental Design | — |
| 8 | Baseline and Candidate Models | The baseline answers: "Apakah model yang lebih kompleks benar-benar memberikan perbaikan?" |
| 9 | Validation and Evaluation | — |
| 10 | Error Analysis | §11: jenis kesalahan; data yang sering salah; kelas tersulit; kemungkinan penyebab; fitur atau kualitas data?; dapat diperbaiki? |
| 11 | Hyperparameter Tuning | §12: parameter, rentang nilai, metode pencarian, strategi validasi, hasil, konfigurasi terbaik |
| 12 | Model Comparison | §10: model, alasan, hasil, strategi validasi, metrik, kelebihan/keterbatasan, *trade-off*. "Model terbaik = paling sesuai tujuan, bukan angka tertinggi." |
| 13 | Final Model Selection | §13: performa, stabilitas, generalisasi, jenis kesalahan, kompleksitas, interpretabilitas, kebutuhan masalah, biaya kesalahan, keterbatasan dataset. Every claim must trace back to an experiment. |
| 14 | Model Interpretation | §14: koefisien, *feature importance*, *permutation importance*, contoh prediksi benar dan salah |
| 15 | Limitations | — |
| 16 | Reproducibility and Documentation | §15 and contract item 8 list 11 things to document: dataset/sumber, preprocessing, *feature engineering*, pembagian data, algoritma, parameter, *hyperparameter*, metrik, hasil, perbandingan model, keputusan pemilihan |
| 17 | Conclusion | — |
| — | References | — |

**Draft note:** in the draft, the "Rencana Pengembangan Setelah UTS" section is numbered 16, so Reproducibility is §15 and Conclusion is §17. That plan is required for the UTS (§8, item 12) but is not one of the §17 headings. For the final report, either keep it as an extra section or fold it into the Conclusion. Make sure Reproducibility ends up numbered as the guideline's 16th item.

## 2. UTS minimum (Guideline §8, Milestone 1)

1. Problem formulation
2. Dataset dan sumber
3. Data understanding
4. EDA
5. Pre-processing
6. Feature engineering awal
7. Baseline
8. Beberapa kandidat model awal
9. Strategi evaluasi
10. Hasil eksperimen awal
11. Analisis awal
12. **Rencana pengembangan setelah UTS** (still marked "*Disusun terpisah*" in the draft)

UTS principle: "Proyek benar-benar telah berjalan dan menghasilkan bukti awal."

## 3. How the lecturer grades (Guideline §2.4, §23, §24)

- The required chain is **Problem → Data → Feature → Model → Evaluation → Evidence → Decision** (§2.4). The report may not say "Model X 95%, sehingga dipilih."
- A lower score is not penalised if the group shows good problem formulation, correct experiments, appropriate evaluation, error analysis, model comparison, a reasoned selection, and clear limitations.
- A high score is penalised if there is leakage, wrong evaluation, no reproducibility, code the group does not understand, an unjustified selection, or results that cannot be defended.
- **Our case:** CV Macro F1 fell from V0 to V3 by 0.133 for MNB (0.9344 → 0.8016) and by 0.103 for LR (0.9687 → 0.8658) once the source shortcuts were removed (report §8.3). Under §23 that drop is a strength. Frame it as "we found and removed leakage", not as "the model got worse".
- **AI use (§16, contract item 10):** allowed, but every member must be able to explain the code, data, experiments, results, interpretation, model choice, report and references. Every reference in the list must be one the group can explain.
- **Defense (§18):** questions go to each member individually, covering 15 topics. The `DECISIONS_*.md` files are the preparation material.

## 4. Scope (Guideline §2.3)

The project must use classical ML; deep learning is out of scope. MNB, LR, LightGBM (gradient boosting) and stacking are all within scope. LSA/k-means fits "metode clustering / PCA".

## 5. Format reference (`PDF REPORT FILE REFERENCE FORMAT.pdf`, 15 pages)

This is a sample report from another project (XAU/USD). Conventions worth copying:

- **Layout:** Word, A4, Aptos font. Section headings are in CAPITALS without numbers (BUSINESS UNDERSTANDING, DATA UNDERSTANDING, DATA PREPARATION, …).
- **Figure captions** go **below** the figure as "Gambar 2.1 …" (chapter.number), in title case, with the period covered: "Gambar 2.1 Tren Harga Penutupan … Periode 2016–2026".
- **Each figure gets a three-paragraph pattern:**
  1. What the figure shows, with concrete numbers ("naik sebesar 288,61%").
  2. What it means.
  3. The consequence for modelling, with an honest caveat ("masih perlu diuji melalui seleksi fitur dan evaluasi model").

  This matches our Evidence → Decision style.
- **Numbers in prose** use Indonesian format: `754.660`, `26,52%`, `0,2478`, ranges with an en dash `2016–2026`, negatives with a minus sign `−9,5%`.
- **Results table** is a two-column "Metrik Evaluasi | Hasil" table.
- **Typos** in the sample ("ANALISI BIVARIAT", "EKSPREIMEN", "MISISNG", "DNA KE LAS") show what to avoid. Proofread the headings.
- **Foreign terms** (*machine learning*, *feature engineering*, *data leakage*) are **not italicised** in the sample. EYD V requires italics; see `BAHASA_INDONESIA_RULES.md`. For the report, foreign words are italic at every occurrence (WRITING_STYLE.md §9).

## 6. Weekly "OUTPUT PROJECT" checklists (each must be visible in the report)

| Meeting | Deck and page | Items |
|---|---|---|
| P3 | W03 p.~44, W04 p.104–105 | A Problem (rumusan, tujuan, target, fitur, jenis masalah); B Dataset (sumber, jumlah, fitur, target, kualitas); C Preprocessing (langkah, **alasan setiap langkah**, data setelahnya); D Baseline (algoritma, training, prediksi); E Evaluasi awal (metrik, hasil, interpretasi); F Temuan awal (apa berhasil, masalah, kelemahan baseline, perbaikan) |
| P3, text mining | W04 p.103, 116 | Minimum F1, precision, recall, accuracy **plus checking predictions on a few example texts**. Flow: Teks → Preprocessing → Representasi fitur → Baseline → Kandidat → Evaluasi → Perbandingan |
| P4 | W05 p.52 | A Validasi (skema split/CV, alasan, **bukti tidak ada leakage**); B Ensemble (model, *hyperparameter* dan tuning, vs baseline); C Imbalance (rasio kelas, metrik, teknik dan hasil); D **Tabel hasil rata-rata ± simpangan baku**, error analysis, kelemahan dan rencana |
| P5 | W06 p.36 | A Data (fitur, scaling, bukti tanpa leakage/Pipeline); B Clustering (algoritma dan alasan, grafik Elbow/Silhouette, K dan ukuran cluster); C Profil (tabel rata-rata fitur per cluster, nama/makna cluster, batasan interpretasi); D Reduksi dimensi (scree plot, plot 2D, dengan vs tanpa PCA) |
| P7 | W07 (no explicit output slide) | Apply the leakage checklist (Bagian E, p.35) and the reproducibility checklist (Bagian G, p.48) to every experiment. Use learning/validation curves. Do error analysis after each model. |

## 7. Slide points worth citing in the report

| Topic | Slide | Key line |
|---|---|---|
| CRISP-DM | W01 p.20–24; W02 p.4–8 | Source given as Chapman et al. (2000) |
| Business understanding fields | W02 p.~13, 21 | Masalah bisnis, tujuan analisis, stakeholder, pertanyaan kunci, metrik sukses. The format PDF uses these exact headings. |
| Precision/Recall/F1, threshold | W04 p.70–79, 83–84 | Lower the threshold for screening |
| Choosing a metric | W04 p.88–90 | "Ketika akurasi menipu"; ask questions before picking a metric |
| Stacking | W05 p.22–23 | "Model yang belajar dari model" |
| Data leakage | W05 p.27; W07 p.32 | Four types: target, train-test contamination, **group** (same entity in train and test), temporal. Our near-duplicates are group or contamination leakage. |
| Correct validation protocol | W05 p.35 | Split off test → CV on train → refit → **test once** → production. "Dosa besar": tuning on test, resampling before split, reporting only the best score. |
| Imbalance workflow | W05 p.41–48 | Stratified split → baseline → class_weight → **threshold chosen on validation** → resampling inside Pipeline. Report P, R, F1, PR-AUC, the confusion matrix, and the number of positives in test. |
| M05 imbalance table | `M05 Data Imbalance Table Reference.jpeg` (W05 p.46) | "PR-AUC hampir tidak membaik oleh resampling/class_weight. Yang berubah adalah **titik operasi**." This directly supports our choice of a threshold `b` over SMOTE. |
| Feature selection | W07 p.6–16 | Filter (χ²), wrapper, embedded; permutation importance p.14 |
| Apple-to-apple comparison | W07 p.20 | Same folds and seed for all models |
| Bias–variance, learning curve | W07 p.22–29 | — |
| Leakage detection | W07 p.33–35 | "Curigai skor yang nyaris sempurna" (p.35). This justifies the H4/H5 investigation. |
| Error analysis report format | W07 p.42 | A table of Kategori penyebab, Proporsi, Contoh & tindak lanjut. **Our Chapter 10 should include such a table.** |
| Reproducibility | W07 p.45 (seed, environment, data versioning), p.46 (experiment tracking), p.48 (checklist) | — |
| Model card | W07 p.47 | Purpose and context, data, method, results **with uncertainty intervals**, limitations and risks. A one-page model card in Chapter 16 would tick this off. |

## 8. Lecturer's reading lists (cite from these where possible)

- **Books (contract):** Géron (2022) *Hands-On ML* 3rd ed.; James et al. (2023) *ISLP*.
- **W01:** Samuel (1959), Mitchell (1997), Chapman et al. (2000) CRISP-DM.
- **W05:** Breiman (1996, 2001); Freund & Schapire (1997); Friedman (2001); Chen & Guestrin (2016); Wolpert (1992); Hastie et al. ESL ch.7; Bergstra & Bengio (2012); Chawla et al. (2002); He & Garcia (2009); Saito & Rehmsmeier (2015).
- **W06:** Lloyd (1982); Arthur & Vassilvitskii (2007); Ester et al. (1996); Rousseeuw (1987); Jolliffe (2002); van der Maaten & Hinton (2008).
- **W07:** scikit-learn User Guide (feature selection, permutation importance, common pitfalls); ISLR ch.2 and ch.5; Andrew Ng, *Machine Learning Yearning* ch.16–20; Mitchell et al. (2019) *Model Cards*.

All of these that we use are verified in `REFERENCES_VERIFIED.md`.
