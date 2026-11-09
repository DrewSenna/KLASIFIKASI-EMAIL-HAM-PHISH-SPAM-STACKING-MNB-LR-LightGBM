# Writing Style Guide — Group 1 Email Ham/Phish/Spam

This guide covers everything we write for the project: meeting notebooks, the report, and handouts. The reference example is `M03/Group_1_Email_Spam_Detection_M03.ipynb`.

The notebook and report are written in **Indonesian**. This guide is in English, and its examples are in Indonesian.

---

## 1. Core principles

1. **Linear.** Each sentence follows from the one before it. A paragraph reads top to bottom with no backtracking.
2. **Academic, but plain.** Use formal Indonesian and correct terms, in short sentences. Write so a classmate can follow on first reading.
3. **Clear without padding.** Say what was done, why, and what came out. Leave out anything the reader does not need: no restating the obvious, no stacked qualifications, no explaining the notebook to itself.
4. **Standalone.** Every notebook reads as a finished document. It never mentions earlier versions, revisions, or "previously it did X". Each setting is justified on its own merits, as if it had always been that way.
5. **Evidence → decision.** The professor grades our progress and decisions. Every choice has a stated reason, and every finding is linked to the decision it affects.

---

## 2. Sentences

| Rule | Do | Don't |
|---|---|---|
| One idea per sentence | Split a long sentence that carries two ideas into two sentences. When splitting at "sehingga", start the new sentence with "Akibatnya," or "Oleh karena itu,", never "Sehingga". | Chain three clauses with semicolons. |
| Subject first | "Sel berikut menghitung …", "Deduplikasi dilakukan sebelum …" | Open with a long condition or a passive chain. |
| State it once | Mark a hypothesis as a hypothesis, once. | Repeat "bukan kesimpulan kausal", "bukan diasumsikan", or "harus dibaca dari output" in every section. |
| Concrete over abstract | "Phish yang diprediksi sebagai Ham masuk ke kotak masuk pengguna." | "Kesalahan ini penting dalam konteks klasifikasi email." |
| Define a term on first use | "Pencilan adalah pesan dengan jumlah kata di atas Q3 + 1,5 × IQR." | Use a term and assume the reader knows the definition we chose. |

---

## 3. Section and cell patterns

**Markdown before a code cell.** Explain what the cell does and, if needed, why. Keep it to one or two sentences.

> Sel berikut menghitung jumlah dan persentase pesan pada setiap kelas, lalu memeriksa apakah ada label di luar 0, 1, dan 2.

**Steps.** Write a procedure as a numbered list, not a paragraph (see C.1).

**Settings.** Use a table with a reason column:

| Komponen | Parameter | Nilai | Alasan |
|---|---|---|---|

**Findings.** Use a table that links each finding to its evidence and the decision it affects (see B.6.4):

| Temuan | Bukti | Keputusan |
|---|---|---|

**Hypotheses.** Collect them in a numbered list under one lead-in line. Each hypothesis should be testable in a later meeting.

> Hipotesis penyebab kesalahan, yang akan diuji pada tahap berikutnya:
> 1. …

**Plans (F.4 and similar).** Write them as a plain text cell. Each item has a bold label, followed by the reason and the method in one or two sentences.

**Headings.** Keep the meeting template's lettered and numbered structure (A–F for M03). Headings name the content ("Audit Teks Tanpa Fitur TF-IDF"), not the activity ("Persiapan untuk audit nanti").

---

## 4. Numbers and generated text

1. **Never hardcode a result in markdown.** Any number produced by the run (metrics, counts, percentages) is written by code through an f-string and `display(Markdown(...))`.
2. **Static text must not depend on results.** Markdown cells only describe what is done and why.
3. **Generated text must not assume the outcome.** Compute which class has the lowest recall or the most errors, and write the sentence from that. Don't hard-write "Phish paling sulit". Use branches when a statement can flip (for example "Model menghasilkan prediksi benar pada ketiga kelas" vs "Belum semua kelas …").
4. **Number format.**
   - Static Indonesian text uses Indonesian format: `365.448 baris`, `1,5 × IQR`.
   - Computed values keep Python formatting (`365,448`, `0.9211`) so they match the tables.
   - Metrics use 4 decimals; percentages use 1–2 decimals.

---

## 5. Terms

Use Indonesian where a common term exists. Keep standard ML terms in English, in *italics* on first use when they are not code. Put code names, parameters and columns in `backticks`.

| Use | Avoid / note |
|---|---|
| pesan, email | "record", "rekaman", "dokumen" (except "dokumen" in TF-IDF theory) |
| data training, data testing | mixing "data latih", "train set", and "test set" in one notebook |
| kelas Ham / Phish / Spam | lowercase class names |
| pencilan | "outlier" in running text (fine in code) |
| duplikat; teks dengan label berbeda | "conflict group", "exact normalized-text deduplication" |
| tidak memiliki satu pun kata dalam vocabulary | "zero-token", "zero-feature" |
| *data leakage* (explain once: data testing ikut memengaruhi pelatihan) | "kebocoran" with no explanation |
| *stratified*, *pipeline*, *cross-validation*, *oversampling*, *unigram* | translating these into awkward Indonesian |
| Phish → Spam (aktual → prediksi) | "Phish salah ke Spam", "Phish/Spam confusion" |
| model kandidat | "metode yang diusulkan", "proposed hybrid method" |

---

## 6. Phrases to avoid → replacements

These come from the M03 rewrite.

| Avoid | Write instead |
|---|---|
| Output yang perlu diperiksa adalah … | Sel berikut menampilkan … |
| Jumlah data dihitung langsung dari `raw_df`, bukan ditulis sebagai hasil tetap. | *(omit; numbers always come from code, see §4)* |
| Distribusi dan proporsinya dibaca dari output, bukan diasumsikan tetap. | *(omit)* |
| Sel berikut memenuhi kebutuhan inspeksi beberapa hasil prediksi. | Sel berikut menampilkan tiga kelompok contoh dari data testing: … |
| Preprocessing baseline dibuat konservatif. `clean_text` menyimpan teks model setelah whitespace dinormalisasi, sedangkan `dedup_key` … | Preprocessing dibuat minimal agar informasi pada teks tidak hilang. Langkah yang dilakukan: 1. … |
| Interpretasi metodologis: exact normalized-text deduplication dilakukan sebelum train-test split untuk mengurangi kebocoran … | Deduplikasi dilakukan sebelum data dibagi menjadi training dan testing. Tanpa langkah ini, teks yang sama dapat muncul di kedua bagian … |
| … adalah hipotesis yang perlu diuji pada eksperimen berikutnya, bukan kesimpulan kausal dari baseline ini. | Hipotesis penyebab kesalahan, yang akan diuji pada tahap berikutnya: 1. … |
| Uji metode yang diusulkan dalam penelitian ini sebagai proposed hybrid method | **Menambahkan model kandidat.** … |
| Ini menguji—bukan mengasumsikan—apakah … | Eksperimen membandingkan … |
| Pada versi sebelumnya …, Batas 50.000 kata sebelumnya … | *(never; state the current setting and its reason)* |

---

## 7. Charts

- Class colours are fixed in every chart: **Ham `#2a78d6`** (blue), **Phish `#eb6834`** (orange), **Spam `#1baf7a`** (aqua). Define them once as `CLASS_COLORS`.
- Titles and axis labels are in Indonesian and name the unit ("Jumlah pesan", "Persentase pesan (%)").
- Use one y-axis per chart; put two measures in two subplots.
- Use thin marks, light grids, and no top/right spines. Label values directly only when there are few (for example the bars in B.6.2).
- If a chart hides something (outliers, a clipped range), say so in the title or the preceding sentence.

---

## 8. Checklist before a notebook is handed over

- [ ] No mention of earlier versions, revisions, or past meetings' mistakes.
- [ ] Every markdown cell before code says what the cell does in 1–2 sentences.
- [ ] Every setting has a reason; every finding is linked to a decision.
- [ ] No hardcoded results in markdown; generated text does not assume the outcome.
- [ ] Terms follow §5; no leftover phrases from §6.
- [ ] Hypotheses appear once, numbered, marked to be tested later.
- [ ] *Restart & run all* completes without errors.

---

## 9. Final report (decided 6 Nov 2026)

The report follows §1, §2, §5–§7 and §4.4 above (§3 and §4.1–4.3 are notebook-only) and `Report/BAHASA_INDONESIA_RULES.md` (EYD V), whose §A is binding. Where this section differs from the notebook rules above, it overrides them for the report only:

| Topic | Report rule |
|---|---|
| Foreign words | Italic at every occurrence, not only first use: prose, headings, table cells and captions. Data *training*, data *testing*, *email*, *fold*, *threshold*, *stacking*. Not in code/backticks, file names or text inside figure images. |
| Not italic | Class labels (Ham, Phish, Spam), model and metric names used as labels (Logistic Regression, Multinomial Naive Bayes, Macro F1, Ham Recall), software (scikit-learn, LightGBM), abbreviations (TF-IDF, SMOTE), "dkk." |
| Numbers | Indonesian format everywhere, tables included (0,8913; 365.448; 99,5%). Metrics to 4 decimals. Code blocks may keep dot decimals. |
| Citations | IEEE numbered [n] in order of first citation; heading "Daftar Pustaka"; entries in `Report/REFERENCES_IEEE.md` |
| Captions | "Tabel x.y Judul" above the table, "Gambar x.y Judul" below the figure, no final period |
| Commas | No comma before "sehingga", "karena", "agar", "jika" after a main clause; comma before "tetapi", "sedangkan" |
| Bound forms | antarmodel, nonnegatif, antar-*fold*, non-ASCII |
