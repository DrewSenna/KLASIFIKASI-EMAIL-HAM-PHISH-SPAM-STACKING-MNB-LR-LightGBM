# Bahasa Indonesia Rules for the Final Report (EYD V + KBBI VI)

Compiled 6 Nov 2026 from the official sources:
- **EYD V:** `https://ejaan.kemendikdasmen.go.id/eyd/…`. Saved copies of the pages used are in the 6 Nov session scratchpad.
- **KBBI VI Daring:** `https://kbbi.kemendikdasmen.go.id/entri/<kata>`.

The old `*.kemdikbud.go.id` hosts now redirect.

**How this file is used**
- For the report, the decisions in §A are **binding**, together with `../WRITING_STYLE.md` §9. They override WRITING_STYLE §1–7 where the two differ.
- §1–§9 apply unless §A says otherwise.
- §A2 records the further decisions taken after the audit.

**Notation in this file**
- *Italics* mark only a word that is italic in the report.
- A word that is merely being talked about is put in "quotes". For example, "dkk." and "dan" are roman in the report.

**Verification status (6 Nov 2026)**
- Every rule number in §1–§6 was checked against the saved EYD V pages.
- 63 words were checked live on KBBI VI Daring.
- Items marked **(mirror)** were checked on kbbi.web.id, which carries KBBI III data, after the anonymous daily limit was reached.
- The Badan Bahasa *Glosarium/Padanan Istilah* is unreachable (DNS failure), so §8 makes no "official equivalent" claims.

---

## A. Binding decisions for the report (decided by the group, 6 Nov 2026)

| # | Topic | Rule for the report | Basis |
|---|---|---|---|
| A1 | *training*, *testing* | data *training* / data *testing*, italic everywhere: prose, headings, table cells, captions. A bare word is italic too (bagian *training*, skor *training*). Never "data latih/uji", "train set" or "test set". | Group decision; foreign words are italic (Huruf Miring 3) |
| A2 | *email* | *email*, italic everywhere: prose, headings, title, tables, captions. Never "surel" or "e-mail". The class labels Ham/Phish/Spam stay roman. | Group decision. In KBBI VI (live), "email" means only "enamel", so the English word is foreign and italic. KBBI's form for electronic mail is "pos-el" (kp pos elektronik); "surel" is not a KBBI VI entry. |
| A3 | *fold* | *fold* italic. In prose, spell the count: lima *fold*, *fold* ke-3, antar-*fold*. Digits are allowed in tables, in a series, and in "k = 5". Never "lipatan". | Group decision; Angka 2; Kata Turunan 1d |
| A4 | Commas with subordinate clauses | No comma before "sehingga", "karena", "agar" or "jika" when the main clause comes first. Comma when the subordinate clause comes first. | Koma 3–4. Koma 4's examples are "kalau", "karena", "agar"; "sehingga" is included by inference, and the group adopted it. |
| A5 | Bound forms | antarmodel, antarkelas, nonnegatif, multikelas, praproses; hyphen before an italic word or a capital: antar-*fold*, non-ASCII | Kata Turunan 1b–1d |
| A6 | Model and metric names | Model names used as labels keep their capitalised English form, roman: Logistic Regression, Multinomial Naive Bayes, LightGBM. Metric labels likewise: Macro F1, Ham Recall, Phish Precision, PR-AUC. Generic senses are lowercase: regresi logistik, *naive Bayes*, akurasi, presisi, *recall*. State once in the section on candidate models: "Nama model ditulis sesuai nama aslinya." | EYD has no rule on this. Huruf Kapital 4 ("teori Darwin") implies "teorema Bayes" and *naive Bayes* by analogy. |
| A7 | Ranges | En dash with no spaces: 0,10–0,13; halaman 1137–1143. Never a spaced hyphen or a spaced dash. | Pisah 3 prescribes the em dash ("2019—2022"), but EYD's own Titik Dua 5 example uses an en dash ("98–105"). Applied consistently, the en dash is defensible. |
| A8 | Citations | IEEE numbered [n], ordered by first citation; heading **Daftar Pustaka**. Running text: "Chawla dkk. [12]", "He dan Garcia [7]"; roman "dkk.", never "et al.". Several sources: [3], [7] or [3]–[5]. The entries are in `REFERENCES_IEEE.md`. The dataset is cited only as the Hugging Face entry `locuoco/the-biggest-spam-ham-phish-email-dataset-300000`; every access date reads "diakses 9 Okt. 2026". | Group decision; JSM (Mikroskil) uses IEEE + Daftar Pustaka. "dkk." is an Indonesian abbreviation (Singkatan 4a). |
| A9 | Pronouns | Impersonal or passive; no "kami". | House style |
| A10 | Numbers | Indonesian format everywhere, tables included: 0,8913; 365.448; 99,5%. Metrics take 4 decimals, percentages 1–2 (WRITING_STYLE §4.4). Code blocks keep dot decimals; say so once in the reproducibility section. | Koma 12; Titik 8 |
| A11 | Captions | "Tabel 8.3 Hasil *Cross-Validation* pada Teks V3" above the table; "Gambar 3.1 Panjang Pesan per Kelas" below the figure. Numbered per chapter; no period after the number or at the end; title case. Referred to as "Tabel 8.3" and "Gambar 3.1". | EYD Titik 6, 10 cover only the punctuation. Placement follows JSM and the course format sample. |

### A2. Further decisions (group, 6 Nov 2026)

| # | Question | Decision |
|---|---|---|
| O1 | "dataset" is not a KBBI VI entry | **Roman**: dataset. It is treated as an established term, so this is a group exception to Huruf Miring 3. |
| O2 | *phishing*, *baseline*, *spam* as a common noun | **Roman**: phishing, baseline, spam. Other foreign words stay italic: *pre-processing*, *pipeline*, *stacking*, *threshold*. |
| O3 | English chapter headings that follow the course guideline's chapter names | **Kept, in italics**: `## 8. *Validation and Evaluation*`. |
| O4 | "library" vs "pustaka" | Software library (Python library): *library*, italic. Bibliography: Indonesian, "Daftar Pustaka". |
| O5 | Ratio format | No spaces: 3,93:1; Ham:Phish. |
| O6 | "adalah" vs "ialah" for definitions | **adalah** |
| O7 | Citing the course guideline | **Not cited.** Refer to it in plain text as "panduan *final project*", with no [n] and no Daftar Pustaka entry. |

---

## 1. Status of EYD V

- **Decree:** Keputusan Kepala Badan Pengembangan dan Pembinaan Bahasa **No. 0424/I/BS.00.01/2022**, signed on 16 Aug 2022 by E. Aminudin Aziz.
- **What it replaces:** it revokes Kepka No. 0321/I/BS.00.00/2021 tentang PUEBI. PUEBI was originally Permendikbud No. 50/2015.

**Changes from PUEBI that matter for us**
1. Bibliography punctuation is **no longer prescribed** by the spelling rules; follow IEEE. Titles of books and journals are still italic, including in the Daftar Pustaka (Huruf Miring 1).
2. New rule: a bound form before an italic word takes a hyphen, and only the foreign part is italic (Kata Turunan 1d).
   - EYD's examples: anti-*mainstream*, pasca-*reshuffle*.
   - Our case: pra-*processing*.
3. A name inside the name of a theory, law or formula is capitalised (Huruf Kapital 4): teori Darwin, hukum Archimedes, and by analogy teorema Bayes.
4. New colon uses for ratios (1:10.000) and times (01:35:20) (Titik Dua 6–7).
5. Trademarks are explicitly **not** italic, like other proper names (Huruf Miring, Catatan a).

## 2. Foreign terms

| Rule | Do | Don't | Source |
|---|---|---|---|
| Italicise foreign words, every occurrence (exceptions by decision: dataset, phishing, baseline, spam; O1–O2) | *cross-validation*, *oversampling*, *recall* | cross-validation (roman) | Huruf Miring 3 |
| Proper names, institutions, trademarks are **not** italic | scikit-learn, Python, Google Colab, Kaggle, Hugging Face, Enron, LightGBM | *scikit-learn* | Huruf Miring, Catatan a |
| Absorbed words (KBBI VI lemmas) are roman | fitur, algoritma, akurasi, presisi, validasi, parameter, model, data, token | *fitur* | KBBI VI (live) |
| Introduce an unfamiliar term with its equivalent in parentheses | validasi silang (*cross-validation*) | — | Tanda Kurung 1 |
| A meaning or gloss goes in single quotes | *stopword* 'kata umum' | — | Huruf Miring 3 example ("*Weltanschauung* bermakna 'pandangan dunia'") |
| Indonesian affix or clitic + foreign word: hyphen, foreign part italic | di-*tuning*, di-*resample*, *recall*-nya | ditraining, di *training*, recall-nya | Tanda Hubung 7. Its examples are prefixes; applying it to "-nya" is the standard reading. |
| Better: use an Indonesian verb | dilatih, diuji, disetel, diperbarui | mentraining, ter-update | style |
| Affix + abbreviation or capital: hyphen | di-SK-kan, se-Indonesia | — | Tanda Hubung 5 |
| Abbreviations upright, no periods; spell out the foreign expansion in italics at first use | TF-IDF (*term frequency–inverse document frequency*), SMOTE (*Synthetic Minority Over-sampling Technique*) | *SMOTE* | Singkatan 3; italic expansion per Huruf Miring 3 |
| Never italic | `code`, parameters, column names, file names (in backticks); text inside figure images; the original English titles in the Daftar Pustaka except the journal or book title | — | house style |

## 3. Capitalisation

| Rule | Do | Don't | Source |
|---|---|---|---|
| Titles and headings: capitalise every word except function words (di, ke, dari, dan, dengan, yang, untuk, pada, per, atau, dalam, terhadap) unless first | Klasifikasi *Email* Ham, Phish, dan Spam dengan *Classical Machine Learning* | … **Dan** Spam **Dengan** … | Huruf Kapital 21 (titles); extended to headings, as in EYD's heading examples (Huruf Tebal 2) |
| Multi-level section numbers: no period after the last number. Single-level list numbers keep one. | 3.1 Sumber Dataset; 3.1.2 …; "1. Latar Belakang" | 3.1. Sumber Dataset. | Titik 3, 4, 10 |
| Lowercase after a colon that continues the sentence; names keep their capitals | … tiga tahap: pembersihan, tokenisasi, dan pembobotan. | … tiga tahap: Pembersihan, … | Titik Dua 1 (examples) |
| Foreign common nouns are lowercase mid-sentence | *classical machine learning*, *supervised learning*, *deep learning* | … dengan *Deep Learning* … | Huruf Kapital (common nouns) |
| "bahasa" lowercase, language name capitalised | bahasa Inggris | Bahasa Inggris (mid-sentence) | Huruf Kapital 11 |
| Cross-references: "Bab", "Tabel", "Gambar" capitalised; "halaman" lowercase | lihat Tabel 8.3; pada Gambar 3.2; Bab 5 | lihat tabel 8.3; bab 5 | EYD examples only (Tanda Kurung 2 "lihat Tabel 10"; Angka 7 "Bab II, Pasal 3, halaman 13") |

## 4. Numbers, units and symbols

| Rule | Do | Don't | Source |
|---|---|---|---|
| Decimal comma | 0,8913 | 0.8913 (prose or tables) | Koma 12 |
| Thousands period, for quantities only | 365.448 pesan | 365,448 / 365448 | Titik 8 |
| No separator in years, pages or IDs | tahun 2026; *seed* 42 | 2.026 | Titik 9 |
| A number that can be said in **one** word is spelled out in prose (EYD V says "satu kata"; PUEBI said "satu atau dua kata") | tiga model, lima *fold* | 3 model (plain prose) | Angka 2 |
| Digits for a series, measurements and values (money, percentages) | 2, 5, dan 10 pohon; 0,5 detik; 99,5% | — | Angka 2–3 |
| Percent with no space | 99,5% | 99,5 % | Angka 3 examples ("5%"); EYD states no explicit spacing rule |
| Number, space, unit; unit symbols take no period | 12,5 m; 3,5 jam; 631 MB | 631MB; 12 km. | Koma 12; Singkatan 5 |
| No digit at the start of a sentence | Sebanyak 94 *email* sah … / Model dengan 150 pohon … | 94 *email* sah … / 150 pohon … | Angka 4 |
| Ordinals | ke-5, kelima | ke 5, ke-lima | Angka 9 |
| Decades | tahun 2000-an | tahun 2000an | Tanda Hubung 5; Angka 10 |
| Ranges | 0,10–0,13 (see A7) | 0,10 - 0,13; 10.000 – 20.000 | Pisah 3; A7 |
| Ratios and times | 3,93:1; 01:35:20 | 3,93 : 1 | Titik Dua 6–7 |
| "/" for "dan", "atau", "setiap" | dan/atau; 150 km/jam; or write "per": 32 sampel per *batch* | — | Garis Miring 2; Partikel 4 |
| Operators and symbols spaced; minus sign U+2212 | α = 2,0; 1,5 × IQR; *b* ≥ 0; −0,0884 | α=2,0; -0,0884 | typographic convention (EYD has only "25 x 1.000" in Tanda Hubung 4) |
| "±" only for mean ± deviation; "sekitar" for approximations | 0,89 ± 0,01; sekitar 3,5 jam | ± 3,5 jam | convention |

## 5. Punctuation

| Rule | Do | Don't | Source |
|---|---|---|---|
| Serial comma before the last item | Ham, Phish, dan Spam | Ham, Phish dan Spam | Koma 1 |
| Comma **before** "tetapi", "melainkan", "sedangkan" | Akurasinya tinggi, tetapi *recall* Phish rendah. | … tinggi tetapi … | Koma 2 |
| Subordinate clause first: comma | Karena kelas tidak seimbang, digunakan Macro F1. | — | Koma 3 |
| Main clause first: **no comma** before "karena", "agar", "jika", "sehingga" (the last by inference; see A4) | Macro F1 digunakan karena kelas tidak seimbang. | …, karena … / …, sehingga … | Koma 4 |
| Sentence-initial connectors take a comma | Namun, … / Oleh karena itu, … / Dengan demikian, … | Namun model … | Koma 5 |
| Commas around an appositive or extra information | LightGBM, model berbasis pohon, … | — | Koma 13 |
| Don't start a sentence with an intra-sentence conjunction. When splitting at "sehingga", start the new sentence with "Akibatnya," or "Oleh karena itu,". | Akan tetapi, … / Akibatnya, … | Tetapi, … / Sehingga … / Dan … | usage guides |
| No "dimana"; no "di mana" or "yang mana" as a relative pronoun | … model **yang** *recall*-nya … / … lingkungan **tempat** kode dijalankan | … model di mana … | usage guides; Kata Depan |
| Colon only after a complete statement | Ada tiga tahap: pembersihan, tokenisasi, dan pembobotan. | Tahapannya meliputi: … | Titik Dua 1–2 |
| Semicolons between verb-phrase list items; "dan" before the last | (1) …; (2) …; dan (3) …. | — | Titik Koma 2–3 |
| No period after an enumerator in parentheses | 1) … | 1). … | Titik 5 |
| IEEE citations: brackets before the period, numbers separated by commas | … oleh Chawla dkk. [12]. / [3], [7] / [3]–[5] | (Chawla dkk., 2002) | IEEE (A8) |
| Em dash for a parenthetical insertion, used sparingly | Model itu—seperti diduga—gagal. | hyphen for asides | Pisah 1–2 |
| Double quotes for article and chapter titles, and for a little-known term or a word with a special meaning | bab "Evaluasi"; "pencilan" | — | Tanda Petik 2–3 |
| Bold only for headings, or emphasis inside text that is already italic; normal emphasis is italic | — | **bold** for emphasis in prose | Huruf Miring 2; Huruf Tebal 1–2 |

## 6. di / di-, pun, per, compounds, bound forms

| Rule | Do | Don't |
|---|---|---|
| Preposition separate | di dalam, di antara, di atas, ke luar (direction: ke luar kota) | didalam, diantara |
| Prefix attached; "keluar" is a verb | dilatih, diuji, dievaluasi; keluar dari | di latih |
| "pun" meaning "even/also": separate | apa pun | apapun |
| "pun" inside conjunctions: attached | meskipun, walaupun, maupun, adapun | meski pun |
| "per" meaning "each": separate | 32 sampel per *batch* | perbatch |
| Compounds written apart | kerja sama, tanggung jawab, bekerja sama, model linear | kerjasama |
| Compounds written as one word | daripada, kosakata | dari pada |
| Compound with a prefix **and** a suffix: attached | pertanggungjawaban, menggarisbawahi | pertanggung jawaban |
| Reduplication with a hyphen | model-model, kata-kata | model model |
| Bound forms attached | antarkelas, antarmodel, multikelas, nonlinear, praproses, hiperparameter, subbagian | antar kelas, multi-kelas |
| Bound form before a capital or abbreviation: hyphen | non-ASCII, non-Ham, pasca-COVID | nonASCII |
| Bound form before an italic word: hyphen | antar-*fold*, pra-*processing*, non-*stopword* | antar*fold* |

Sources: Kata Depan; Partikel 2–4; Kata Turunan 1b–1d, 2a, 3a–3e (Gabungan Kata).

## 7. Baku vs tidak baku (KBBI VI live, 6 Nov 2026)

| Baku ✔ | Tidak baku ✘ |
|---|---|
| analisis, menganalisis | analisa, menganalisa |
| sistem | sistim |
| metode | metoda |
| kualitas | kwalitas |
| risiko | resiko |
| praktik | praktek |
| efektif, efektivitas | efektip, efektifitas, keefektivitasan |
| aktivitas | aktifitas |
| objek | obyek |
| teknik | tehnik |
| frekuensi | frekwensi |
| hipotesis | hipotesa |
| diagnosis | diagnosa, diagnose |
| standar | standard, standart, setandar |
| eksperimen | eksperimentasi |
| algoritma | algoritme (tidak baku in KBBI VI; it was the lemma in older editions) |
| kategori, mengategorikan | katagori, mengkategorikan |
| memengaruhi | mempengaruhi |
| **memperhatikan** | memerhatikan (a trap: "memper-" + "hati") |
| mengonversi | mengkonversi |
| menerjemahkan | menterjemahkan |
| mengklasifikasikan, memproses, memprediksi (no melting before a consonant cluster) | mengelasifikasikan, memroses |
| mengubah | merubah |
| teoretis | teoritis |
| ekstrem | ekstrim |
| persentase | prosentase, presentase |
| konkret | konkrit, kongkret, kongkrit |
| sekadar | sekedar |
| kompleks | komplek |
| matriks | matrik |

**Also baku (KBBI VI lemmas, write roman):**
- karakteristik, akurasi, parameter, validasi, probabilitas, implementasi, optimal, efisien, variabel, sampel, fitur, presisi, token, model, data, kalibrasi;
- tokenisasi ("proses pemisahan teks ke dalam unit-unit kecil yang disebut token");
- deduplikasi ("proses menghilangkan duplikasi atau pengulangan");
- daring.

The following are lemmas **(mirror)**: akurat, signifikan, kuantitatif, kualitatif, konsisten, relevan, komponen, atribut, terdistribusi, validitas, empiris, numerik, iterasi, korelasi, rerata, metrik, logaritma.

**linear vs linier:** both are separate KBBI VI lemmas, and neither is marked tidak baku.
- "linear" (Mat., "berbentuk garis") is the one used for the ML sense: model linear, regresi linear, nonlinear.
- "linier" means "terletak pada suatu garis lurus".

**Not KBBI VI entries:** hiperparameter, praproses, multikelas.
- They are regular formations (hiper- + parameter, pra- + proses, multi- + kelas), so write them roman. Don't claim they are KBBI entries.

**Mirror only:**
- "hirarki", "aritmatika", "pararel" and "sistimatis" are absent. Use hierarki, aritmetika, paralel, sistematis.
- "subyek" redirects to "subjek".
- "kategoris" means "tepat; langsung", not "categorical": write data kategorikal or *categorical*.

## 8. ML terms (one term per concept; no official glossary verified)

| English | Use in the report |
|---|---|
| machine learning | *machine learning*, lowercase mid-sentence. The course name "Pembelajaran Mesin" stays as it is. |
| dataset | dataset, roman (O1) |
| training / test data | data *training* / data *testing* (A1) |
| feature | fitur (KBBI VI lemma; its ML sense is an extension, so gloss it once: fitur (*feature*)) |
| outlier | pencilan (KBBI VI: "data yang berperilaku menyimpang dari kelompok mayoritas datanya") |
| threshold | *threshold* (WRITING_STYLE §9) |
| accuracy / precision | akurasi / presisi; as metric labels: Accuracy → "Akurasi" |
| recall, F1 | *recall*; labels Macro F1, Ham Recall (A6) |
| cross-validation | *cross-validation*; optional gloss once: validasi silang (*cross-validation*) |
| preprocessing | *pre-processing*, matching the guideline's chapter name. The bound form "praproses" is acceptable but not mixed. |
| hyperparameter | hiperparameter |
| stopword, overfitting, ensemble, stacking, bigram, pipeline | italic: *stopword*, *overfitting*, *ensemble*, *stacking*, *bigram*, *pipeline*. Roman by decision (O2): baseline, phishing, spam. |
| confusion matrix | *confusion matrix* |
| library | *library* for software; "pustaka" only in "Daftar Pustaka" (O4) |
| out-of-fold | *out-of-fold* (OOF) |

## 9. Report conventions

**Abstract**
- "Abstrak" plus "Kata kunci" (3–5 words), at most about 200 words (JSM guideline).

**Captions (A11)**
- In markdown, a table caption is a separate roman line directly above the table, for example `Tabel 3.1 Distribusi Kelas`.
- For figures, the image alt text is the caption (pandoc turns it into the Word caption). Write `![Gambar 3.1 Panjang Pesan per Kelas](figures/…)` with no second caption paragraph.
- A caption is a short title. Explanations go in the body, or on a "Keterangan:" line under the figure.
- Panels: Gambar 8.2(a).

**Cross-references**
- One form only: Bab 5, Subbab 3.6, Tabel 8.3, Gambar 3.1, Gambar 8.2(a).
- Re-check every number after chapters are renumbered.

**Abbreviations**
- Expand each at first use in the body: TF-IDF, SMOTE, PR-AUC, SVD, PCA, KNN, NLTK, IQR, ARI, RFE, OOF.

**Daftar Pustaka**
- Every citation in the text is listed, and every listed item is cited (IEEE, A8).

**Word choice**
- Avoid:
  - "adalah merupakan" (pleonasm: KBBI glosses "merupakan" as "adalah");
  - "adalah" before an adjective;
  - "daripada" meaning "of". Its only KBBI sense is comparison.
- "adalah" vs "ialah" (advisory):
  - KBBI glosses "adalah" as "identik dengan; sama maknanya dengan; termasuk dalam kelompok", and "ialah" as introducing "perincian atau penjelasan".
  - Both can introduce a definition; the report uses "adalah" (O6).

## 10. Universitas Mikroskil

No public *Pedoman Penulisan* was found.

**Layout**
- Follow the course format sample PDF: Aptos, A4, Word.
- Unlike the sample, which uses unnumbered capital headings, the report uses numbered headings in title case (§3).

**Jurnal SIFO Mikroskil (JSM)**
- JSM is cited only as support for IEEE + Daftar Pustaka, caption placement, and Abstrak + Kata kunci.
- Its own pages contain "dimana" and "teoritis", so it is not a spelling authority.

Ask Pak Darwin whether a faculty template exists.

---

## Pre-submission checklist

Tags: **[D]** group decision, **[E]** EYD V / KBBI VI rule, **[H]** house convention.

1. [D] *email*, data *training*, data *testing*, *fold* and other foreign words are italic everywhere, headings and tables included; dataset, phishing, baseline and spam stay roman (A1–A3, O1–O2, §2).
2. [E] Software, library and dataset names are **not** italic: scikit-learn, LightGBM, Hugging Face, Enron (§2).
3. [E] Abbreviations (TF-IDF, SMOTE, PR-AUC) are upright; their expanded forms are italic at first use (§2, §9).
4. [H] Each technical term a classmate may not know is explained at first use. Common ML terms (WRITING_STYLE §5) need no translation.
5. [E] An affix or "-nya" on a foreign word uses a hyphen (di-*tuning*, *recall*-nya); prefer an Indonesian verb (§2).
6. [E] Bound forms are attached (antarmodel, nonnegatif), or hyphenated before a capital or an italic word (non-ASCII, antar-*fold*) (A5, §6).
7. [E] "di" is separate as a preposition and attached as a prefix; ke luar vs keluar (§6).
8. [E] No "dimana"; no "di mana" or "yang mana" as a relative pronoun (§5).
9. [E] apa pun / meskipun; per *fold* written separately (§6).
10. [E] kerja sama, tanggung jawab, pertanggungjawaban; model-model (§6).
11. [E] None of the §7 tidak baku forms appear (run §B).
12. [E] Decimal comma and thousands period in prose **and tables**; no separator in years or seeds; metrics to 4 decimals (A10, §4).
13. [E] 99,5% with no space; a space before units; ranges with an en dash and no spaces (A7, §4).
14. [E] Numbers that are one word are spelled out in prose (tiga model), except in a series, a measurement or a value (§4).
15. [E] No sentence starts with a digit; ordinals ke-5 / kelima (§4).
16. [E] Comma before "tetapi", "sedangkan", "melainkan"; none before "karena", "sehingga", "agar", "jika" after a main clause (A4, §5).
17. [E] "Namun," and "Oleh karena itu," at sentence start; no sentence starts with "Tetapi", "Sehingga" or "Dan" (§5).
18. [E] Colon only after a complete statement; ratios without spaces (§4, §5).
19. [E] Headings in title case with function words lowercase; no period after the last number of a section number (§3).
20. [D] "Tabel x.y" above tables, "Gambar x.y" below figures, no final period. Referred to as Tabel/Gambar/Bab with a capital letter (A11, §3, §9).
21. [D] IEEE [n] in order of first citation; "dkk." not "et al."; heading "Daftar Pustaka"; the citations and the list match; the dataset is cited; the course guideline is not cited (A8, O7).
22. [D] Abstrak ≤ about 200 words + Kata kunci; impersonal style; no "adalah merupakan" (A9, §9).
23. [D] One term per concept: *email* (never surel or e-mail); data *training* / data *testing* (never data latih/uji); *threshold*; *pre-processing*; pencilan (§8).
24. [H] Model and metric labels are capitalised and roman (Logistic Regression, Macro F1); generic terms are lowercase (A6).

## B. Quick check script

The script is `check_bahasa.sh` in this folder. Run it as `bash check_bahasa.sh Laporan_UTS_Group_1.md` (GNU grep with `-P`).

How it works:
- It blanks out inline code, image lines and link targets.
- It stops at the "Daftar Pustaka" or "Referensi" heading, so English reference titles don't trigger it.
- Line numbers match the file.

What it flags:
- tidak baku forms;
- a comma before a subordinator;
- sentence-initial conjunctions;
- bound forms written apart;
- an affix joined to a foreign word without a hyphen;
- decided foreign words left roman;
- number-format slips;
- a digit at the start of a sentence;
- "1).";
- APA or "et al." leftovers;
- old-style captions;
- spaced ratios.

Hits are candidates to review, not automatic replacements.
