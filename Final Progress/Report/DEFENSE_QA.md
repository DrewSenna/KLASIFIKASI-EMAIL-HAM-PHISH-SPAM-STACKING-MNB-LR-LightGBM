# Defense Q&A — Laporan UTS Group 1

Preparation guide for defending `Laporan_UTS_Group_1.md`. Explanations for the group are in English. Model answers are in Indonesian and follow `BAHASA_INDONESIA_RULES.md`: italic *email*, data *training*/*testing*, *fold*, *threshold*, *stacking*; roman phishing, spam and baseline; decimal comma; "dkk."; IEEE numbers [n] that match the current Daftar Pustaka.

**About the numbers.**
- Every result number in a model answer was checked against the current report text. Each one carries its Tabel, Gambar or Subbab reference, so you can point to it.
- Slide material is cited as "W05, halaman 44", the same way the report cites it, using the PDF page number. Every slide page and phrase quoted here was checked with `pdftotext`.
- The report's style rule A9 ("no *kami*") is for the written report. In spoken answers, "kami" is natural and fine.

---

## 1. How to use this guide

### The three-layer answer pattern

Most "why" questions are answered in four steps. Say them in this order and keep each one to a sentence or two.

1. **Course material.** What the course teaches about this choice (slide page or the guideline).
2. **Reference.** A source from the Daftar Pustaka that supports it.
3. **Own evidence.** The number from our own experiment, with its Tabel or Subbab.
4. **Decision.** What we chose, in one sentence.

Steps 1–3 are the three layers of support; step 4 is the conclusion. If you only have time for one layer, use step 3: our own evidence is the strongest, because nobody else can contest the numbers from our notebooks.

### The honesty pattern

When a question touches a weak point, don't defend it as if it were perfect. Use these three steps:

1. **What it is.** State the fact plainly ("Nilai ini tidak di-*tuning* pada V3").
2. **Where it came from.** Give its origin ("Ditetapkan dari uji pendahuluan sebelum *notebook* dijalankan").
3. **Why it's still reasonable.** Give the evidence that limits the damage ("Jumlah pohon, yang paling berpengaruh, sudah di-*tuning* ulang, dan kami mencatatnya sebagai keterbatasan di Bab 14").

An examiner usually stops pressing once a limitation is stated openly and its effect is bounded. Being caught hiding a weakness hurts far more than the weakness itself.

---

## 2. The general question: "Dasar pengerjaannya dari mana?"

This usually comes first, or as a follow-up to any decision. Answer it in four parts: course framework, literature, the data itself, documented experiments.

> **Model answer.**
> Dasar pengerjaan kami ada empat.
>
> **Pertama, kerangka mata kuliah.** Alurnya mengikuti *data science pipeline* yang diajarkan sejak Pertemuan 1–2, dimulai dari *Business Understanding*. Setiap pertemuan kami tutup dengan butir *output project* pada *slide* minggu tersebut. Struktur laporan dan dua belas butir minimum UTS mengikuti panduan *final project* (bagian "Cakupan *Milestone* 1" di awal laporan).
>
> **Kedua, literatur.** Setiap metode utama punya rujukan:
> - TF-IDF dari Manning dkk. [2];
> - Multinomial Naive Bayes dengan frekuensi kata dari McCallum dan Nigam [8];
> - SMOTE dari Chawla dkk. [13], dan masalah data tidak seimbang dari He dan Garcia [4];
> - *boosting* dan LightGBM dari Friedman [10] dan Ke dkk. [11];
> - *stacking* dari Wolpert [12];
> - *bootstrap* dari Efron dan Tibshirani [9];
> - risiko *dataset shift* pada penyaring *email* dari Jáñez-Martino dkk. [1].
>
> **Ketiga, data itu sendiri.** Banyak keputusan berasal dari temuan EDA, bukan dari kebiasaan umum. Contohnya, sumber Phish tidak memiliki *stopword*: median porsi *stopword* NLTK adalah 0,0% pada Phish, dibandingkan 30,5% pada Ham dan 26,3% pada Spam (Subbab 3.6, Gambar 3.2). Temuan ini yang mendorong penyeragaman teks V3.
>
> **Keempat, eksperimen yang terdokumentasi.** Setiap keputusan diuji dengan satu faktor per eksperimen (Subbab 6.2) dan aturan pemilihan yang ditulis di *notebook* (Subbab 6.5). Semua hasil ada di *notebook* M03–M07 dan Final Evaluation, dan setiap keputusan dicatat beserta alasannya di berkas DECISIONS per pertemuan.

**If they push on "which slide?":** the strongest single slide is W05, halaman 41, "Peta Strategi Penanganan": *"Mulai dari yang paling sederhana. Resampling bukan langkah pertama."* It backs question A directly.

---

## 3. Questions about specific decisions

### A. Kenapa pakai *threshold* Ham, bukan SMOTE atau *class weight*?

**For the group.** This is the most likely question, because the original plan said "MNB + TF-IDF + SMOTE". The answer has three layers. The course's own strategy map puts the threshold before resampling. SMOTE's synthetic points don't fit text. And our experiment shows no technique helped. What makes the answer stick is the business link: the threshold is one number that directly controls the error we care about most.

Slide pages verified:
- W05, halaman 41: strategy order is Evaluasi → Algoritma (`class_weight`) → Keputusan ("Geser ambang") → Data (resampling), with "Resampling bukan langkah pertama".
- W05, halaman 42: the SMOTE formula.
- W05, halaman 44: "ambang dipilih di validation set … mengikuti biaya bisnis FP vs FN".
- W05, halaman 46: "PR-AUC (kemampuan ranking) hampir tidak membaik oleh resampling/class_weight".

> **Model answer.**
> Kami tetap menguji SMOTE dan *class weight*. Keputusannya ada tiga lapis.
>
> **Materi kuliah.** Peta strategi pada W05, halaman 41, mengurutkan penanganan dari evaluasi, algoritma, lalu keputusan (menggeser ambang), dan baru terakhir *resampling*: "*resampling* bukan langkah pertama". W05, halaman 44, juga menyebut ambang dipilih pada data validasi berdasarkan biaya bisnis. Halaman 46 menunjukkan bahwa PR-AUC hampir tidak membaik oleh *resampling* atau *class weight*; yang berubah hanya titik operasinya.
>
> **Rujukan.** SMOTE dari Chawla dkk. [13] membuat titik sintetis di antara dua sampel minoritas (x_baru = x_i + λ(x_tetangga − x_i)). Pada teks, hasilnya adalah campuran frekuensi kata dua *email* yang tidak pernah ada (Subbab 8.2).
>
> **Bukti kami.** Pada Tabel 8.2, tidak ada teknik yang menaikkan Macro F1:
> - tanpa penanganan 0,9441 dengan 1.050 *email* sah hilang;
> - SMOTE 0,9371 dengan 1.102 hilang;
> - *class weight* 0,9353 dengan 1.201 hilang.
>
> Logistic Regression dengan *class weight* naik hanya 0,0004, tetapi kehilangan 153 *email* sah lebih banyak. Rasio kelasnya juga hanya 2,98:1 untuk Ham:Phish (Subbab 8.2).
>
> **Keputusan.** Kami memakai *threshold* Ham b = 1,777 (Subbab 6.4). Artinya, *email* baru keluar dari kotak masuk jika Spam atau Phish dinilai lebih dari 5,9 kali lebih mungkin daripada Ham. Pada data *testing*, *threshold* ini menyelamatkan 208 *email* sah dibandingkan b = 0: dari 302 menjadi 94 (Subbab 8.5).

**Likely follow-up: "Bukankah *threshold* juga bisa dianggap *overfitting* ke data validasi?"**
> b dipilih dari prediksi *out-of-fold*, yaitu prediksi dari model yang tidak pernah melihat *email* tersebut (Subbab 6.4). Pada data *testing*, Ham Recall 0,9963 dengan interval 0,9956–0,9971 (Tabel 8.7). Batas bawahnya tetap di atas target 99,5%, jadi b tidak terlalu pas pada data validasi.

**Weak spot to know.** The imbalance test (Tabel 8.2) was run on the original text V0, not on the final V3 text. If asked, the honest answer is that the class ratio is a property of the labels and doesn't change between text versions, so the test still applies to the same imbalance. Whether each technique would *behave* the same on V3 was not re-tested.

---

### B. Kenapa Multinomial Naive Bayes pakai frekuensi kata, bukan TF-IDF?

**For the group.** Explain the mechanism, not just the score. TF-IDF weights are tiny per message, so the fixed smoothing term α × |V| swamps them and the Phish word probabilities become almost flat. Have the four numbers from Subbab 5.1 ready.

> **Model answer.**
> **Rujukan.** Multinomial Naive Bayes memodelkan berapa kali setiap kata muncul, jadi frekuensi kata adalah masukan alaminya [8].
>
> **Bukti kami.** Pada validasi *hold-out* (Tabel 5.1), frekuensi kata menghasilkan Macro F1 0,9323 dan Phish Recall 0,9498. TF-IDF hanya 0,8286 dan 0,5144.
>
> **Penyebabnya ada di rumus *Laplace smoothing* (Subbab 5.1).** Setiap kelas mendapat tambahan α × |V| = 664.008. Dengan TF-IDF, total bobot Phish hanya 207.178, atau 0,31 kali tambahan *smoothing*, sehingga probabilitas kata Phish hampir rata. Dengan frekuensi kata, total Phish 3.249.018, atau 4,9 kali *smoothing*, sehingga masalah ini tidak terjadi.
>
> Ini juga terlihat di Subbab 5.2: menurunkan α menjadi 0,1 tanpa membuang kata menaikkan Macro F1 TF-IDF sebesar 0,0619. Jadi masalahnya memang *smoothing*.
>
> **Keputusan.** Multinomial Naive Bayes memakai frekuensi kata, sedangkan Logistic Regression tetap memakai TF-IDF dengan `sublinear_tf=True`. Logistic Regression mempelajari bobot kata secara langsung, sehingga tidak terpengaruh *smoothing*.

**Likely follow-up: "Lalu kenapa baseline tetap TF-IDF?"**
> Baseline sengaja dibuat dari pilihan awal yang paling umum dan dibekukan sebagai titik pembanding (Subbab 7.1). Perbaikannya kami tunjukkan sebagai langkah terpisah di Tabel 5.1. Kalau baseline ikut diubah, kemajuan dari eksperimen tidak lagi terlihat.

**Optional backup (side verification only).** Subbab 7.5 shows the same effect on the test data with a hand-written Naive Bayes: Macro F1 0,8377 for TF-IDF vs 0,9337 for word counts (Tabel 7.5). Use it only if they ask whether you understand the math. It is a separate check outside the main experiment flow.

---

### C. Kenapa *stopword* dihapus? Bukankah itu tidak selalu dianjurkan?

**For the group.** Agree with the premise first. Stopword removal is *not* generally recommended; Manning dkk. [2] describe the trend in IR toward small or no stop lists. Then explain that we removed stopwords to remove a **shortcut**, not to clean text. The Phish source had already been stripped of stopwords, so "no stopwords → Phish" was a free signal that real email would never give. The report says this explicitly in Subbab 4.4.

> **Model answer.**
> **Rujukan.** Betul, penghapusan *stopword* bukan praktik yang selalu dianjurkan. Manning dkk. [2] membahas bahwa sistem modern cenderung memakai daftar *stopword* kecil atau tidak memakainya sama sekali. Karena itu, baseline kami tidak menghapus *stopword* (Subbab 4.4).
>
> **Bukti kami.** Penghapusan pada V3 dilakukan karena data kami. Sumber Phish sudah tidak memiliki *stopword*: median porsi *stopword* NLTK adalah 0,0% pada Phish, dibandingkan 30,5% pada Ham dan 26,3% pada Spam (Subbab 3.6, Gambar 3.2). Model jadi bisa belajar "tidak ada *stopword* → Phish", padahal ciri ini tidak ada pada *email* nyata.
>
> Kami uji sebagai Hipotesis 5 (Subbab 8.3). Dari V2 ke V3, Macro F1 turun 0,0884 pada Multinomial Naive Bayes dan 0,0766 pada Logistic Regression, jauh di atas batas 0,005. Pada V2, 8 dari 15 kata berbobot tertinggi Logistic Regression untuk Spam adalah *stopword*; pada V3 jumlahnya nol. Artinya, skor yang lebih tinggi sebelumnya berasal dari jalan pintas format, bukan dari isi.
>
> **Keputusan.** *Stopword* dihapus untuk semua kelas sehingga format ketiga kelas sama. Daftar NLTK dipilih karena sesuai dengan daftar yang dipakai penyusun sumber Phish. Daftar scikit-learn yang lebih panjang akan ikut menghapus kata isi seperti `please`, `call`, dan `system` (Subbab 4.3).

**Likely follow-up: "Skor turun, berarti modelnya lebih buruk?"**
> Skor V3 lebih rendah, tetapi lebih jujur. Skor V0 sebagian berasal dari cara sumber Phish diproses. Phish Recall Multinomial Naive Bayes turun dari 0,9838 menjadi 0,7297 (Subbab 8.3); angka 0,9838 itu tidak akan tercapai pada *email* dari organisasi lain.

**Known side effect (be ready to admit it).** The negations `not`, `no` and `nor` are removed along with the other stopwords (Bab 14, item 5).

---

### D. Kenapa pilih *stack* B (MNB + LR + LightGBM), padahal *stack* A lebih sederhana?

**For the group.** The core argument: the two stacks are tied on Macro F1, so the business priorities decide. B lets fewer harmful emails into the inbox at the same Ham loss, and the bootstrap shows this isn't luck.

**Honesty point: know this before you go in.**
- The tie-break was changed after the first V3 run (R1). R1 used "fewer base models" as the only tie-break and picked stack A.
- After comparing the two tied stacks in detail, the rule was changed to the business priorities, with simplicity last. The final run (R2) picked B.
- This happened **before** the test data was used, and `M07/DECISIONS_M07.md` records the change.
- But Subbab 6.5 of the report says "Setiap aturan ditulis di *notebook* sebelum hasilnya terlihat". Strictly, that sentence doesn't hold for this one rule.
- If an examiner asks whether the rule was fixed in advance, use the honesty pattern below. Don't deny the change.

> **Model answer.**
> **Bukti kami.** Kedua *stack* seri pada Macro F1: 0,8855 untuk B dan 0,8847 untuk A, selisih 0,0008, di bawah toleransi seri 0,001 (Subbab 6.5, Tabel 11.2). Karena seri, pemecah serinya adalah prioritas bisnis (Subbab 2.6).
>
> Kedua *stack* kehilangan *email* sah yang sama, 511, dan 457 di antaranya adalah *email* yang sama. Namun B lebih sedikit meloloskan *email* berbahaya ke kotak masuk:
> - phishing 582 dibandingkan 746, atau 164 lebih sedikit (*bootstrap* 95%: 129–198);
> - spam 3.089 dibandingkan 3.545, atau 456 lebih sedikit (*bootstrap* 95%: 394–524).
>
> Intervalnya tidak mencakup nol, jadi bedanya bukan kebetulan. B juga lebih tinggi pada 4 dari 5 *fold*. Gambar 11.1 menunjukkan B paling rendah di wilayah perlindungan Ham yang ketat, termasuk pada target 0,5%.
>
> **Harga yang dibayar.** B menyaring 232 phishing lebih banyak ke Spam tanpa peringatan (10.444 dibandingkan 10.212), tetapi *email* itu tetap di luar kotak masuk. Biaya tambahannya satu model dasar, sekitar 3,4 menit pelatihan termasuk pembentukan matriks frekuensi (Bab 11).
>
> **Keputusan.** Kesederhanaan tetap menjadi kriteria terakhir. Di sini perbedaannya menyangkut langsung prioritas bisnis, jadi kompleksitas tambahan kami terima (Bab 12, butir 6).

**If asked: "Apakah aturan pemecah seri itu ditetapkan sebelum hasil terlihat?"** Use the honesty pattern.
> **Apa yang terjadi.** Aturan pemecah seri sempat diubah. Pada putaran pertama eksperimen V3, pemecah serinya hanya "model dasar paling sedikit", dan itu memilih *stack* A.
>
> **Dari mana perubahannya.** Setelah kedua *stack* dibandingkan secara rinci, aturan itu ternyata mengabaikan prioritas bisnis yang kami tetapkan sendiri di Subbab 2.6. Aturan diubah menjadi Phish → Ham, lalu Spam → Ham, lalu jumlah model dasar. Perubahan ini tercatat di berkas keputusan Pertemuan 7.
>
> **Kenapa tetap wajar.** Perubahan dilakukan sebelum data *testing* dipakai, jadi data *testing* tidak memengaruhi pilihan. Urutan barunya sama dengan prioritas di Subbab 2.6, yang sudah ada sejak awal. Bedanya juga terbukti dengan *bootstrap*.

---

### E. Kenapa deduplikasi dilakukan sebelum *split*?

**For the group.** An examiner may quote W07, halaman 35: "Lakukan train-test split … SEBELUM melakukan eksplorasi atau preprocessing apa pun". The answer is the slide's own "aturan emas" on the same page: anything that calls `.fit()` or "learns" from data must go inside the per-fold pipeline. Deduplication learns nothing. It doesn't compute a mean, a vocabulary or a weight; it just drops identical copies. Doing it **after** the split would leave the same text on both sides, which is exactly the leakage the slide warns about.

> **Model answer.**
> **Materi kuliah.** *Checklist* W07, halaman 35, memang meminta *split* sebelum *preprocessing*. Aturan emas di halaman yang sama menjelaskan alasannya: apa pun yang memanggil `.fit()` atau "belajar" dari data harus berada di dalam *pipeline* yang dilatih ulang per *fold*. Deduplikasi tidak belajar apa pun dari data. Ia tidak menghitung rata-rata, kosakata, atau bobot, tetapi hanya membuang salinan teks yang identik. Semua langkah yang belajar dari data, yaitu *vectorizer* dan *resampling*, ada di dalam *pipeline* dan dilatih ulang pada setiap *fold* (Subbab 4.2).
>
> **Bukti kami.** Ada 86.460 duplikat berlabel sama setelah normalisasi spasi dan huruf kecil (Tabel 3.2). Jika *split* dilakukan dulu, salinan teks yang sama bisa berada di data *training* dan data *testing* sekaligus. Model lalu diuji pada teks yang sudah dipelajarinya. Itulah *data leakage* yang dihindari (Subbab 4.1).
>
> Teks yang sama dengan label berbeda, 14 kelompok atau 44 baris, dihapus seluruhnya karena labelnya tidak dapat ditentukan (Tabel 3.2).
>
> **Keputusan.** Deduplikasi dilakukan sebelum *split*. Hasilnya 278.941 pesan unik (Tabel 4.1), yang dibagi 223.152 untuk data *training* dan 55.789 untuk data *testing* (Tabel 4.2).

**Likely follow-up: "Bagaimana dengan teks yang hampir sama?"** Admit the limit, then bound it.
> Deduplikasi kami hanya menghapus teks identik. Sebanyak 55,2% *email* validasi masih memiliki pasangan dengan kemiripan ≥ 0,9 di data *training* (Subbab 9.4). Namun, tingkat kesalahan model akhir pada *email* tersebut 7,37%, hampir sama dengan 7,58% pada *email* lainnya. Jadi teks yang hampir sama tidak menaikkan skor secara berarti. Hal ini tetap kami catat sebagai keterbatasan (Bab 14, butir 2).

---

## 4. Questions that need an honest limitation

### F. Dari mana `num_leaves=63` dan `learning_rate=0.2`?

**For the group.**
- **Origin.** These two values, and `colsample_bytree=0.3`, came from a preliminary time-and-quality test on a sample of the data before the M05 notebook was run. The planning notes give the sample as about 15%.
- **What's missing.** The test's results were not saved in any notebook. That's why the report says only "uji pendahuluan di luar *notebook* (tidak tercatat)" (Subbab 7.4, Tabel 10.1).
- **Don't quote any pilot score.** Earlier drafts had "0,9677 vs 0,9675". It was removed because it couldn't be traced, and 0,9677 is actually the score of a different model in M05.
- **The bounding argument.** The number of trees, which interacts most with `learning_rate`, *was* re-tuned on V3 (Gambar 10.1(b)). Bab 14, item 7, lists the rest as a limitation.

> **Model answer.**
> **Apa itu.** `learning_rate=0.2`, `num_leaves=63`, dan `colsample_bytree=0.3` tidak dipilih dengan *grid search*. Nilainya ditetapkan dari uji waktu dan kualitas pada sampel data sebelum *notebook* Pertemuan 5 dijalankan. Hasil uji itu tidak tersimpan di *notebook*, jadi di laporan kami menulisnya terus terang sebagai uji pendahuluan yang tidak tercatat (Subbab 7.4, Tabel 10.1).
>
> **Kenapa tidak di-*tuning* penuh.** Satu kali *training* lima *fold* LightGBM saja sudah memakan sekitar 119 menit (Subbab 7.4).
>
> **Kenapa tetap wajar.** Parameter yang paling berinteraksi dengan `learning_rate`, yaitu jumlah pohon, kami *tuning* ulang pada V3. Hasilnya 125 pohon: skor validasi 0,8875 dan mendatar setelahnya (Subbab 9.6, Gambar 10.1(b)). Parameter yang belum di-*tuning* ulang kami cantumkan sebagai keterbatasan (Bab 14, butir 7) dan menjadi rencana setelah UTS.

---

### G. Kenapa di Bab 11 tertulis 510–511, tidak sama persis?

**For the group.**
- **How b is chosen.** For each model, b is the smallest value that gives Ham Recall ≥ 99,5% on the out-of-fold predictions. b lands exactly on the score of one Ham email.
- **Why the count differs.** That boundary email sits on the threshold to within about 10⁻¹⁶, so floating-point rounding decides which side it falls on. It counts as lost for three models (B, A and MNB + LightGBM) and as kept for the other four. Hence 510–511.
- **Exact numbers, for your own understanding.** 0,5% of 102.192 is 510,96. Losing 511 gives Ham Recall 0,99499961, which is 0,9950 at four decimals but technically 0,0000004 under the target. Losing 510 gives 0,99500939. Only bring this up if pressed.
- **Why it doesn't matter.** On the test data, Ham Recall is 0,9963 and the whole 95% interval is above the target (Tabel 8.7).

> **Model answer.**
> **Apa itu.** b dipilih sebagai nilai terkecil yang memberi Ham Recall ≥ 99,5% pada prediksi *out-of-fold* (Subbab 6.4). Nilai b itu jatuh tepat pada skor satu *email* sah. *Email* tersebut berada persis di batas *threshold*, sehingga pembulatan angka di komputer yang menentukan apakah ia ikut hilang atau tidak. Karena itu, sebagian model kehilangan 510 dan sebagian 511 dari 102.192 *email* sah (Bab 11).
>
> **Kenapa tetap wajar.** Selisih satu *email* dari lebih dari seratus ribu tidak mengubah perbandingan apa pun. Pada Tabel 8.4, Ham Recall model akhir tertulis 0,9950 dalam empat desimal. Yang lebih penting, pada data *testing* Ham Recall-nya 0,9963 dengan interval 95% 0,9956–0,9971 (Tabel 8.7). Batas bawahnya pun di atas target 99,5%.

---

### H. 179 atau 153 *stopword*?

**For the group.**
- **What the report says.** The report now gives only **153** (Subbab 4.3). That's the number of unique words the code actually removes. You can count it directly from `final_config.json`.
- **Where 179 came from.** A comment in the M07 notebook, describing the full NLTK English list, which includes apostrophe forms such as "don't".
- **Why it was removed.** NLTK isn't installed on the checking machine, so 179 couldn't be verified.
- **Why the apostrophe forms don't matter.** V3 turns punctuation into spaces first, so an apostrophe form can never appear in the processed text.

> **Model answer.**
> **Yang dipakai 153.** Kode kami menghapus 153 kata unik dari daftar *stopword* NLTK, yaitu daftar tanpa bentuk yang memakai tanda apostrof (Subbab 4.3). Daftar lengkapnya tersimpan di `final_config.json` dan bisa dihitung langsung.
>
> **Asal angka 179.** Angka itu muncul di komentar *notebook* untuk daftar NLTK lengkap, yang juga memuat bentuk berapostrof seperti "don't". Bentuk seperti itu tidak relevan bagi kami. Pada V3, tanda baca diganti spasi lebih dulu, sehingga kata berapostrof sudah terpecah sebelum *stopword* dihapus.
>
> **Ringkasnya.** Yang benar-benar diterapkan pada data adalah 153 kata, dan itulah yang kami laporkan.

---

### I. Siapa yang bilang 3:1 itu ketidakseimbangan "sedang"?

**For the group.**
- **"Sedang" is our judgement.** There is no universal cut-off. Don't claim a source defines it.
- **Comparison point.** The course shows what severe imbalance looks like: W05, halaman 37, the credit-card fraud data, with 492 fraud cases in 284.807 transactions. That is 0,172%, a ratio of 1 : 578.
- **Our data.** Phish is 15,36% of the clean data (Tabel 3.1), a Ham:Phish ratio of 2,98:1.
- **The deciding evidence.** None of the imbalance techniques helped (Tabel 8.2). We don't rest the decision on the word "sedang".

> **Model answer.**
> **Apa itu.** Istilah "sedang" adalah penilaian kami, bukan batas baku dari satu sumber. Tidak ada angka universal yang memisahkan ketidakseimbangan ringan, sedang, dan berat.
>
> **Dasar penilaian.** Pembandingnya contoh pada materi W05, halaman 37: data *fraud* kartu kredit dengan proporsi kelas minoritas 0,172%, atau rasio 1 : 578. Pada data kami, Phish mencakup 15,36% data bersih, dengan rasio Ham:Phish 2,98:1 (Tabel 3.1). Itu jauh dari kasus ekstrem seperti *fraud*.
>
> **Yang sebenarnya menentukan.** Keputusan kami tidak bergantung pada label "sedang", tetapi pada hasil eksperimen. Empat teknik penanganan tidak menaikkan Macro F1 dan semuanya menghilangkan lebih banyak *email* sah (Tabel 8.2). Meski begitu, kami tetap memakai Macro F1 dan metrik per kelas, karena akurasi bisa menyesatkan pada data tidak seimbang [4]. Pada baseline, akurasi 0,8881, padahal Phish Recall hanya 0,5316 (Subbab 3.2).

---

### J. How to answer when you don't remember a number

**For the group.** Never guess. A wrong number said confidently is worse than "it's in Tabel 8.6". Say where the number is recorded, give the approximate value only if you're sure of it, and offer to open the table.

> **Model answer pattern.**
> "Angka persisnya ada di Tabel 8.6 pada Subbab 8.5. Seingat kami, *email* sah yang hilang turun hampir separuh dibandingkan baseline. Boleh saya buka tabelnya supaya angkanya tepat?"

**Where everything is recorded:**
- **The report:** the table or figure named in each answer above.
- **`Final Progress/CONSTANTS.md`:** every setting and constant, with how it was chosen.
- **`M0x/DECISIONS_M0x.md` and `Final Evaluation/DECISIONS_FINAL.md`:** the reason for each decision.
- **The executed notebooks:** the raw outputs, which are the ground truth.

**Key numbers worth memorising** (all from the current report):

| Topic | Number | Where |
|---|---|---|
| Data before and after cleaning | 365.448 → 278.941 | Tabel 3.1, Tabel 4.1 |
| Data *training* / data *testing* | 223.152 / 55.789 | Tabel 4.2 |
| Phish share, Ham:Phish ratio | 15,36%, 2,98:1 | Tabel 3.1 |
| Baseline on data *testing* | Akurasi 0,8881, Macro F1 0,8377, Phish Recall 0,5316 | Tabel 8.5 |
| Final model on data *testing* | Macro F1 0,8913 (0,8878–0,8946), Ham Recall 0,9963 (0,9956–0,9971) | Tabel 8.5, Tabel 8.7 |
| Priority errors, baseline → final | Ham lost 174 → 94; Phish → Ham 538 → 107; Spam → Ham 1.991 → 781 | Tabel 8.6 |
| Threshold | b = 1,777, about 5,9 times | Subbab 6.4 |
| CV, stack B vs A | 0,8855 ± 0,0011 vs 0,8847 | Tabel 8.4, Tabel 11.2 |
| H4 drop (V0 → V2) | 0,0444 MNB, 0,0263 LR | Subbab 8.3 |
| H5 drop (V2 → V3) | 0,0884 MNB, 0,0766 LR | Subbab 8.3 |
| SMOTE vs none | 0,9371 vs 0,9441 | Tabel 8.2 |
| TF-IDF vs word counts (MNB) | 0,8286 vs 0,9323 | Tabel 5.1 |

---

## 5. Phrasing table

| Don't say | Say instead |
|---|---|
| "Akurasinya 92%." | "Macro F1 pada data *testing* 0,8913, dan *email* sah yang hilang 94 dari 25.548." Accuracy alone can mislead on imbalanced data (Subbab 3.2). |
| "Kami coba-coba saja sampai bagus." | "Setiap pilihan diuji satu faktor per eksperimen, dengan aturan pemilihan yang ditulis di *notebook* (Subbab 6.2, 6.5)." |
| "SMOTE itu jelek." | "Pada data dan metrik kami, SMOTE tidak menaikkan Macro F1 dan menambah *email* sah yang hilang (Tabel 8.2)." |
| "*Stopword* memang harus dihapus." | "*Stopword* dihapus karena terbukti menjadi ciri sumber Phish (Hipotesis 5), bukan sebagai praktik umum (Subbab 4.4)." |
| "Hampir sama." | "Selisihnya 0,0008, di bawah toleransi seri 0,001." |
| "Kami pilih B karena lebih bagus." | "Macro F1-nya seri; B dipilih karena 164 phishing dan 456 spam lebih sedikit masuk kotak masuk, pada *email* sah hilang yang sama (Tabel 11.2)." |
| "Itu nilai *default*." | "Nilai itu tidak di-*tuning* pada V3, dan kami mencatatnya sebagai keterbatasan (Bab 14, butir 7)." |
| "Model kami bisa dipakai di dunia nyata." | "Pada dataset ini hasilnya baik, tetapi pada *email* dari organisasi lain performa bisa lebih rendah karena *dataset shift* [1] (Bab 14)." |
| "Data *testing*-nya kami pakai untuk mengecek pilihan." | "Data *testing* hanya dipakai di akhir; semua pilihan dibuat dengan *cross-validation* (Subbab 6.1)." |
| "Skornya turun, jadi modelnya lebih buruk." | "Skor V3 lebih rendah tetapi lebih jujur, karena jalan pintas format sudah dihapus (Subbab 8.3)." |
| "Saya lupa angkanya, kira-kira 0,9 sekian." | "Angkanya ada di Tabel 8.5; boleh saya buka supaya tepat?" |
| "Itu sudah pasti benar." | "Pada data ini, buktinya adalah [angka dan tabel]." |

---

## 6. Closing message

If you get a chance to sum up, or are asked "apa kesimpulan proses kalian?", close with the same idea the whole report is built on.

> **Model answer.**
> Setiap keputusan kami **berdasarkan kerangka mata kuliah**, mulai dari *Business Understanding* sampai evaluasi satu kali pada data *testing*. Keputusan itu **didukung literatur**, dan **dibuktikan dengan eksperimen kami sendiri yang terdokumentasi**.
>
> Ketika hasil eksperimen bertentangan dengan rencana awal, kami mengikuti buktinya:
> - SMOTE tidak dipakai (Tabel 8.2);
> - Multinomial Naive Bayes memakai frekuensi kata (Tabel 5.1);
> - jalan pintas format sumber dihapus walaupun skor turun (Subbab 8.3).
>
> Keterbatasan yang tersisa kami tulis terbuka di Bab 14.

**For the group.** The point to leave the examiner with: the project's value isn't the highest number, it's that every decision can be traced from the course material, through a reference, to our own evidence, and that the weak points are stated openly.
