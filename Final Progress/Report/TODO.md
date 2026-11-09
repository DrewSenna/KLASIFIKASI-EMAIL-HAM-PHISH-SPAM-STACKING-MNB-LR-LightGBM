# TODO — Laporan UTS Kelompok 1

## Before converting to Word

- [ ] Write **Bab 16 Rencana Pengembangan Setelah UTS** (currently a placeholder: "*(Disusun terpisah.)*").
- [ ] Add the member names to the report header.

## Bab 16: what the guideline asks for

- **§8, item 12:** "Rencana pengembangan setelah UTS" is one of the minimum UTS contents.
- **§9 Tahap Setelah UTS:** use the early results to decide the next experiments. Possible steps:
  - developing the candidate models;
  - feature selection;
  - validation;
  - error analysis;
  - class imbalance handling (if relevant);
  - threshold analysis (if relevant);
  - hyperparameter tuning;
  - model comparison;
  - model interpretation.

## Bab 16: open items already in the report

1. **Tuning on V3.** Logistic Regression's `C` and three LightGBM parameters (`learning_rate`, `num_leaves`, `colsample_bytree`) were tuned only on V0 or a 15% V0 sample (Bab 10, 14.7).
2. **Negation words.** V3 also deletes `not`, `no` and `nor`. Re-test V3 with negation words kept (9.7).
3. **Dataset shift.** Test on an email dataset from another source. The current test set comes from the same corpora, so it does not measure performance on new sources (14.1–14.2).
4. **Very short messages.** Messages of 10 words or fewer have the highest error rates (9.3).
5. **Near-duplicate templates.** Group near-identical template emails so the same template does not appear in both training and validation (9.4).
6. **The test set has already been used once.** Any model changed after UTS cannot be scored fairly on the same test set again, so the plan needs a new evaluation method.

## Outside the report

- [ ] Replace the local `M07/m07_artifacts/final_config.json` with the Drive copy. The local copy still names stack A (b = 1,926), while the final model is stack B (b = 1,777).
