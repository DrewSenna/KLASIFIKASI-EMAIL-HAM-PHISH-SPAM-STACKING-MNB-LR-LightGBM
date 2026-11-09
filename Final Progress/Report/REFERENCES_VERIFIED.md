# Verified References — Group 1 Ham/Phish/Spam (IF2116)

Prepared 6 Nov 2026 for the final report. Each entry was checked against Crossref, the publisher page, dblp, JMLR, the ACL Anthology, the NeurIPS proceedings, or arXiv. Format: APA 7 with a DOI when one exists. "→" says which claim in our report the entry supports.

**Spot-check:** 7 DOIs (AlShaikh 2025, Maslan 2025, Elangovan 2021, Champa 2024, Al-Subaiey 2024, Rastenis 2021, Firmansyah 2025) were re-checked against Crossref and match.

---

## 0. What the dataset card says (important for Chapter 3 and Limitations)

The Hugging Face card (`locuoco/the-biggest-spam-ham-phish-email-dataset-300000`) was fetched on 6 Nov 2026. Its key points:

- **Title:** "The Biggest Spam Ham Phish Email Dataset (250000+)". The repository name says 300000.
- **Self-description:** "large-scale, unified, and **deduplicated** collection … constructed by combining multiple publicly available and open-source datasets". It claims "approximately unique 250,000+ samples".
- **Labels:** 0 → Ham, 1 → Phish, 2 → Spam.
- **Licence and origin:** MIT licence, mirrored from Kaggle `akshatsharma2/the-biggest-spam-ham-phish-email-dataset-300000`.
- **Sources:** neither the Hugging Face card nor the Kaggle card names the source corpora.

What this means for the report:
1. **Provenance gap.** The card does not say which corpora the data came from. State this as a data limitation.
2. **The card contradicts our data.** It claims "deduplicated", but we found duplicates and 51.7% near-duplicates between validation and training. This strengthens our leakage finding.
3. **Sources are inferred.** Describe them as inferred from artifacts, not documented:
   - `enron`, `vince`, `louise` in Ham point to Enron.
   - `josemonkeyorg` in Phish points to jose@monkey.org, the Nazario phishing corpus.
   - `uwaterloo` in Spam points to the TREC public corpora (University of Waterloo).
4. **Likely upstream aggregates (unconfirmed).** Champa (2023), on Zenodo, combines Enron, SpamAssassin, TREC 05/06/07, CEAS_08, Nazario, Ling and Nigerian Fraud. Al-Subaiey et al. (2024), on Kaggle, combines a similar set. Which aggregate the Kaggle author actually merged is not confirmed.

---

## 1. Email spam / phishing detection with ML

- Sahami, M., Dumais, S., Heckerman, D., & Horvitz, E. (1998). A Bayesian approach to filtering junk e-mail. In *Learning for Text Categorization: Papers from the 1998 Workshop* (AAAI Technical Report WS-98-05, pp. 55–62). AAAI Press. https://cdn.aaai.org/Workshops/1998/WS-98-05/WS98-05-009.pdf
  - → NB is the historical spam-filter baseline, and blocking legitimate mail costs more than missing spam.
- Androutsopoulos, I., Koutsias, J., Chandrinos, K. V., & Spyropoulos, C. D. (2000). An experimental comparison of naive Bayesian and keyword-based anti-spam filtering with personal e-mail messages. In *Proceedings of the 23rd Annual International ACM SIGIR Conference* (pp. 160–167). ACM. https://doi.org/10.1145/345508.345569
  - → Cost-sensitive evaluation that weights blocked Ham heavily. Supports the Ham Recall ≥ 99,5% rule.
- Androutsopoulos, I., Koutsias, J., Chandrinos, K. V., Paliouras, G., & Spyropoulos, C. D. (2000). An evaluation of naive Bayesian anti-spam filtering. In *Proceedings of the Workshop on Machine Learning in the New Information Age, 11th European Conference on Machine Learning* (pp. 9–17). https://arxiv.org/abs/cs/0006013
  - → Introduced the Ling-Spam corpus.
- Metsis, V., Androutsopoulos, I., & Paliouras, G. (2006). Spam filtering with naive Bayes – which naive Bayes? In *Proceedings of the 3rd Conference on Email and Anti-Spam (CEAS 2006)*. Mountain View, CA. *(no DOI or pages)*
  - → Compares multinomial NB variants for spam. Introduced the Enron-Spam corpus.
- Cormack, G. V. (2008). Email spam filtering: A systematic review. *Foundations and Trends in Information Retrieval, 1*(4), 335–455. https://doi.org/10.1561/1500000006
  - → Standard review of spam filtering and its evaluation methodology.
- Fette, I., Sadeh, N., & Tomasic, A. (2007). Learning to detect phishing emails. In *Proceedings of the 16th International Conference on World Wide Web* (pp. 649–656). ACM. https://doi.org/10.1145/1242572.1242660
  - → Phishing is a separate target from spam, which justifies a separate Phish class.
- Abu-Nimeh, S., Nappa, D., Wang, X., & Nair, S. (2007). A comparison of machine learning techniques for phishing detection. In *Proceedings of the Anti-Phishing Working Groups 2nd Annual eCrime Researchers Summit* (pp. 60–69). ACM. https://doi.org/10.1145/1299015.1299021
  - → LR and other classical models are competitive for phishing detection.
- Dada, E. G., Bassi, J. S., Chiroma, H., Abdulhamid, S. M., Adetunmbi, A. O., & Ajibuwa, O. E. (2019). Machine learning for email spam filtering: Review, approaches and open research problems. *Heliyon, 5*(6), Article e01802. https://doi.org/10.1016/j.heliyon.2019.e01802
  - → Survey context.
- Karim, A., Azam, S., Shanmugam, B., Kannoorpatti, K., & Alazab, M. (2019). A comprehensive survey for intelligent spam email detection. *IEEE Access, 7*, 168261–168295. https://doi.org/10.1109/ACCESS.2019.2954791
  - → Survey of methods and datasets.
- Gangavarapu, T., Jaidhar, C. D., & Chanduka, B. (2020). Applicability of machine learning in spam and phishing email filtering: Review and approaches. *Artificial Intelligence Review, 53*(7), 5019–5081. https://doi.org/10.1007/s10462-020-09814-9
  - → One of the few reviews that covers spam and phishing together, which fits our 3-class framing.
- Salloum, S., Gaber, T., Vadera, S., & Shaalan, K. (2022). A systematic literature review on phishing email detection using natural language processing techniques. *IEEE Access, 10*, 65703–65727. https://doi.org/10.1109/ACCESS.2022.3183083
  - → TF-IDF and bag-of-words remain common features for phishing detection.
- Ahmed, N., Amin, R., Aldabbas, H., Koundal, D., Alouffi, B., & Shah, T. (2022). Machine learning techniques for spam detection in email and IoT platforms: Analysis and research challenges. *Security and Communication Networks, 2022*, Article 1862888. https://doi.org/10.1155/2022/1862888
  - → *(Our PDF.)* Recent survey.
- AlShaikh, M., Alrajeh, Y., Alamri, S., Melhem, S., & Abu-Khadrah, A. (2025). Supervised methods of machine learning for email classification: A literature survey. *Systems Science & Control Engineering, 13*(1), Article 2474450. https://doi.org/10.1080/21642583.2025.2474450
  - → *(Our PDF.)* Recent survey of NB, SVM, DT and RF for email classification.
- Chaudhuri, N. B., & Chaudhuri, S. B. (2024). Designing an intelligent architecture to detect malevolent e-mail using machine learning approach. In *2024 1st International Conference on Cognitive, Green and Ubiquitous Computing (IC-CGU)* (pp. 1–6). IEEE. https://doi.org/10.1109/IC-CGU58078.2024.10530788
  - → *(Our PDF.)*
- Rastenis, J., Ramanauskaitė, S., Suzdalev, I., Tunaitytė, K., Janulevičius, J., & Čenys, A. (2021). Multi-language spam/phishing classification by email body text: Toward automated security incident investigation. *Electronics, 10*(6), Article 668. https://doi.org/10.3390/electronics10060668
  - → The closest precedent for separating spam from phishing using body text.
- Tian, Y., Dai, X., Li, Z., Guo, H., & Mao, X. (2025). Improving the accuracy of cybersecurity spam email detection using ensemble techniques: A stacking approach. *PLOS ONE, 20*(9), Article e0331574. https://doi.org/10.1371/journal.pone.0331574
  - → Recent precedent for stacking in spam detection.

No peer-reviewed paper was found with exactly a 3-class Ham/Phish/Spam setup on email. Gangavarapu (2020) and Rastenis (2021) are the closest. This can be stated as part of the report's contribution.

## 2. Source corpora (inferred; see §0)

- Klimt, B., & Yang, Y. (2004). The Enron corpus: A new dataset for email classification research. In *Machine Learning: ECML 2004* (LNCS Vol. 3201, pp. 217–226). Springer. https://doi.org/10.1007/978-3-540-30115-8_22
- Cormack, G., & Lynam, T. (2005). TREC 2005 spam track overview. In *The Fourteenth Text REtrieval Conference (TREC 2005)* (NIST SP 500-266). NIST. https://trec.nist.gov/pubs/trec14/papers/SPAM.OVERVIEW.pdf
- Cormack, G. V. (2007). TREC 2007 spam track overview. In *The Sixteenth Text REtrieval Conference (TREC 2007)* (NIST SP 500-274). NIST. https://trec.nist.gov/pubs/trec16/papers/SPAM.OVERVIEW16.pdf
- Apache SpamAssassin Project. (n.d.). *SpamAssassin public mail corpus* [Data set]. https://spamassassin.apache.org/old/publiccorpus/
  - Open this page yourselves before writing the access date; the agent's fetch timed out.
- Nazario, J. (n.d.). *Phishing corpus* [Data set]. https://monkey.org/~jose/phishing/
- Radev, D. (2008). *CLAIR collection of fraud email* (ACL Data and Code Repository, ADCR2008T001) [Data set]. https://www.aclweb.org/aclwiki/CLAIR_collection_of_fraud_emails_(Repository)
- Champa, A. I., Rabbi, M. F., & Zibran, M. F. (2024). Curated datasets and feature analysis for phishing email detection with machine learning. In *2024 IEEE 3rd International Conference on Computing and Machine Intelligence (ICMI)* (pp. 1–7). IEEE. https://doi.org/10.1109/ICMI60790.2024.10585821
- Champa, A. I. (2023). *Phishing email curated datasets* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.8339691
- Al-Subaiey, A., Al-Thani, M., Abdullah Alam, N., Antora, K. F., Khandakar, A., & Uz Zaman, S. M. A. (2024). Novel interpretable and robust web-based AI platform for phishing email detection. *Computers and Electrical Engineering, 120*, Article 109625. https://doi.org/10.1016/j.compeleceng.2024.109625

Do not cite a "CEAS 2008 corpus paper" (none exists) or a "Ling-Spam paper" other than Androutsopoulos et al. (2000b).

## 3. Text representation, feature selection, clustering

- Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation, 28*(1), 11–21. https://doi.org/10.1108/eb026526
  - → Origin of IDF weighting.
- Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management, 24*(5), 513–523. https://doi.org/10.1016/0306-4573(88)90021-0
  - → TF-IDF variants and sublinear tf.
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to information retrieval*. Cambridge University Press. https://doi.org/10.1017/CBO9780511809071
  - → Bag-of-words, sublinear tf, multinomial NB and χ² feature selection.
- Sebastiani, F. (2002). Machine learning in automated text categorization. *ACM Computing Surveys, 34*(1), 1–47. https://doi.org/10.1145/505282.505283
  - → Text classification framed as vector-space features plus a supervised learner.
- Yang, Y., & Pedersen, J. O. (1997). A comparative study on feature selection in text categorization. In *Proceedings of the Fourteenth International Conference on Machine Learning (ICML '97)* (pp. 412–420). Morgan Kaufmann.
  - → χ² is among the most effective criteria for selecting terms.
- Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research, 3*, 1157–1182. https://jmlr.org/papers/v3/guyon03a.html
- Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harshman, R. (1990). Indexing by latent semantic analysis. *Journal of the American Society for Information Science, 41*(6), 391–407. https://doi.org/10.1002/(SICI)1097-4571(199009)41:6<391::AID-ASI1>3.0.CO;2-9
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics, 20*, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7
- Kaufman, L., & Rousseeuw, P. J. (1990). *Finding groups in data: An introduction to cluster analysis*. Wiley. https://doi.org/10.1002/9780470316801

## 4. Models and ensembles

- McCallum, A., & Nigam, K. (1998). A comparison of event models for naive Bayes text classification. In *Learning for Text Categorization: Papers from the 1998 AAAI Workshop* (AAAI Technical Report WS-98-05, pp. 41–48). AAAI Press. https://cdn.aaai.org/Workshops/1998/WS-98-05/WS98-05-007.pdf
  - → Multinomial NB beats Bernoulli NB on large vocabularies.
- Lewis, D. D. (1998). Naive (Bayes) at forty: The independence assumption in information retrieval. In *Machine Learning: ECML-98* (LNCS Vol. 1398, pp. 4–15). Springer. https://doi.org/10.1007/BFb0026666
- Rennie, J. D. M., Shih, L., Teevan, J., & Karger, D. R. (2003). Tackling the poor assumptions of naive Bayes text classifiers. In *Proceedings of the Twentieth International Conference on Machine Learning (ICML-2003)* (pp. 616–623). AAAI Press.
  - → Known MNB weaknesses: skewed classes and length effects.
- Ng, A. Y., & Jordan, M. I. (2002). On discriminative vs. generative classifiers: A comparison of logistic regression and naive Bayes. In *Advances in Neural Information Processing Systems 14*. MIT Press. https://proceedings.neurips.cc/paper_files/paper/2001/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html
  - → NB converges faster, but LR reaches lower asymptotic error. This explains why LR beats MNB on our ~300k emails.
  - Pages are omitted because they are not shown on the official site.
- Wang, S., & Manning, C. D. (2012). Baselines and bigrams: Simple, good sentiment and topic classification. In *Proceedings of the 50th Annual Meeting of the ACL (Volume 2: Short Papers)* (pp. 90–94). ACL. https://aclanthology.org/P12-2018/
  - → Bigrams with NB or linear models make strong text baselines.
- Joachims, T. (1998). Text categorization with support vector machines: Learning with many relevant features. In *Machine Learning: ECML-98* (LNCS Vol. 1398, pp. 137–142). Springer. https://doi.org/10.1007/BFb0026683
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232. https://doi.org/10.1214/aos/1013203451
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems 30*. Curran Associates. https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks, 5*(2), 241–259. https://doi.org/10.1016/S0893-6080(05)80023-1
- Ting, K. M., & Witten, I. H. (1999). Issues in stacked generalization. *Journal of Artificial Intelligence Research, 10*, 271–289. https://doi.org/10.1613/jair.594
  - → The meta-learner should use class probabilities, not hard labels. Justifies stacking on OOF log-probabilities.
- Dietterich, T. G. (2000). Ensemble methods in machine learning. In *Multiple Classifier Systems* (LNCS Vol. 1857, pp. 1–15). Springer. https://doi.org/10.1007/3-540-45014-9_1
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*(85), 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

## 5. Evaluation, model selection, thresholds

- Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management, 45*(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002
  - → Macro-averaging weights every class equally, which is why we use Macro F1.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), Article e0118432. https://doi.org/10.1371/journal.pone.0118432
  - → Also on the W05 slide reading list.
- Davis, J., & Goadrich, M. (2006). The relationship between precision-recall and ROC curves. In *Proceedings of the 23rd International Conference on Machine Learning* (pp. 233–240). ACM. https://doi.org/10.1145/1143844.1143874
- Elkan, C. (2001). The foundations of cost-sensitive learning. In *Proceedings of the Seventeenth International Joint Conference on Artificial Intelligence (IJCAI-01)* (pp. 973–978). Morgan Kaufmann.
  - → With unequal error costs, shift the decision threshold. This is the theoretical basis for the Ham bias `b`.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. In *Proceedings of the 22nd International Conference on Machine Learning* (pp. 625–632). ACM. https://doi.org/10.1145/1102351.1102430
  - → NB and boosted trees are poorly calibrated, so `b` is a tuned offset, not a probability.
- Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall. https://doi.org/10.1201/9780429246593
- Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. In *Proceedings of the Fourteenth International Joint Conference on Artificial Intelligence (IJCAI-95)* (Vol. 2, pp. 1137–1143). Morgan Kaufmann. https://www.ijcai.org/proceedings/1995/147
  - → Stratified 10-fold CV is the recommended method for model selection (abstract).
  - **Pages resolved on 6 Nov 2026:** the official IJCAI PDF (`Proceedings/95-2/Papers/016.pdf`) is 7 pages with printed numbers 1137–1143. The IJCAI web record says 1137–1144, so cite the printed range. The local copy `References for ML/Kohavi 1995.pdf` is the author's 7-page preprint ("Appears in IJCAI 1995").
- Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*(70), 2079–2107. https://jmlr.org/papers/v11/cawley10a.html
  - → Choose `b` and the model on CV only; touch the test set once.
- Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics, 7*, Article 91. https://doi.org/10.1186/1471-2105-7-91
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research, 13*(10), 281–305. https://jmlr.org/papers/v13/bergstra12a.html
  - → Also on the W05 slide reading list.

## 6. Class imbalance

- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357. https://doi.org/10.1613/jair.953
- He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering, 21*(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239
- Japkowicz, N., & Stephen, S. (2002). The class imbalance problem: A systematic study. *Intelligent Data Analysis, 6*(5), 429–449. https://doi.org/10.3233/IDA-2002-6504
  - → Imbalance hurts mainly when combined with class overlap. At moderate imbalance, reweighting is enough, which matches our M05 result.
- Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning. *Journal of Machine Learning Research, 18*(17), 1–5. https://jmlr.org/papers/v18/16-365.html

## 7. Leakage, shortcut learning, duplicates, dataset shift

These support H4/H5 and the near-duplicate finding.

- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data, 6*(4), 1–21. https://doi.org/10.1145/2382577.2382579
- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns, 4*(9), Article 100804. https://doi.org/10.1016/j.patter.2023.100804
  - → Its leakage taxonomy includes train/test duplicates and sampling bias. Both match our findings.
- Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence, 2*(11), 665–673. https://doi.org/10.1038/s42256-020-00257-z
  - → Models learn spurious source cues (here, the preprocessing format).
- Gururangan, S., Swayamdipta, S., Levy, O., Schwartz, R., Bowman, S., & Smith, N. A. (2018). Annotation artifacts in natural language inference data. In *Proceedings of NAACL-HLT 2018, Volume 2 (Short Papers)* (pp. 107–112). ACL. https://doi.org/10.18653/v1/N18-2017
  - → Text analogue: class-specific artifacts let bag-of-words models "solve" a task.
- Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A., & Lawrence, N. D. (Eds.). (2009). *Dataset shift in machine learning*. MIT Press. https://doi.org/10.7551/mitpress/9780262170055.001.0001
- Moreno-Torres, J. G., Raeder, T., Alaiz-Rodríguez, R., Chawla, N. V., & Herrera, F. (2012). A unifying view on dataset shift in classification. *Pattern Recognition, 45*(1), 521–530. https://doi.org/10.1016/j.patcog.2011.06.019
- Jáñez-Martino, F., Alaiz-Rodríguez, R., González-Castro, V., Fidalgo, E., & Alegre, E. (2023). A review of spam email detection: Analysis of spammer strategies and the dataset shift problem. *Artificial Intelligence Review, 56*(2), 1145–1173. https://doi.org/10.1007/s10462-022-10195-4
- Lee, K., Ippolito, D., Nystrom, A., Zhang, C., Eck, D., Callison-Burch, C., & Carlini, N. (2022). Deduplicating training data makes language models better. In *Proceedings of the 60th Annual Meeting of the ACL (Volume 1: Long Papers)* (pp. 8424–8445). ACL. https://doi.org/10.18653/v1/2022.acl-long.577
- Elangovan, A., He, J., & Verspoor, K. (2021). Memorization vs. generalization: Quantifying data leakage in NLP performance evaluation. In *Proceedings of the 16th Conference of the European Chapter of the ACL: Main Volume* (pp. 1325–1335). ACL. https://doi.org/10.18653/v1/2021.eacl-main.113
  - → The closest match to our 51.7% near-duplicate finding.

## 8. Interpretability

- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324
  - → Origin of permutation importance; also on the W05 slide reading list.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1135–1144). ACM. https://doi.org/10.1145/2939672.2939778
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30*. Curran Associates. https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
- Molnar, C. (2025). *Interpretable machine learning: A guide for making black box models explainable* (3rd ed.). https://christophm.github.io/interpretable-ml-book/
- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. In *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT\* '19)* (pp. 220–229). ACM. https://doi.org/10.1145/3287560.3287596
  - → On the W07 reading list (p. 47–48, 51). Verified in Crossref (pp. 220–229). Supports the model-card summary in Chapter 15.

## 9. Methodology and course textbooks

- Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS.
  - Verified only through mirrors. This is the source cited on slide W01 p. 24.
- Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. In *Proceedings of the 4th International Conference on the Practical Applications of Knowledge Discovery and Data Mining* (pp. 29–39).
- Géron, A. (2022). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.
  - **Course main text** (contract, Pustaka Utama [1]).
- James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). *An introduction to statistical learning: With applications in Python*. Springer. https://doi.org/10.1007/978-3-031-38747-0
  - **Course text** (contract [2]).
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: Data mining, inference, and prediction* (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-84858-7

## 10. Indonesian papers (local context; DOIs verified in Crossref)

- Laksono, E., Basuki, A., & Bachtiar, F. (2020). Optimization of K value in KNN algorithm for spam and ham email classification. *Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi), 4*(2), 377–383. https://doi.org/10.29207/resti.v4i2.1845
- Maslan, A., Rahman, A., Faruq, U., & Al-Jawr, R. R. A. (2025). Spam email classification optimization with NLP-based naïve Bayes on TF-IDF and SMOTE. *Jurnal Nasional Teknik Elektro dan Teknologi Informasi, 14*(4), 263–271. https://doi.org/10.22146/jnteti.v14i4.20931
  - → Reports 100% accuracy after SMOTE. Useful as a contrast: a near-perfect score is a warning sign of leakage (slide W07 p. 35).
  - Check the 4th author's surname (Al-Jawr or Al-Jawry) in the PDF.
- Firmansyah, F. A., Enri, U., & Maulana, I. (2025). Penerapan algoritma naive Bayes dengan chi-square untuk klasifikasi spam email berbasis kata dan frekuensi. *Jurnal Informatika dan Teknik Elektro Terapan, 13*(1). https://doi.org/10.23960/jitet.v13i1.5506
  - → Same pairing as ours (NB + χ²). Take the page numbers from the PDF.
- Ghani, M. A., & Sulaiman, H. (2023). Deteksi spam email dengan metode naive Bayes dan particle swarm optimization (PSO). *Infotek: Jurnal Informatika dan Teknologi, 6*(1), 11–20. https://doi.org/10.29408/jit.v6i1.7049
- Purnama, A., & Hamidin, D. (2025). Metode algoritma logistic regression dalam klasifikasi email spam. *Journal Software, Hardware and Information Technology, 5*(1), 39–47. https://doi.org/10.24252/shift.v5i1.159
- Sadid, S. M., Young, J. C., & Rusli, A. (2022). Spam filtering on user feedback via text classification using multinomial naïve Bayes and TF-IDF. *Ultimatics: Jurnal Teknik Informatika, 13*(2), 108–113. https://doi.org/10.31937/ti.v13i2.2149
- Fadli, M., Wijaya, V., Pribadi, M. R., & Widhiarso, W. (2023). Effect of TF-IDF extraction and application of SMOTE on model performance in detecting spam email. In *2023 10th International Conference on Electrical Engineering, Computer Science and Informatics (EECSI)* (pp. 637–641). IEEE. https://doi.org/10.1109/EECSI59885.2023.10295851

SINTA ranks were not verified individually. Check them on sinta.kemdiktisaintek.go.id if the lecturer requires SINTA-indexed sources.

---

## Corrections to the reference list in `Laporan_UTS_Group_1.md`

All 13 existing entries have correct authors, years, volumes and pages. The remaining issues are APA 7 completeness:

| Entry | Fix |
|---|---|
| All journal articles and books | Add the DOI (see above). |
| Jáñez-Martino et al. (2023) | Add the issue: *56*(2). |
| Friedman (2001) | Journal name is *The Annals of Statistics*. |
| Hastie et al. (2009) | Add the subtitle: *…: Data mining, inference, and prediction*. |
| McCallum & Nigam (1998) | Add "(AAAI Technical Report WS-98-05, pp. 41–48). AAAI Press." |
| Ke et al. (2017) | Cite as proceedings with URL; pages may be omitted. |
| Course texts | Géron (2022) and James et al. (2023) are the course's required texts and are not yet cited. Add them. |

## Not verified / do not use

- No source-corpus list exists on the dataset card. §2 is inference.
- **Dataset citation (group decision, 6 Nov 2026):** cite the Hugging Face repository `locuoco/the-biggest-spam-ham-phish-email-dataset-300000` under its repository name, accessed 9 Oct 2026. Do not cite the Kaggle original separately.
- No citable CEAS 2008 corpus paper exists.
- Which upstream preprocessing introduced the `escapenumber` placeholders cannot be confirmed. Describe it as an observed artifact; do not attribute it to a named corpus.
- Page numbers commonly cited for Ng & Jordan (841–848), Ke et al. (3146–3154) and Lundberg & Lee (4765–4774) are not on the official pages. Omit them.
- The CRISP-DM 1.0 guide is verified only through mirrors.
