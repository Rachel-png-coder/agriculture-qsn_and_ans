An English question-answering assistant for crop-production questions. A fine-tuned Flan-T5-base generates short advisory answers, and an out-of-domain (OOD) gate refuses questions that are not about agriculture. The project compares the model against a TF-IDF retrieval baseline on a leakage-controlled test set, and documents where it fails, including unsafe dose and treatment errors.

	
Live app	https://agriculture-qsnandans-wausmntbpagefhsythxe63.streamlit.app/

Model and gate (Hugging Face Hub)	Toronga/agri-qa-flan-t5
Dataset	talhakk/agriculture-qa (25,410 pairs before cleaning)
Report	report/Agricultural_Advisory_QA_Report.pdf

Safety notice. The answers have not been checked by agronomists, and the advice is specific to South Asian practices, products and units. The model sometimes outputs wrong doses and wrong treatments. Do not use it for real pesticide or fertiliser decisions without expert review.

Problem and motivation

There are far too few agricultural extension officers per farmer. A question-answering assistant could give first-line advice at scale: which fungicide, at what concentration, or how much fertiliser per unit area. Generative models suit this because answers are free-form advice, but they hallucinate, and in agriculture a plausible-sounding wrong dose is a safety problem.

Task. Given a short English question about crop production, return a short advisory answer, and refuse questions outside the domain.

Users. The long-term goal is an assistant any farmer can ask. This prototype is not safe for unsupervised use, so the realistic near-term users are extension officers and advisory services that review answers before they reach farmers.

Research questions.

Does a fine-tuned generative model beat a strong retrieval baseline on a highly templated advisory dataset once train–test leakage is removed?
Which training choices (learning rate, data size, epochs, model size, retrieval context, decoding) matter?
Can a lightweight gate stop the assistant from answering out-of-domain questions?
What do overlap metrics miss, particularly about doses and treatments?
Dataset

Source. talhakk/agriculture-qa on Hugging Face: 25,410 English question–answer pairs with South Asian terminology (e.g. bigha, local pesticide names). See the dataset page for its licence terms.

Cleaning (25,410 → 20,370 → 17,623). The first notebook removes exact duplicate pairs, livestock, veterinary, poultry and aquaculture questions (keyword match on the question), administrative questions (PM-Kisan, KCC, Aadhaar, pension, bank), raw weather records and corrupted text. The modelling notebook then strips appended phrases such as "in South Asia context" (13.7% of questions), removes 85 non-informative answers, and applies length and duplicate-question filters. Every step and its row count is in Table 1 of the report.

Split and leakage control. The 17,623 pairs are split 88/6/6 (seed 42) into 15,508 train, 1,057 validation and 1,058 test. About a third of validation (32.4%) and test (33.9%) rows have a near-duplicate training question (TF-IDF cosine similarity above 0.9) or an identical training answer. These are removed, leaving 715 clean validation questions (a fixed random subset of 300 is used to compare experiments) and 699 clean test questions.

Characteristics. Questions average 9.5 words and answers 18.4 words. About 70% of answers are in an advisory-log style ("suggested him to apply...") and 30% are free-form. The text is highly templated, which favours memorisation.

Limitations. Answers have not been verified by agronomists, and the advice is region-specific.

Methodology
Data. 25,410 pairs from talhakk/agriculture-qa, cleaned to 20,370 and then to 17,623 (duplicates, livestock and aquaculture keywords, administrative questions, weather records, corrupted text, non-informative answers).
Baseline. Return the answer of the most similar training question (TF-IDF cosine similarity).
Model. Flan-T5-base (248M parameters), input question: <text>, 128 input and output tokens. AdamW with learning rate 3e-4, weight decay 0.01, linear decay with 5% warmup, effective batch size 16, 5 epochs, seed 42. Final decoding: beam search with 4 beams and no repeated trigrams.
Experiments. Zero-shot (E1), learning rate (E2, E3), data size and epochs (E4a, E4), model size (E5), retrieval-augmented input (E6), decoding strategies (E7).
OOD gate. A question is accepted only if its stemmed TF-IDF similarity to the training questions and a logistic-regression classifier score are both above thresholds calibrated on in-domain validation data.
Experiments

Validation set: 300 clean questions, one training run (seed 42) per configuration. Times are on a Tesla T4. Full table with confidence intervals and BERTScore is in the report.

ID	Setting	ROUGE-L	Dose-number F1	Train time
E0b	TF-IDF retrieval baseline	0.292	0.329	–
E1a	Zero-shot Flan-T5-base	0.098	0.015	–
E2	LR 3e-5, 8k examples, 3 epochs	0.279	0.252	15.4 min
E3	LR 3e-4, 8k examples, 3 epochs	0.329	0.303	15.0 min
E4a	LR 3e-4, all data, 3 epochs	0.351	0.313	27.3 min
E4	LR 3e-4, all data, 5 epochs (final)	0.365	0.341	45.3 min
E5	Flan-T5-small, all data, 5 epochs	0.330	0.305	16.7 min
E6	Nearest-neighbour answer added to the input (8k, 3 epochs)	0.352	0.333	21.0 min
E7a	E4 with greedy decoding	0.354	0.366	–
E7b	E4 with beam 4, no n-gram block	0.364	0.347	–
E7c	E4 with beam 8	0.362	0.349	–

What the experiments show:

Fine-tuning is essential. Zero-shot Flan-T5 is at the random-answer floor.
The higher learning rate wins by 0.050, but both runs had the same 3-epoch budget, so 3e-5 is probably under-trained.
More data (+0.021) and more epochs (+0.014) help significantly, with diminishing returns.
Flan-T5-small trails base by 0.035 but trains in about a third of the time.
Retrieval-augmented input (E6) did not clearly help (+0.023 over its matched control E3, confidence interval touches zero).
Decoding changes are within noise of the final setting.
Results

Clean test set (n = 699, evaluated once). Intervals are bootstrap 95% confidence intervals.

System	ROUGE-L [95% CI]	BERTScore F1	Dose-number F1
Random answer	0.104 [0.096, 0.112]	−0.038	0.106
TF-IDF retrieval baseline	0.311 [0.293, 0.329]	0.253	0.310
Flan-T5-base (final model)	0.370 [0.353, 0.387]	0.326	0.330
The model beats retrieval by +0.059 ROUGE-L (paired bootstrap CI [0.039, 0.078]), but dose-number F1 barely changes.
About a third of a random validation or test split has a near-duplicate question or an identical answer in training. On those rows the baseline scores ROUGE-L 0.544–0.551 versus about 0.3 on clean rows, so all reported scores use leakage-free subsets.
20.3% of generated answers are verbatim copies of a training answer.
The OOD gate reaches AUROC 0.996, rejects 99–100% of general-knowledge and health questions, and accepts 91.7% of in-domain validation questions (70–85% on naturally phrased ones).
Error analysis

Automatic categories on the 699 test predictions (heuristic rules on ROUGE-L and dose numbers):

Category	Share
Good match (ROUGE-L ≥ 0.6)	18.3%
Missing dose	3.9%
Wrong dose	14.6%
Too short or incomplete	8.7%
Other low overlap (wrong treatment, wrong crop, or a valid paraphrase)	54.5%

The "other" category is a residual, not a measured error rate, and the share of wrong or unsafe treatments has not been measured by manual review. Typical failures:

One template reused across crops. "urea-19kg, ssp-45kg, mop-12kg per bigha" is the most repeated prediction (10 of 699), including for okra, whose reference dose is different.
Retrieval beats generation when the answer already exists. For a watermelon fertiliser question the model gives no dose, while TF-IDF retrieves a similar answer with numbers.
Unsafe out-of-scope answers. The model recommends the insecticide rogor for a fish ulcer and for bird flu, and the OOD gate cannot catch these because the questions look in-domain.
Overlap metrics miss chemical correctness. Malathion in place of dimethoate scores ROUGE-L 0.70.
Known limitations
Single training seed per configuration, so small differences (E6, E7) are suggestive only.
ROUGE and BERTScore measure similarity, not factual correctness, and the rate of wrong treatments has not been measured by manual review.
Livestock, fish and poultry questions that use words outside the keyword list survive the cleaning, and the OOD gate cannot catch them because they look in-domain.
In the first cleaning notebook, two filter masks were applied to a re-indexed table, which deleted 64 unrelated rows (about 0.3% of the data). This is documented in Table 1 of the report and does not change the conclusions.
English only, text only.
Repository structure

Adjust the names below to match your repository.

.
├── notebooks/
│   ├── 01_data_cleaning.ipynb            # 25,410 -> 20,370 rows, writes agriculture_qa_clean.csv
│   └── 02_modelling_and_evaluation.ipynb # 17,623 pairs, split, baseline, experiments E0-E7, error analysis, OOD gate
├── app.py                                # Streamlit app
├── requirements.txt
├── report/
│   └── Agricultural_Advisory_QA_Report.pdf
└── README.md
Setup

Requires Python 3.10+ and, for training, a GPU (all runs used a Tesla T4 on Google Colab).

bash
git clone https://github.com/Rachel-png-coder/agriculture-qsn_and_ans.git
cd agriculture-qsn_and_ans

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

The main dependencies are torch, transformers, datasets, scikit-learn, pandas, numpy, joblib, sentencepiece, streamlit, rouge-score and bert-score. The results in the report were produced with transformers 5.18.0 in fp32.

Reproducing the results

Run the notebooks in order (Google Colab with a T4 GPU is recommended).

Data cleaning (notebooks/01_data_cleaning.ipynb). Downloads talhakk/agriculture-qa from Hugging Face, applies the filters listed in Table 1 of the report, and saves agriculture_qa_clean.csv (20,370 rows).
Modelling and evaluation (notebooks/02_modelling_and_evaluation.ipynb). Loads agriculture_qa_clean.csv and runs, in order:
question and answer cleaning (17,623 pairs), then the 88/6/6 split (seed 42);
the leakage analysis and the clean validation and test sets;
the TF-IDF baseline and the Flan-T5 experiments E0–E7;
the final test evaluation, error analysis and the OOD gate.

Approximate training times on a T4: E2 and E3 about 15 min, E4a 27 min, E4 (final model) 45 min, E5 17 min, E6 21 min.

All scores use ROUGE-L with stemming, BERTScore F1 (roberta-large, rescaled with baseline), and a dose-number F1 that compares the numbers in the prediction and the reference. Model differences are tested with a paired bootstrap on per-example ROUGE-L.

Running the app locally
bash
streamlit run app.py

The app loads the fine-tuned model and the gate from the Hugging Face Hub (Toronga/agri-qa-flan-t5). A question first passes the OOD gate. Rejected questions return a refusal. Accepted questions are answered by the model with beam search, and the app shows the gate decision, the similarity and classifier scores, and the most similar training question.

References

Full reference list in the report. Key sources: Raffel et al. (2020) for T5, Chung et al. (2022) for Flan-T5, Lewis et al. (2020) for retrieval-augmented generation, and Lewis et al. (2021) and Lee et al. (2022) for train–test overlap and duplication.
