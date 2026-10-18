An English question-answering assistant for crop-production questions. A fine-tuned Flan-T5-base generates short advisory answers, and an out-of-domain (OOD) gate refuses questions that are not about agriculture. The project compares the model against a TF-IDF retrieval baseline on a leakage-controlled test set, and documents where it fails, including unsafe dose and treatment errors.

	
Live app	https://agriculture-qsnandans-wausmntbpagefhsythxe63.streamlit.app/
Model and gate (Hugging Face Hub)	Toronga/agri-qa-flan-t5
Dataset	talhakk/agriculture-qa (25,410 pairs before cleaning)
Report	report/Agricultural_Advisory_QA_Report.pdf

Safety notice. The answers have not been checked by agronomists, and the advice is specific to South Asian practices, products and units. The model sometimes outputs wrong doses and wrong treatments. Do not use it for real pesticide or fertiliser decisions without expert review.

Main results

Clean test set (n = 699, evaluated once). Intervals are bootstrap 95% confidence intervals.

System	ROUGE-L [95% CI]	BERTScore F1	Dose-number F1
Random answer	0.104 [0.096, 0.112]	−0.038	0.106
TF-IDF retrieval baseline	0.311 [0.293, 0.329]	0.253	0.310
Flan-T5-base (final model)	0.370 [0.353, 0.387]	0.326	0.330
The model beats retrieval by +0.059 ROUGE-L (paired bootstrap CI [0.039, 0.078]), but dose-number F1 barely changes.
About a third of a random validation or test split has a near-duplicate question or an identical answer in training. On those rows the baseline scores ROUGE-L 0.544–0.551 versus about 0.3 on clean rows, so all reported scores use leakage-free subsets.
20.3% of generated answers are verbatim copies of a training answer.
The OOD gate reaches AUROC 0.996, rejects 99–100% of general-knowledge and health questions, and accepts 91.7% of in-domain validation questions (70–85% on naturally phrased ones).
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

Method in brief
Data. 25,410 pairs from talhakk/agriculture-qa, cleaned to 20,370 and then to 17,623 (duplicates, livestock and aquaculture keywords, administrative questions, weather records, corrupted text, non-informative answers).
Baseline. Return the answer of the most similar training question (TF-IDF cosine similarity).
Model. Flan-T5-base (248M parameters), input question: <text>, 128 input and output tokens. AdamW with learning rate 3e-4, weight decay 0.01, linear decay with 5% warmup, effective batch size 16, 5 epochs, seed 42. Final decoding: beam search with 4 beams and no repeated trigrams.
Experiments. Zero-shot (E1), learning rate (E2, E3), data size and epochs (E4a, E4), model size (E5), retrieval-augmented input (E6), decoding strategies (E7).
OOD gate. A question is accepted only if its stemmed TF-IDF similarity to the training questions and a logistic-regression classifier score are both above thresholds calibrated on in-domain validation data.
Known limitations
Single training seed per configuration, so small differences (E6, E7) are suggestive only.
ROUGE and BERTScore measure similarity, not factual correctness, and the rate of wrong treatments has not been measured by manual review.
Livestock, fish and poultry questions that use words outside the keyword list survive the cleaning, and the OOD gate cannot catch them because they look in-domain.
In the first cleaning notebook, two filter masks were applied to a re-indexed table, which deleted 64 unrelated rows (about 0.3% of the data). This is documented in Table 1 of the report and does not change the conclusions.
English only, text only.
References

Full reference list in the report. Key sources: Raffel et al. (2020) for T5, Chung et al. (2022) for Flan-T5, Lewis et al. (2020) for retrieval-augmented generation, and Lewis et al. (2021) and Lee et al. (2022) for train–test overlap and duplication.# agriculture-qsn_and_ans
