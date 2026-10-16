import joblib, torch, streamlit as st
from huggingface_hub import hf_hub_download
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "Toronga/agri-qa-flan-t5"   # Hub repo with the model and gate.joblib (must be public)
USE_BF16 = True                        # halves memory; set False for outputs identical to the evaluated fp32 model

st.set_page_config(page_title="Agriculture Q&A", page_icon="🌾")


@st.cache_resource(show_spinner="Loading model (the first load takes about a minute)...")
def load():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if USE_BF16:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
        except TypeError:                      # older transformers versions use torch_dtype
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    gate = joblib.load(hf_hub_download(MODEL_ID, "gate.joblib"))
    return tok, model.eval(), gate


tok, model, gate = load()


def respond(q, beams):
    """Returns (answer, domain-check text, nearest training question or '-')."""
    mode = gate["mode"]            # "sim" | "conf" | "both" | "clf" | "clf_sim"
    sims = cosine_similarity(gate["vectorizer"].transform([q]), gate["train_matrix"])[0]
    sim, near = float(sims.max()), gate["train_questions"][int(sims.argmax())]
    if mode in ("sim", "both", "clf_sim") and sim < gate["tau_sim"]:
        return gate["ood_message"], "Out of domain (low similarity to training questions: %.2f)" % sim, "-"
    p = None
    if mode in ("clf", "clf_sim"):
        p = float(gate["clf"].predict_proba(gate["clf_vec"].transform([q]))[0, 1])
        if p < gate["tau_clf"]:
            return gate["ood_message"], "Out of domain (domain classifier score: %.2f)" % p, "-"
    inp = tok("question: " + q, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=128, num_beams=max(2, int(beams)), no_repeat_ngram_size=3,
                             early_stopping=True, output_scores=True, return_dict_in_generate=True)
    conf = float(out.sequences_scores[0])
    if mode in ("conf", "both") and conf < gate["tau_conf"]:
        return gate["ood_message"], "Out of domain (low model confidence: %.2f)" % conf, "-"
    answer = tok.decode(out.sequences[0], skip_special_tokens=True)
    status = "In domain | similarity %.2f | confidence %.2f" % (sim, conf)
    if p is not None:
        status += " | domain classifier %.2f" % p
    return answer, status, near


st.title("🌾 Agriculture Q&A")
st.caption("Fine-tuned Flan-T5 with an out-of-domain check. Answers are model-generated and may be wrong. "
           "Always confirm chemicals and doses with a local agricultural extension officer.")

EXAMPLES = ["how to control aphids in sugarcane", "recommended fertilizer dose for mustard", "Who won the 2018 football world cup?"]
if "q" not in st.session_state:
    st.session_state.q = ""
for col, ex in zip(st.columns(len(EXAMPLES)), EXAMPLES):
    if col.button(ex):
        st.session_state.q = ex

question = st.text_area("Your question", key="q", height=80)
beams = st.slider("Beam width (higher = slower)", 2, 6, 4)

if st.button("Ask", type="primary"):
    if len(question.strip()) < 5:
        st.warning("Please type a question.")
    else:
        with st.spinner("Thinking..."):
            answer, status, near = respond(question.strip(), beams)
        st.subheader("Answer")
        st.write(answer)
        st.caption(status)
        st.caption("Most similar training question: " + near)
