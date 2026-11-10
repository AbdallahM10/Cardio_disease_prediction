"""Run with: .venv/Scripts/python -m streamlit run interface/app.py"""
import pandas as pd
import streamlit as st

from predictor import load_model, predict

st.set_page_config(page_title="Cardio | Classification demo", layout="wide")
st.markdown("""<style>
:root{--ink:#18334a;--muted:#526577;--accent:#176b78;--line:#dce5e8;--paper:#f6f8fa}
.stApp{background:var(--paper);color:var(--ink)}
.block-container{max-width:1180px;padding-top:2.6rem;padding-bottom:2rem}
header[data-testid="stHeader"]{background:transparent}
h1,h2,h3{color:var(--ink);letter-spacing:-.04em}h1{font-size:2.7rem!important;line-height:1.15!important}
.eyebrow{color:var(--accent);font-size:.75rem;font-weight:700;letter-spacing:.17em;margin-bottom:.6rem}
.intro{font-size:1.05rem;color:var(--muted);max-width:660px;line-height:1.6}
.meta{border-top:1px solid var(--line);padding-top:12px;margin-top:20px;font-size:.85rem;color:var(--muted)}
[data-testid="stForm"]{background:white;border:1px solid var(--line);border-radius:12px;padding:1.5rem}
[data-testid="stNumberInput"] input{min-height:44px;font-size:16px}
[data-testid="stSelectbox"] [role="combobox"]{min-height:44px;font-size:16px}
button[kind="primary"]{background:var(--accent)!important;border:1px solid var(--accent)!important;min-height:48px;font-weight:600}
button:focus-visible,input:focus-visible{outline:3px solid #176b78!important;outline-offset:2px}
.result-box{background:white;border:1px solid var(--line);border-top:4px solid var(--accent);border-radius:10px;padding:1.6rem;margin-bottom:1rem}
.result-box.flagged{border-top-color:#a65a20}.result-label{font-size:.72rem;font-weight:700;letter-spacing:.12em;color:var(--muted)}
.result-title{font-size:1.8rem;line-height:1.25;font-weight:700;margin:1rem 0 .6rem;letter-spacing:-.04em}
.result-copy{font-size:.95rem;line-height:1.6;color:var(--muted);margin:0}
.idle-number{font-size:3.2rem;line-height:1.2;color:#8a9aa6;font-weight:300;margin:1.4rem 0}
.notice{font-size:.85rem;color:var(--muted);border-top:1px solid var(--line);padding-top:14px;line-height:1.6}
@media(max-width:640px){.block-container{padding:1.5rem 1rem}h1{font-size:2.1rem!important}[data-testid="stForm"]{padding:1rem}}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the saved classifier...")
def get_model():
    return load_model()


st.markdown('<div class="eyebrow">MACHINE LEARNING / INTERACTIVE DEMONSTRATION</div>', unsafe_allow_html=True)
st.title("Cardiovascular disease classifier")
st.markdown('<p class="intro">Enter examination measurements to see how the project’s trained model classifies one record.</p>', unsafe_allow_html=True)
st.caption("Prepared by Abdallah Mestrah · Supervisor: Dr. Hussein Kanaan")
try:
    bundle = get_model()
except (OSError, ValueError, RuntimeError, ImportError) as error:
    st.error(f"The saved model could not be loaded: {error}")
    st.stop()

left, right = st.columns([1.55, 1], gap="large")
with left:
    with st.form("patient_inputs"):
        st.subheader("01 / Examination inputs")
        st.caption("All 11 inputs are required. Example values are prefilled; replace them before predicting.")
        a, b, c = st.columns(3)
        age = a.number_input("Age (years)", min_value=18.0, max_value=120.0, value=55.0, step=0.1,
                             help="Converted to days using 365.25 days per year; the model was trained on age in days.")
        gender = b.selectbox("Gender code", [1, 2], format_func=lambda x: f"Code {x}",
                             help="Use the dataset's code. The project did not verify which code represents each gender.")
        height = c.number_input("Height (cm)", min_value=1.0, max_value=300.0, value=165.0, step=1.0)
        a, b, c = st.columns(3)
        weight = a.number_input("Weight (kg)", min_value=1.0, max_value=600.0, value=70.0, step=0.5)
        systolic = b.number_input("Systolic (mmHg)", min_value=1, max_value=500, value=120, step=1,
                                 help="The higher, first blood pressure reading.")
        diastolic = c.number_input("Diastolic (mmHg)", min_value=1, max_value=400, value=80, step=1,
                                  help="The lower, second blood pressure reading.")
        st.divider()
        st.subheader("02 / Lab categories & lifestyle")
        category_names = {1: "Normal (1)", 2: "Above normal (2)", 3: "Well above normal (3)"}
        a, b = st.columns(2)
        cholesterol = a.selectbox("Cholesterol category", [1, 2, 3], format_func=category_names.get)
        glucose = b.selectbox("Glucose category", [1, 2, 3], format_func=category_names.get)
        a, b, c = st.columns(3)
        smoking = a.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x else "No")
        alcohol = b.selectbox("Alcohol intake", [0, 1], format_func=lambda x: "Yes" if x else "No")
        active = c.selectbox("Physically active", [1, 0], format_func=lambda x: "Yes" if x else "No")
        st.caption("Inputs stay in this local app session and are not saved to a patient file.")
        submitted = st.form_submit_button("Predict disease presence", type="primary", width="stretch")
    if submitted:
        values = {"age": round(age * 365.25), "gender": gender, "height": height, "weight": weight,
                  "ap_hi": systolic, "ap_lo": diastolic, "cholesterol": cholesterol, "gluc": glucose,
                  "smoke": smoking, "alco": alcohol, "active": active}
        try:
            st.session_state.result = predict(bundle, values)
            st.session_state.pop("prediction_error", None)
        except ValueError as error:
            st.session_state.pop("result", None)
            st.session_state.prediction_error = str(error)

with right:
    st.subheader("Prediction")
    if "prediction_error" in st.session_state:
        st.error(st.session_state.prediction_error)
        st.info("Correct the inputs, then click Predict again.")
    elif "result" not in st.session_state:
        st.markdown('''<div class="result-box"><div class="result-label">AWAITING INPUT</div>
<div class="idle-number">—</div><div class="result-title">Ready when you are.</div>
<p class="result-copy">Complete the form and select Predict. The saved model is already loaded; no training is needed.</p></div>''', unsafe_allow_html=True)
    else:
        result = st.session_state.result
        title = "Disease predicted" if result["disease"] else "No disease predicted"
        text = "The model assigned this record to class 1." if result["disease"] else "The model assigned this record to class 0."
        state_class = " flagged" if result["disease"] else ""
        st.markdown(f'''<div class="result-box{state_class}"><div class="result-label">LAST SUBMITTED PREDICTION</div>
<div class="result-title">{title}</div><p class="result-copy">{text}</p></div>''', unsafe_allow_html=True)
        st.metric("Disease model score", f'{result["probability"]:.1%}')
        st.progress(result["probability"])
        st.caption("Disease is predicted when the score is at least 50%. This uncalibrated model score is not a clinical risk estimate.")
        st.caption("After changing any field, select Predict again to update the result.")
        if result["capped"]:
            st.info("Some measurements were capped using the saved IQR boundaries before prediction.")
        with st.expander("Review submitted model inputs"):
            st.dataframe(result["processed"].T.rename(columns={0: "Value used"}), width="stretch")
            if result["capped"]:
                st.dataframe(pd.DataFrame(result["capped"]), hide_index=True, width="stretch")
    st.markdown('<div class="notice"><b>Project demonstration, not a diagnosis.</b><br>A prediction does not confirm or rule out disease. The model was evaluated on one supplied dataset and has no independent clinical validation.</div>', unsafe_allow_html=True)
    with st.expander("About this model"):
        st.write("Histogram gradient boosting · saved IQR clipping · decision threshold 0.50.")
        st.write("Recorded test accuracy: 73.85%. Disease recall: 69.66%.")
        st.caption("These are dataset results, not a guarantee for an individual prediction. Adult input limits are form checks, not validation for all adult ages.")

st.markdown('<div class="meta">LOCAL DEMO · 11 INPUTS · SAVED MODEL · NO RETRAINING</div>', unsafe_allow_html=True)
