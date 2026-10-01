import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Hiring Prediction System",
    page_icon="💼",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .hire-box {
        background-color: #123524;
        border: 1px solid #2ecc71;
    }

    .reject-box {
        background-color: #3a1717;
        border: 1px solid #ff4b4b;
    }

    .result-title {
        font-size: 30px;
        font-weight: 700;
    }

    .metric-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #1f2937;
        text-align: center;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 15px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "final_hiring_prediction_pipeline.pkl"
    )

    features = joblib.load(
        "model_features.pkl"
    )

    return model, features


model, model_features = load_model()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">💼 AI-Based Hiring Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine-learning based candidate screening demonstration'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "This is an educational/demo system. "
    "The prediction should not be used as an autonomous hiring decision."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("👤 Candidate Information")

name = st.sidebar.text_input(
    "Candidate Name",
    placeholder="Enter candidate name"
)

experience = st.sidebar.number_input(
    "Experience (Years)",
    min_value=0,
    max_value=10,
    value=2,
    step=1
)

salary = st.sidebar.number_input(
    "Salary Expectation ($)",
    min_value=40085,
    max_value=119901,
    value=60000,
    step=1000
)

projects = st.sidebar.number_input(
    "Projects Count",
    min_value=0,
    max_value=10,
    value=2,
    step=1
)

education = st.sidebar.selectbox(
    "Education",
    [
        "B.Sc",
        "MBA",
        "B.Tech",
        "PhD",
        "M.Tech"
    ]
)

job_role = st.sidebar.selectbox(
    "Job Role",
    [
        "AI Researcher",
        "Data Scientist",
        "Cybersecurity Analyst",
        "Software Engineer"
    ]
)

certification = st.sidebar.selectbox(
    "Certification",
    [
        "No Certification",
        "Google ML",
        "AWS Certified",
        "Deep Learning Specialization"
    ]
)

skills = st.sidebar.multiselect(
    "Skills",
    [
        "C++",
        "Cybersecurity",
        "Deep Learning",
        "Ethical Hacking",
        "Java",
        "Linux",
        "Machine Learning",
        "NLP",
        "Networking",
        "Python",
        "Pytorch",
        "React",
        "SQL",
        "TensorFlow"
    ]
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔍 Predict Hiring Decision",
    use_container_width=True
)


if predict_button:

    # ========================================================
    # CREATE CANDIDATE FEATURES
    # ========================================================

    candidate = {
        "Experience (Years)": experience,
        "Salary Expectation ($)": salary,
        "Projects Count": projects
    }


    # ========================================================
    # SKILLS
    # ========================================================

    all_skills = [
        "C++",
        "Cybersecurity",
        "Deep Learning",
        "Ethical Hacking",
        "Java",
        "Linux",
        "Machine Learning",
        "NLP",
        "Networking",
        "Python",
        "Pytorch",
        "React",
        "SQL",
        "TensorFlow"
    ]

    for skill in all_skills:

        candidate[skill] = (
            1 if skill in skills else 0
        )


    # ========================================================
    # EDUCATION
    # ========================================================

    education_values = [
        "B.Sc",
        "MBA",
        "B.Tech",
        "PhD",
        "M.Tech"
    ]

    for value in education_values:

        candidate[f"Education_{value}"] = (
            1 if education == value else 0
        )


    # ========================================================
    # JOB ROLE
    # ========================================================

    job_roles = [
        "AI Researcher",
        "Data Scientist",
        "Cybersecurity Analyst",
        "Software Engineer"
    ]

    for value in job_roles:

        candidate[f"Job Role_{value}"] = (
            1 if job_role == value else 0
        )


    # ========================================================
    # CERTIFICATIONS
    # ========================================================

    certifications = [
        "AWS Certified",
        "Deep Learning Specialization",
        "Google ML",
        "No Certification"
    ]

    for value in certifications:

        candidate[f"Certifications_{value}"] = (
            1 if certification == value else 0
        )


    # ========================================================
    # DATAFRAME
    # ========================================================

    candidate_df = pd.DataFrame([candidate])


    # Make sure the feature order is identical
    # to the training data.

    candidate_df = candidate_df.reindex(
        columns=model_features,
        fill_value=0
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict(candidate_df)[0]

    probabilities = model.predict_proba(candidate_df)[0]

    reject_probability = probabilities[0]

    hire_probability = probabilities[1]


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        '<div class="section-title">Prediction Result</div>',
        unsafe_allow_html=True
    )


    if prediction == 1:

        st.markdown(
            """
            <div class="result-box hire-box">
                <div class="result-title">
                    ✅ HIRE
                </div>
                <div>
                    The model predicts a Hire outcome.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="result-box reject-box">
                <div class="result-title">
                    ❌ REJECT
                </div>
                <div>
                    The model predicts a Reject outcome.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PROBABILITY METRICS
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Hire Probability",
            f"{hire_probability * 100:.2f}%"
        )


    with col2:

        st.metric(
            "Reject Probability",
            f"{reject_probability * 100:.2f}%"
        )


    # ========================================================
    # CANDIDATE SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">Candidate Summary</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        st.write(
            f"**Name:** {name if name else 'Not provided'}"
        )

        st.write(
            f"**Experience:** {experience} years"
        )


    with col2:

        st.write(
            f"**Education:** {education}"
        )

        st.write(
            f"**Job Role:** {job_role}"
        )


    with col3:

        st.write(
            f"**Projects:** {projects}"
        )

        st.write(
            f"**Salary:** ${salary:,}"
        )


    st.write(
        f"**Certification:** {certification}"
    )

    st.write(
        f"**Skills:** "
        f"{', '.join(skills) if skills else 'None selected'}"
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

with st.expander("📊 Model Information"):

    st.write(
        """
        **Model:** Logistic Regression

        **Preprocessing:** StandardScaler + feature encoding

        **Hyperparameter tuning:** GridSearchCV with 5-fold cross-validation

        **Best C:** 100

        **Solver:** lbfgs

        **Test Accuracy:** 99.5%

        **Test Precision:** 100%

        **Test Recall:** 99.38%

        **Test F1 Score:** 99.69%

        **5-Fold Cross-Validation F1:** 98.92%
        """
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "⚠️ This project is intended for educational and demonstration "
    "purposes. The model should not replace human judgment in real "
    "recruitment decisions."
)