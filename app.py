
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Hypertension Classification System",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "hypertension_model_v2.pkl"
DATASET_PATH = BASE_DIR / "dataset.csv"

FEATURE_COLS = [
    "bmi",
    "family_history",
    "smoking_status",
    "stress_score",
    "bp_history"
]


# =====================================================
# 2. GLOBAL STYLING
# =====================================================

st.markdown("""
<style>

html, body, .stApp {
    background-color: #111111;
    color: #F5F5F5;
    font-family: 'Segoe UI', sans-serif;
}

h1, h2, h3, h4, h5, h6, label,
.stText, .markdown-text-container {
    color: #F5F5F5 !important;
}

.stButton > button {
    background-color: #1f77b4;
    color: white;
    border-radius: 6px;
    padding: 0.6em 1.2em;
    border: none;
}

.stDownloadButton > button {
    background-color: #117A65;
    color: white;
    border-radius: 6px;
}

.stMetric {
    background-color: #1c1c1c;
    padding: 12px;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD MACHINE LEARNING MODEL
# =====================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    return joblib.load(MODEL_PATH)


# =====================================================
# 4. LOAD DATASET
# =====================================================

@st.cache_data
def load_data():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATASET_PATH.name}"
        )

    data = pd.read_csv(DATASET_PATH)

    data.columns = (
        data.columns
        .str.lower()
        .str.strip()
    )

    return data


# =====================================================
# 5. NORMALIZE BINARY LABELS
# =====================================================

def normalize_binary_labels(values):

    labels = pd.Series(values)

    labels = (
        labels.astype("string")
        .str.strip()
        .str.lower()
    )

    mapping = {
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        "1.0": 1,
        "0.0": 0,
        "true": 1,
        "false": 0
    }

    return labels.map(mapping)


def normalize_single_label(value):

    result = normalize_binary_labels([value])

    return result.iloc[0]


# =====================================================
# 6. LOAD MODEL AND DATASET
# =====================================================

try:

    model = load_model()
    df = load_data()

    st.sidebar.success(
        "Model and dataset loaded successfully"
    )

except Exception as error:

    st.error(
        f"Application loading error: {error}"
    )

    st.stop()


# =====================================================
# 7. CHECK REQUIRED DATASET COLUMNS
# =====================================================

required_columns = FEATURE_COLS + ["has_hypertension"]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        f"Your dataset is missing these columns: {missing_columns}"
    )

    st.stop()


# =====================================================
# 8. BMI CALCULATOR
# =====================================================

st.sidebar.header("⚖️ BMI Calculator")

st.sidebar.markdown(
    "_Don't know your BMI? Calculate it here:_"
)

with st.sidebar.form(key="bmi_form"):

    weight = st.number_input(
        "Weight (kg)",
        min_value=10.0,
        value=65.0,
        step=0.5,
        format="%.1f"
    )

    height = st.number_input(
        "Height (meters)",
        min_value=0.5,
        value=1.70,
        step=0.01,
        format="%.2f"
    )

    calc_bmi_button = st.form_submit_button(
        "📏 Calculate BMI"
    )


calculated_bmi = 25.0

if calc_bmi_button:

    calculated_bmi = round(
        weight / (height ** 2),
        2
    )

    if 10 <= calculated_bmi <= 60:

        st.sidebar.success(
            f"Your BMI is: {calculated_bmi}"
        )

    else:

        st.sidebar.warning(
            f"Calculated BMI: {calculated_bmi}. "
            "Please verify your measurements."
        )

        calculated_bmi = 25.0


st.sidebar.markdown("---")

st.sidebar.markdown("### 📚 What is BMI?")

st.sidebar.markdown("""
**Body Mass Index (BMI)** is calculated using weight
and height.

- ⚠️ Underweight: BMI < 18.5
- ✅ Normal weight: 18.5–24.9
- ⚠️ Overweight: 25–29.9
- 🔶 Obesity Class I: 30–34.9
- 🔴 Obesity Class II: 35–39.9
- 🔴 Obesity Class III: 40+
""")

st.sidebar.caption(
    "BMI is a screening measure, not a diagnosis."
)


# =====================================================
# 9. APPLICATION HEADER
# =====================================================

st.markdown("""
<h1 style='text-align:center;'>
🩺 Hypertension Classification System
</h1>

<h5 style='text-align:center;color:#B0BEC5;'>
Powered by XGBoost | Binary Classification | Built with Streamlit
</h5>

<hr style='border-top:1px solid #555;'>
""", unsafe_allow_html=True)


st.markdown("""
This system uses machine learning to classify patients
into two categories based on their recorded information.

- **Class 0:** Not Hypertensive
- **Class 1:** Hypertensive
""")


# =====================================================
# 10. PATIENT INPUT FEATURES
# =====================================================

st.markdown("### 🔍 Enter Patient Information")

col1, col2 = st.columns(2)

with col1:

    bmi = st.number_input(
        "💪 BMI (Body Mass Index)",
        min_value=10.0,
        max_value=60.0,
        value=float(max(10, min(calculated_bmi, 60))),
        step=0.1
    )

    stress_score = st.slider(
        "😖 Stress Score",
        min_value=0,
        max_value=10,
        value=5
    )


with col2:

    family_history = st.selectbox(
        "👪 Family History of Hypertension?",
        ["yes", "no"]
    )

    smoking_status = st.selectbox(
        "🚬 Smoking Status",
        ["Never", "Former", "Current"]
    )


bp_history = st.selectbox(
    "💓 Blood Pressure History",
    ["Normal", "Elevated", "Stage 1", "Stage 2"]
)


# =====================================================
# 11. CREATE MODEL INPUT
# =====================================================

input_df = pd.DataFrame({

    "bmi": [bmi],

    "family_history": [
        family_history.lower()
    ],

    "smoking_status": [
        smoking_status
    ],

    "stress_score": [
        stress_score
    ],

    "bp_history": [
        bp_history
    ]

}, columns=FEATURE_COLS)


# =====================================================
# 12. HYPERTENSION BINARY CLASSIFICATION
# =====================================================

st.markdown("---")

st.markdown("### 📊 Patient Classification")

st.markdown("""
The model predicts one of two possible classifications:

- **Class 0:** Not Hypertensive
- **Class 1:** Hypertensive
""")

if st.button(
    "🔍 Classify Patient",
    use_container_width=True
):

    try:

        prediction = model.predict(input_df)

        prediction = np.asarray(prediction).reshape(-1)[0]

        predicted_label = normalize_single_label(
            prediction
        )

        if pd.isna(predicted_label):

            st.error(
                "The model returned an unsupported classification."
            )

        else:

            predicted_label = int(predicted_label)

            st.markdown("---")

            st.markdown("### Classification Result")

            if predicted_label == 1:

                st.error(
                    "⚠️ HYPERTENSIVE"
                )

                st.markdown("""
                **Predicted Classification: Class 1**

                The machine-learning model has classified
                this patient into the hypertension-positive
                category.

                The patient should receive appropriate clinical
                assessment and blood pressure measurement
                by a qualified healthcare professional.
                """)

            else:

                st.success(
                    "✅ NOT HYPERTENSIVE"
                )

                st.markdown("""
                **Predicted Classification: Class 0**

                The machine-learning model has classified
                this patient into the hypertension-negative
                category.

                Continue appropriate blood pressure monitoring.
                A negative classification does not rule out
                hypertension.
                """)

            st.caption(
                "This is an experimental machine-learning "
                "classification, not a clinical diagnosis."
            )

    except Exception as error:

        st.error(
            f"Classification error: {error}"
        )


# =====================================================
# 13. MODEL PERFORMANCE EVALUATION
# =====================================================

st.markdown("---")

st.subheader(
    "📊 Classification Model Performance"
)


try:

    # Prepare dataset features
    X = df[FEATURE_COLS].copy()

    # Normalize target labels
    y = normalize_binary_labels(
        df["has_hypertension"]
    )

    # Remove invalid target labels
    valid_target = y.notna()

    X = X.loc[valid_target].copy()

    y = y.loc[valid_target].astype(int)

    # Remove missing feature records
    valid_features = X.notna().all(axis=1)

    X = X.loc[valid_features]

    y = y.loc[valid_features]

    # Validate dataset size
    if len(y) < 5:

        raise ValueError(
            "Not enough valid records for evaluation."
        )

    # Ensure both classes exist
    if y.nunique() != 2:

        raise ValueError(
            "The dataset must contain both classes: "
            "0 (Not Hypertensive) and 1 (Hypertensive)."
        )

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )

    # Generate predictions
    y_pred_raw = model.predict(X_test)

    # Normalize predictions
    y_pred = normalize_binary_labels(
        np.asarray(y_pred_raw).reshape(-1)
    )

    # Check prediction validity
    if y_pred.isna().any():

        raise ValueError(
            "Model predictions contain invalid labels."
        )

    y_pred = y_pred.astype(int)

    # Ensure lengths match
    if len(y_test) != len(y_pred):

        raise ValueError(
            "Actual and predicted values have different lengths."
        )

    # Convert to NumPy arrays
    y_true_array = y_test.to_numpy(dtype=int)

    y_pred_array = y_pred.to_numpy(dtype=int)

    # Calculate metrics
    accuracy = accuracy_score(
        y_true_array,
        y_pred_array
    )

    precision = precision_score(
        y_true_array,
        y_pred_array,
        zero_division=0
    )

    recall = recall_score(
        y_true_array,
        y_pred_array,
        zero_division=0
    )

    f1 = f1_score(
        y_true_array,
        y_pred_array,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_true_array,
        y_pred_array,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # Specificity
    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    # Display metrics
    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Recall / Sensitivity",
            f"{recall * 100:.2f}%"
        )

        st.metric(
            "F1-Score",
            f"{f1 * 100:.2f}%"
        )

    st.metric(
        "Specificity",
        f"{specificity * 100:.2f}%"
    )

    # Confusion matrix summary
    st.caption(
        f"Evaluation records: {len(y_test)} | "
        f"True negatives: {tn} | "
        f"False positives: {fp} | "
        f"False negatives: {fn} | "
        f"True positives: {tp}"
    )

    # Display confusion matrix
    st.markdown("### Confusion Matrix")

    confusion_df = pd.DataFrame(
        cm,

        index=[
            "Actual: Not Hypertensive",
            "Actual: Hypertensive"
        ],

        columns=[
            "Predicted: Not Hypertensive",
            "Predicted: Hypertensive"
        ]
    )

    st.dataframe(
        confusion_df,
        use_container_width=True
    )

    # Classification report
    with st.expander(
        "View Detailed Classification Report"
    ):

        report = classification_report(
            y_true_array,
            y_pred_array,
            labels=[0, 1],
            target_names=[
                "Not Hypertensive",
                "Hypertensive"
            ],
            zero_division=0,
            output_dict=True
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df,
            use_container_width=True
        )

    # Download results
    st.markdown("### 📥 Download Evaluation Results")

    results_df = X_test.copy()

    results_df["Actual Classification"] = y_true_array

    results_df["Predicted Classification"] = y_pred_array

    results_df["Actual Label"] = results_df[
        "Actual Classification"
    ].map({
        0: "Not Hypertensive",
        1: "Hypertensive"
    })

    results_df["Predicted Label"] = results_df[
        "Predicted Classification"
    ].map({
        0: "Not Hypertensive",
        1: "Hypertensive"
    })

    if st.checkbox("📂 Show Test Predictions"):

        st.dataframe(
            results_df.head(20),
            use_container_width=True
        )

    csv = results_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        label="📥 Download Classification Results",

        data=csv,

        file_name="hypertension_classification_results.csv",

        mime="text/csv"

    )

    st.info(
        "Evaluation uses a newly created random split of the "
        "available dataset. Confirm that these records were "
        "not used to train the saved model before interpreting "
        "these metrics as independent test performance."
    )

except Exception as error:

    st.error(
        f"Model evaluation error: {error}"
    )


# =====================================================
# 14. FOOTER
# =====================================================

st.markdown("---")

st.markdown("""
<div style="
    font-size:13px;
    color:#999999;
    text-align:center;
">

<strong>Hypertension Classification System</strong>

<br><br>

Developed by <strong>Peter Mukuku</strong>

<br>

Machine Learning | XGBoost | Streamlit

<br><br>

Experimental research application.
Not intended to replace clinical diagnosis.

</div>
""", unsafe_allow_html=True)
