
import pandas as pd
import numpy as np
import joblib
import warnings

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

import plotly.graph_objects as go
from plotly.subplots import make_subplots


warnings.filterwarnings("ignore")


# ==================================================
# 1. LOAD DATASET
# ==================================================

print("\nLoading hypertension dataset...")

df = pd.read_csv("dataset.csv")

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)


# ==================================================
# 2. SELECT THE FIVE MODEL FEATURES
# ==================================================

features = [
    "BMI",
    "Family_History",
    "Smoking_Status",
    "Stress_Score",
    "BP_History"
]

target = "Has_Hypertension"

required_columns = features + [target]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing dataset columns: {missing}"
    )

df = df[required_columns].copy()

# Remove records without a target
df = df.dropna(subset=[target])

print("\nSelected model features:")
for feature in features:
    print("-", feature)


# ==================================================
# 3. CONVERT TARGET TO BINARY VALUES
# ==================================================

df[target] = (
    df[target]
    .astype(str)
    .str.strip()
    .str.lower()
)

target_mapping = {
    "no": 0,
    "yes": 1
}

unknown_values = sorted(
    set(df[target].unique()) - set(target_mapping.keys())
)

if unknown_values:
    raise ValueError(
        f"Unexpected target values: {unknown_values}. "
        "Expected Yes and No."
    )

df[target] = df[target].map(target_mapping).astype(int)

X = df[features]
y = df[target]

print("\nTarget distribution:")

print(
    y.value_counts()
    .rename(index={
        0: "Not Hypertensive",
        1: "Hypertensive"
    })
)

if y.nunique() != 2:
    raise ValueError(
        "The dataset must contain both target classes."
    )


# ==================================================
# 4. DEFINE DATA PREPROCESSING
# ==================================================

numeric_features = [
    "BMI",
    "Stress_Score"
]

categorical_features = [
    "Family_History",
    "Smoking_Status",
    "BP_History"
]

numeric_transformer = Pipeline(steps=[
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

categorical_transformer = Pipeline(steps=[
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ==================================================
# 5. TRAINING AND TESTING SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDATA SPLITTING")
print("-" * 55)

print("Total records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print(
    "Training hypertensive cases:",
    int(y_train.sum())
)

print(
    "Testing hypertensive cases:",
    int(y_test.sum())
)


# ==================================================
# 6. DEFINE FIVE MACHINE LEARNING ALGORITHMS
# ==================================================

models = {

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=1
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "SVM": SVC(
        probability=True,
        class_weight="balanced",
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ==================================================
# 7. FIVE-FOLD CROSS-VALIDATION
# ==================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

results = []
trained_models = {}

print("\n")
print("=" * 65)
print("STARTING MACHINE LEARNING MODEL COMPARISON")
print("=" * 65)

for name, classifier in models.items():

    print(f"\nEvaluating {name}...")

    # Create a complete pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # Five-fold cross-validation on training data
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    # Train on the full training split
    pipeline.fit(
        X_train,
        y_train
    )

    # Test on the held-out test dataset
    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    # Calculate performance metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    sensitivity = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    # Store all results
    results.append({

        "Algorithm": name,

        "CV Accuracy": cv_results[
            "test_accuracy"
        ].mean(),

        "CV Precision": cv_results[
            "test_precision"
        ].mean(),

        "CV Recall": cv_results[
            "test_recall"
        ].mean(),

        "CV F1": cv_results[
            "test_f1"
        ].mean(),

        "CV ROC-AUC": cv_results[
            "test_roc_auc"
        ].mean(),

        "CV ROC-AUC SD": cv_results[
            "test_roc_auc"
        ].std(),

        "Test Accuracy": accuracy,

        "Precision": precision,

        "Sensitivity (Recall)": sensitivity,

        "Specificity": specificity,

        "F1 Score": f1,

        "ROC-AUC": auc,

        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

    trained_models[name] = pipeline

    print(
        f"{name} completed successfully."
    )


# ==================================================
# 8. CREATE COMPARISON RESULTS TABLE
# ==================================================

results_df = pd.DataFrame(results)

# Rank models by cross-validation ROC-AUC
results_df = results_df.sort_values(
    by="CV ROC-AUC",
    ascending=False
).reset_index(drop=True)

results_df.insert(
    0,
    "Rank",
    range(1, len(results_df) + 1)
)

# Save results
results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

print("\n")
print("=" * 80)
print("HEALTH DATA ANALYTICS MODEL COMPARISON")
print("=" * 80)

print(
    results_df.round(4).to_string(index=False)
)


# ==================================================
# 9. IDENTIFY THE BEST MODEL AND EXPLAIN WHY
# ==================================================

best_row = results_df.iloc[0]

best_model_name = best_row["Algorithm"]

best_model = trained_models[best_model_name]

# Second-best model
second_best = results_df.iloc[1]

# Average cross-validation ROC-AUC
average_cv_auc = results_df["CV ROC-AUC"].mean()

# Difference between the top two models
auc_difference = (
    best_row["CV ROC-AUC"]
    - second_best["CV ROC-AUC"]
)

# Collect reasons for selecting the model
reasons = []

reasons.append(
    f"{best_model_name} achieved the highest "
    f"five-fold cross-validation ROC-AUC "
    f"of {best_row['CV ROC-AUC']:.3f}."
)

reasons.append(
    f"It ranked above {second_best['Algorithm']}, "
    f"which achieved a cross-validation ROC-AUC "
    f"of {second_best['CV ROC-AUC']:.3f}."
)

if best_row["CV ROC-AUC"] >= average_cv_auc:

    reasons.append(
        "Its cross-validation ROC-AUC was above "
        "the average performance of the five algorithms."
    )

reasons.append(
    f"Its held-out test accuracy was "
    f"{best_row['Test Accuracy']:.1%}."
)

reasons.append(
    f"It achieved a sensitivity of "
    f"{best_row['Sensitivity (Recall)']:.1%}, "
    f"identifying this proportion of actual "
    f"hypertensive cases in the test dataset."
)

reasons.append(
    f"Its specificity was "
    f"{best_row['Specificity']:.1%}, "
    f"indicating how well it identified "
    f"non-hypertensive cases."
)

reasons.append(
    f"Its F1-score was {best_row['F1 Score']:.3f}, "
    f"summarising the balance between precision "
    f"and sensitivity."
)


# Print the winning model
print("\n")
print("=" * 70)
print("BEST MACHINE LEARNING ALGORITHM")
print("=" * 70)

print("\nSELECTED ALGORITHM:", best_model_name)

print("\nWHY WAS THIS MODEL SELECTED?")

for i, reason in enumerate(reasons, start=1):

    print(f"\n{i}. {reason}")


# Display winning model's performance
print("\n")
print("=" * 70)
print("WINNING MODEL PERFORMANCE")
print("=" * 70)

performance_metrics = [
    ("Cross-validation ROC-AUC", "CV ROC-AUC"),
    ("Cross-validation Recall", "CV Recall"),
    ("Test Accuracy", "Test Accuracy"),
    ("Test Precision", "Precision"),
    ("Test Sensitivity", "Sensitivity (Recall)"),
    ("Test Specificity", "Specificity"),
    ("Test F1 Score", "F1 Score"),
    ("Test ROC-AUC", "ROC-AUC")
]

for label, column in performance_metrics:

    print(
        f"{label:<30}: "
        f"{best_row[column]:.4f}"
    )


# Classification report
best_predictions = best_model.predict(X_test)

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        best_predictions,
        target_names=[
            "Not Hypertensive",
            "Hypertensive"
        ],
        zero_division=0
    )
)


# ==================================================
# 10. SAVE BEST MODEL EXPLANATION
# ==================================================

with open(
    "best_model_explanation.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "BEST HYPERTENSION PREDICTION MODEL\n"
    )

    file.write("=" * 65 + "\n\n")

    file.write(
        f"Selected Algorithm: {best_model_name}\n\n"
    )

    file.write("REASONS FOR SELECTION\n\n")

    for i, reason in enumerate(reasons, start=1):

        file.write(
            f"{i}. {reason}\n\n"
        )

    file.write("\nPERFORMANCE RESULTS\n")

    for label, column in performance_metrics:

        file.write(
            f"{label}: {best_row[column]:.4f}\n"
        )

    file.write("\nCONFUSION MATRIX\n")

    file.write(
        f"True Negatives: {int(best_row['TN'])}\n"
    )

    file.write(
        f"False Positives: {int(best_row['FP'])}\n"
    )

    file.write(
        f"False Negatives: {int(best_row['FN'])}\n"
    )

    file.write(
        f"True Positives: {int(best_row['TP'])}\n"
    )

print(
    "\nBest model explanation saved."
)


# ==================================================
# 11. SAVE ALL TRAINED MODELS
# ==================================================

print("\nSaving trained models...")

for name, model in trained_models.items():

    filename = (
        name.lower()
        .replace(" ", "_")
        + "_model.pkl"
    )

    joblib.dump(
        model,
        filename
    )

    print("Saved:", filename)


# Save the winning model separately
joblib.dump(
    best_model,
    "best_hypertension_model.pkl"
)

print(
    "\nBest model saved as "
    "best_hypertension_model.pkl"
)


# ==================================================
# 12. CREATE VISUAL COMPARISON DASHBOARD
# ==================================================

print("\nCreating visual comparison dashboard...")

dashboard_df = results_df.copy()

algorithms = dashboard_df["Algorithm"].tolist()

# Identify winning model
best_algorithm = best_model_name

best_row = dashboard_df[
    dashboard_df["Algorithm"] == best_algorithm
].iloc[0]


# --------------------------------------------------
# GENERATE WINNER EXPLANATION FOR DASHBOARD
# --------------------------------------------------

best_reason = (
    f"{best_algorithm} was selected because it achieved "
    f"the highest mean five-fold cross-validation ROC-AUC "
    f"of {best_row['CV ROC-AUC']:.3f}. "
    f"Its test ROC-AUC was {best_row['ROC-AUC']:.3f}, "
    f"sensitivity was "
    f"{best_row['Sensitivity (Recall)']:.1%}, "
    f"specificity was {best_row['Specificity']:.1%}, "
    f"and F1-score was {best_row['F1 Score']:.3f}."
)


# --------------------------------------------------
# DASHBOARD LAYOUT
# --------------------------------------------------

fig = make_subplots(

    rows=3,

    cols=1,

    row_heights=[
        0.40,
        0.28,
        0.32
    ],

    vertical_spacing=0.12,

    specs=[
        [{"type": "xy"}],
        [{"type": "xy"}],
        [{"type": "table"}]
    ],

    subplot_titles=[
        "Performance Metrics Comparison",
        "Test ROC-AUC vs Cross-Validation ROC-AUC",
        "Detailed Model Comparison Results"
    ]
)


# ==================================================
# CHART 1: PERFORMANCE METRICS
# ==================================================

metrics = [
    "Test Accuracy",
    "Precision",
    "Sensitivity (Recall)",
    "Specificity",
    "F1 Score"
]

for metric in metrics:

    fig.add_trace(

        go.Bar(

            name=metric,

            x=algorithms,

            y=dashboard_df[metric],

            text=[
                f"{value:.1%}"
                for value in dashboard_df[metric]
            ],

            textposition="outside",

            hovertemplate=(
                "<b>%{x}</b><br>"
                + metric
                + ": %{y:.2%}<extra></extra>"
            )

        ),

        row=1,

        col=1

    )


fig.update_yaxes(

    title_text="Performance (%)",

    range=[0, 1.15],

    tickformat=".0%",

    row=1,

    col=1

)


# ==================================================
# CHART 2: ROC-AUC COMPARISON
# ==================================================

fig.add_trace(

    go.Bar(

        name="Test ROC-AUC",

        x=algorithms,

        y=dashboard_df["ROC-AUC"],

        text=[
            f"{value:.3f}"
            for value in dashboard_df["ROC-AUC"]
        ],

        textposition="outside"

    ),

    row=2,

    col=1

)


fig.add_trace(

    go.Bar(

        name="Cross-validation ROC-AUC",

        x=algorithms,

        y=dashboard_df["CV ROC-AUC"],

        text=[
            f"{value:.3f}"
            for value in dashboard_df["CV ROC-AUC"]
        ],

        textposition="outside"

    ),

    row=2,

    col=1

)


fig.update_yaxes(

    title_text="ROC-AUC Score",

    range=[0, 1.15],

    row=2,

    col=1

)


# ==================================================
# TABLE: DETAILED COMPARISON
# ==================================================

table_columns = [
    "Algorithm",
    "Test Accuracy",
    "Precision",
    "Sensitivity (Recall)",
    "Specificity",
    "F1 Score",
    "ROC-AUC",
    "CV ROC-AUC",
    "TN",
    "FP",
    "FN",
    "TP"
]

table_headers = [
    "Algorithm",
    "Accuracy",
    "Precision",
    "Sensitivity",
    "Specificity",
    "F1 Score",
    "Test AUC",
    "CV AUC",
    "TN",
    "FP",
    "FN",
    "TP"
]

table_values = []

for column in table_columns:

    values = dashboard_df[column].tolist()

    if column in table_columns[1:8]:

        values = [
            f"{value:.2%}"
            for value in values
        ]

    table_values.append(values)


fig.add_trace(

    go.Table(

        header=dict(

            values=table_headers,

            fill_color="#17365D",

            font=dict(
                color="white",
                size=12
            ),

            align="center",

            height=35

        ),

        cells=dict(

            values=table_values,

            fill_color=[

                [
                    "#D9EAF7"
                    if dashboard_df.iloc[i]["Algorithm"] == best_algorithm
                    else (
                        "#F1F5F9"
                        if i % 2 == 0
                        else "#FFFFFF"
                    )

                    for i in range(len(dashboard_df))
                ]

                for _ in table_columns

            ],

            font=dict(
                color="#1F2937",
                size=11
            ),

            align="center",

            height=35

        )

    ),

    row=3,

    col=1

)


# ==================================================
# DASHBOARD TITLE AND DESIGN
# ==================================================

fig.update_layout(

    title=dict(

        text=(
            "<b>HYPERTENSION PREDICTION MODEL COMPARISON</b>"
            "<br><sup>Machine Learning Algorithm Performance Evaluation</sup>"
        ),

        x=0.5,

        xanchor="center",

        font=dict(size=23)

    ),

    barmode="group",

    template="plotly_white",

    height=1650,

    margin=dict(
        l=70,
        r=50,
        t=160,
        b=70
    ),

    legend=dict(

        orientation="h",

        yanchor="bottom",

        y=1.01,

        xanchor="center",

        x=0.5

    ),

    hovermode="x unified"

)


fig.update_xaxes(
    tickangle=-15
)


# ==================================================
# DISPLAY WINNING MODEL AND REASON
# ==================================================

fig.add_annotation(

    text=(
        f"<b>WINNING MODEL: {best_algorithm}</b><br>"
        f"{best_reason}"
    ),

    xref="paper",

    yref="paper",

    x=0.5,

    y=1.075,

    showarrow=False,

    align="center",

    font=dict(
        size=13,
        color="#17365D"
    ),

    bgcolor="#E8F1FA",

    bordercolor="#17365D",

    borderwidth=1,

    borderpad=12

)


# ==================================================
# EXPORT FULL DASHBOARD
# ==================================================

fig.write_html(

    "model_comparison_dashboard.html",

    include_plotlyjs=True,

    full_html=True

)

print(
    "\nFull dashboard saved: "
    "model_comparison_dashboard.html"
)


# ==================================================
# 13. CREATE WINNING MODEL PERFORMANCE CHART
# ==================================================

print("\nCreating winning model performance chart...")

winner_metrics = [
    "Test Accuracy",
    "Precision",
    "Sensitivity (Recall)",
    "Specificity",
    "F1 Score",
    "ROC-AUC"
]

winner_values = [
    best_row[metric]
    for metric in winner_metrics
]

winner_fig = go.Figure()


winner_fig.add_trace(

    go.Bar(

        x=winner_metrics,

        y=winner_values,

        text=[
            f"{value:.1%}"
            for value in winner_values
        ],

        textposition="outside",

        marker=dict(
            color="#2471A3"
        )

    )

)


winner_fig.update_layout(

    title=dict(

        text=(
            f"<b>WINNING ALGORITHM: {best_algorithm}</b>"
            "<br><sup>Performance Evaluation</sup>"
        ),

        x=0.5,

        xanchor="center"

    ),

    yaxis=dict(

        title="Performance Score",

        range=[0, 1.15],

        tickformat=".0%"

    ),

    template="plotly_white",

    height=550

)


winner_fig.add_annotation(

    text=best_reason,

    xref="paper",

    yref="paper",

    x=0.5,

    y=-0.25,

    showarrow=False,

    align="center",

    font=dict(
        size=12,
        color="#17365D"
    )

)


winner_fig.write_html(

    "best_model_performance.html",

    include_plotlyjs=True,

    full_html=True

)


# ==================================================
# 14. FINAL SUMMARY
# ==================================================

print("\n")
print("=" * 70)
print("MODEL COMPARISON COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nSELECTED BEST ALGORITHM:")
print(best_model_name)

print("\nREASON:")
print(best_reason)

print("\nGENERATED FILES:")
print("- model_comparison_results.csv")
print("- model_comparison_dashboard.html")
print("- best_model_performance.html")
print("- best_model_explanation.txt")
print("- best_hypertension_model.pkl")

print("\nALL FIVE TRAINED MODELS HAVE BEEN SAVED.")

print("\n" + "=" * 70)
