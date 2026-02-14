import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

st.set_page_config(page_title="ML Assignment 2 - Streamlit", layout="wide")


# -----------------------------
# Helper Functions
# -----------------------------
@st.cache_data
def load_dataset_from_repo():
    """
    Streamlit Cloud cannot access your local dataset.
    So we use a public dataset URL.

    NOTE:
    This is a stable UCI mirror.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    return url


def build_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return preprocessor


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_proba)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def train_all_models(df):
    """
    Train all 6 models on Bank Marketing dataset.
    Returns:
      - trained_models: dict of {model_name: pipeline/model}
      - metrics_df: DataFrame of evaluation metrics
      - X_test, y_test: test set for confusion matrix/report
    """

    # Convert target
    df["y"] = df["y"].map({"yes": 1, "no": 0})

    X = df.drop(columns=["y"])
    y = df["y"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=9),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42
        )
    }

    trained_models = {}
    results = []

    for name, clf in models.items():

        # Naive Bayes needs dense input, so handle separately
        if name == "Naive Bayes":
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            if hasattr(X_train_processed, "toarray"):
                X_train_processed = X_train_processed.toarray()
                X_test_processed = X_test_processed.toarray()

            clf.fit(X_train_processed, y_train)

            y_pred = clf.predict(X_test_processed)
            y_proba = clf.predict_proba(X_test_processed)[:, 1]

            trained_models[name] = (preprocessor, clf)

        else:
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", clf)
            ])

            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]

            trained_models[name] = pipe

        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics["Model"] = name
        results.append(metrics)

    metrics_df = pd.DataFrame(results)[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]

    return trained_models, metrics_df, X_test, y_test


@st.cache_data
def load_bank_data_from_zip():
    """
    Loads bank-additional-full.csv directly from UCI zip.
    """
    import zipfile
    import io
    import requests

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    # File inside zip:
    # bank-additional/bank-additional-full.csv
    with z.open("bank-additional/bank-additional-full.csv") as f:
        df = pd.read_csv(f, sep=";")

    return df


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📌 ML Assignment 2 - Classification Models + Streamlit Deployment")

st.caption("Submitted by: Raghavendra V | BITS ID: 2025AA05984")
st.caption("M.Tech AIML/DSE | Machine Learning - Assignment 2")

st.markdown(
    """
This Streamlit app implements **6 classification models** on the **UCI Bank Marketing dataset**.

### Models Implemented
- Logistic Regression  
- Decision Tree  
- KNN  
- Naive Bayes  
- Random Forest  
- XGBoost  

### Metrics Displayed
Accuracy, AUC, Precision, Recall, F1, MCC  
"""
)

st.divider()

# Train button
if "trained_models" not in st.session_state:
    st.session_state.trained_models = None
    st.session_state.metrics_df = None
    st.session_state.X_test = None
    st.session_state.y_test = None

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ Training Step")
    st.write("Click below to train all 6 models.")
    train_btn = st.button("🚀 Train Models")

with col2:
    st.subheader("📌 Notes")
    st.markdown(
        """
- Training is done only when you click **Train Models** (avoids Streamlit auto-crash).  
- Once trained, metrics and model dropdown will appear.  
- Upload your own test CSV for predictions.  
"""
    )

if train_btn:
    with st.spinner("Training models... (may take 1-2 minutes)"):
        df = load_bank_data_from_zip()
        trained_models, metrics_df, X_test, y_test = train_all_models(df)

        st.session_state.trained_models = trained_models
        st.session_state.metrics_df = metrics_df
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

    st.success("Training completed successfully!")
    
    # Show class distribution (unique UI feature)
    st.subheader("📊 Dataset Class Distribution (y)")
    y_counts = df["y"].value_counts()
    st.bar_chart(y_counts)
    st.info("The dataset is imbalanced (more 'no' than 'yes'), so precision/recall trade-offs are expected.")


st.divider()

# If trained, show metrics table + dropdown
if st.session_state.trained_models is not None:

    st.subheader("📊 Model Comparison Table (Test Set Metrics)")
    st.dataframe(st.session_state.metrics_df, use_container_width=True)

    st.divider()

    st.subheader("🧠 Select Model + Upload CSV for Prediction")

    model_names = list(st.session_state.trained_models.keys())
    selected_model = st.selectbox("Select a model", model_names)

    uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(test_df.head(10), use_container_width=True)

        try:
            model_obj = st.session_state.trained_models[selected_model]

            # Naive Bayes special case
            if selected_model == "Naive Bayes":
                preprocessor, nb_model = model_obj
                X_processed = preprocessor.transform(test_df)
                if hasattr(X_processed, "toarray"):
                    X_processed = X_processed.toarray()

                y_pred = nb_model.predict(X_processed)
                y_proba = nb_model.predict_proba(X_processed)[:, 1]

            else:
                y_pred = model_obj.predict(test_df)
                y_proba = model_obj.predict_proba(test_df)[:, 1]

            out = test_df.copy()
            out["Prediction"] = y_pred
            out["Probability_Yes"] = y_proba

            st.subheader("✅ Prediction Output")
            st.dataframe(out.head(20), use_container_width=True)

            csv_data = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv_data,
                file_name="predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

        st.divider()

        st.subheader("📌 Confusion Matrix + Classification Report (If y exists)")
        st.info(
            "If your uploaded CSV contains a column named `y` (yes/no or 0/1), "
            "the app will evaluate confusion matrix + metrics."
        )

        if "y" in test_df.columns:
            y_true = test_df["y"].copy()

            if y_true.dtype == "object":
                y_true = y_true.map({"yes": 1, "no": 0})

            y_true = y_true.astype(int)

            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

            report = classification_report(y_true, y_pred)
            st.text("Classification Report:\n" + report)

            metrics = compute_metrics(y_true, y_pred, y_proba)
            st.subheader("📌 Evaluation Metrics on Uploaded CSV")
            st.json(metrics)

        else:
            st.warning("No 'y' column found. Upload a file containing y to see evaluation.")

else:
    st.warning("Models not trained yet. Click **Train Models** to begin.")

