import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

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

import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="ML Assignment 2 - Classification Models", layout="wide")


@st.cache_data
def load_metrics_table():
    path = "model/model_metrics.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def list_models():
    if not os.path.exists("model"):
        return []
    files = [f for f in os.listdir("model") if f.endswith(".pkl") and "preprocessor" not in f]
    return sorted(files)


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    st.title("📌 ML Assignment 2 - Classification Models (6 Models + Streamlit Deployment)")

    st.markdown(
        """
This app demonstrates **6 classification models** trained on the **UCI Bank Marketing dataset**.

### Models Implemented
- Logistic Regression  
- Decision Tree  
- KNN  
- Naive Bayes  
- Random Forest  
- XGBoost  

### Evaluation Metrics
Accuracy, AUC, Precision, Recall, F1, MCC
"""
    )

    st.divider()

    # Load metrics table
    metrics_table = load_metrics_table()

    colA, colB = st.columns([1, 1])

    with colA:
        st.subheader("📊 Model Comparison Table (Test Set Metrics)")
        if metrics_table is None:
            st.warning("Metrics table not found. Please run train_and_save_models.py first.")
        else:
            st.dataframe(metrics_table, use_container_width=True)

    with colB:
        st.subheader("📝 Observations (Quick Notes)")
        st.markdown(
            """
- **Logistic Regression**: Simple baseline, works well with large data.  
- **Decision Tree**: Can overfit, depends heavily on depth.  
- **KNN**: Works but slower for large datasets.  
- **Naive Bayes**: Fast, but assumes feature independence.  
- **Random Forest**: Strong performance and stable.  
- **XGBoost**: Usually best, handles non-linearity well.  
"""
        )

    st.divider()

    st.subheader("⬆️ Upload Test CSV & Predict")

    st.info(
        "Upload a CSV file containing test data. "
        "It should have the SAME feature columns as the training dataset."
    )

    uploaded_file = st.file_uploader("Upload CSV (Test Data Only)", type=["csv"])

    model_files = list_models()

    if len(model_files) == 0:
        st.error("No models found. Run train_and_save_models.py to generate models first.")
        return

    selected_model_file = st.selectbox(
        "Select a Model",
        model_files
    )

    # Load selected model
    model_path = os.path.join("model", selected_model_file)
    loaded_model = joblib.load(model_path)

    # If user uploads data
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(test_df.head(10), use_container_width=True)

        st.divider()

        st.subheader("🔍 Prediction Output")

        # Prediction
        try:
            # Special case: Naive Bayes (stored separately)
            if "Naive Bayes" in selected_model_file:
                preprocessor = joblib.load("model/Naive Bayes_preprocessor.pkl")
                X_processed = preprocessor.transform(test_df)

                if hasattr(X_processed, "toarray"):
                    X_processed = X_processed.toarray()

                y_pred = loaded_model.predict(X_processed)
                y_proba = loaded_model.predict_proba(X_processed)[:, 1]

            else:
                y_pred = loaded_model.predict(test_df)
                y_proba = loaded_model.predict_proba(test_df)[:, 1]

            output = test_df.copy()
            output["Prediction"] = y_pred
            output["Probability_Yes"] = y_proba

            st.dataframe(output.head(20), use_container_width=True)

            st.success("Prediction completed successfully!")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        st.divider()

        st.subheader("📌 Confusion Matrix + Classification Report (Demo)")
        st.warning(
            "Confusion matrix requires true labels. "
            "If your uploaded CSV includes a column named 'y', the app will evaluate it."
        )

        if "y" in test_df.columns:
            y_true = test_df["y"].copy()

            # Convert if y is yes/no
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

            report = classification_report(y_true, y_pred, output_dict=False)
            st.text("Classification Report:\n" + report)

            # Show metrics too
            metrics = compute_metrics(y_true, y_pred, y_proba)

            st.subheader("📌 Evaluation Metrics on Uploaded CSV")
            st.json(metrics)

        else:
            st.info("No 'y' column found in uploaded CSV, so evaluation cannot be shown.")

    else:
        st.info("Upload a CSV file to run predictions.")


if __name__ == "__main__":
    main()
