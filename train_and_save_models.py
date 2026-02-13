import os
import joblib
import numpy as np
import pandas as pd

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
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


def load_dataset():
    # Load from local CSV (recommended for BITS Lab)
    path = "bank-additional-full.csv"
    df = pd.read_csv(path, sep=";")
    return df



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
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    print("Loading dataset...")
    df = load_dataset()

    # Target: y (yes/no)
    # Convert to 1/0
    df["y"] = df["y"].map({"yes": 1, "no": 0})

    X = df.drop(columns=["y"])
    y = df["y"].astype(int)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42
        )
    }

    os.makedirs("model", exist_ok=True)

    results = []

    for name, clf in models.items():
        print(f"\nTraining: {name}")

        # Naive Bayes needs dense input, so we handle separately
        if name == "Naive Bayes":
            # Preprocess separately
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Convert sparse -> dense if needed
            if hasattr(X_train_processed, "toarray"):
                X_train_processed = X_train_processed.toarray()
                X_test_processed = X_test_processed.toarray()

            clf.fit(X_train_processed, y_train)

            y_pred = clf.predict(X_test_processed)
            y_proba = clf.predict_proba(X_test_processed)[:, 1]

            # Save both preprocessor + model
            joblib.dump(preprocessor, f"model/{name}_preprocessor.pkl")
            joblib.dump(clf, f"model/{name}.pkl")

        else:
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", clf)
            ])

            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]

            joblib.dump(pipe, f"model/{name}.pkl")

        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics["Model"] = name
        results.append(metrics)

    results_df = pd.DataFrame(results)[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]

    results_df.to_csv("model/model_metrics.csv", index=False)

    print("\n==============================")
    print("Training complete. Saved models in /model")
    print("Saved metrics: model/model_metrics.csv")
    print("==============================\n")
    print(results_df)


if __name__ == "__main__":
    main()
