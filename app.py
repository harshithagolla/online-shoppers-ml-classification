import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

st.set_page_config(page_title="Online Shopper Purchase Prediction", layout="wide")

st.title("🛒 Online Shopper Purchase Prediction")
st.markdown("Upload dataset and evaluate different ML models.")

# Model selection
model_name = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "Revenue" not in df.columns:
        st.error("Dataset must contain 'Revenue' column.")
    else:
        df["Revenue"] = df["Revenue"].astype(int)

        X = df.drop("Revenue", axis=1)
        y = df["Revenue"]

        categorical_cols = X.select_dtypes(include=["object"]).columns
        numerical_cols = X.select_dtypes(exclude=["object"]).columns

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols)
            ]
        )

        # Select model
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
        elif model_name == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_name == "Naive Bayes":
            model = GaussianNB()
        elif model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
        else:
            model = XGBClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=1
            )

        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        with st.spinner("Training model... Please wait ⏳"):
            pipe.fit(X_train, y_train)


        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader("📊 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{acc:.4f}")
        col1.metric("AUC", f"{auc:.4f}")

        col2.metric("Precision", f"{prec:.4f}")
        col2.metric("Recall", f"{rec:.4f}")

        col3.metric("F1 Score", f"{f1:.4f}")
        col3.metric("MCC", f"{mcc:.4f}")

        st.subheader("📌 Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(3,3))
        ax.imshow(cm, cmap="Blues")

        ax.set_title("Confusion Matrix", fontsize=10)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)

        ax.set_xticks([0,1])
        ax.set_yticks([0,1])
        ax.set_xticklabels(["No Purchase", "Purchase"], fontsize=8)
        ax.set_yticklabels(["No Purchase", "Purchase"], fontsize=8)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
