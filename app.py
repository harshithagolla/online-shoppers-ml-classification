import numpy as np
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

st.set_page_config(
    page_title="Online Shopper Purchase Prediction", layout="wide")

st.title("🛒 Online Shopper Purchase Prediction")
st.markdown("Upload test dataset and evaluate different ML models.")

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

# Load model
model_path = f"model/{model_name.replace(' ', '_')}.pkl"
model = joblib.load(model_path)

# Upload dataset
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "Revenue" not in df.columns:
        st.error("Uploaded file must contain 'Revenue' column.")
    else:
        df["Revenue"] = df["Revenue"].astype(int)

        X = df.drop("Revenue", axis=1)
        y = df["Revenue"]

        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_proba)
        mcc = matthews_corrcoef(y, y_pred)

        st.subheader("📊 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{acc:.4f}")
        col1.metric("AUC", f"{auc:.4f}")

        col2.metric("Precision", f"{prec:.4f}")
        col2.metric("Recall", f"{rec:.4f}")

        col3.metric("F1 Score", f"{f1:.4f}")
        col3.metric("MCC", f"{mcc:.4f}")

        st.subheader("📌 Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots(figsize=(3,3))
        ax.imshow(cm, cmap="Blues")

        ax.set_title("Confusion Matrix", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=9)
        ax.set_ylabel("True Label", fontsize=9)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Purchase", "Purchase"], fontsize=8)
        ax.set_yticklabels(["No Purchase", "Purchase"], fontsize=8)


        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j],
                        ha="center", va="center",
                        fontsize=9,
                        color="black")
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

