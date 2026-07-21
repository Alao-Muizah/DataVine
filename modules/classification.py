import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# ============================================================
# UTILITIES
# ============================================================

def download_chart(fig, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="⬇️ Download Chart",
        data=buf,
        file_name=filename,
        mime="image/png"
    )


def download_model(model, filename):
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    st.download_button(
        label="⬇️ Download Trained Model",
        data=buf,
        file_name=filename,
        mime="application/octet-stream"
    )


# ============================================================
# MODEL BUILDER
# ============================================================

def get_classification_model(model_type, params):
    if model_type == "Logistic Regression":
        return LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"],
            solver=params["solver"],
            penalty=params["penalty"],
            random_state=42
        )

    elif model_type == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            max_leaf_nodes=params["max_leaf_nodes"],
            random_state=42
        )

    elif model_type == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=42
        )

    elif model_type == "XGBoost":
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )

    elif model_type == "SVM":
        return SVC(
            C=params["C"],
            kernel=params["kernel"],
            gamma=params["gamma"],
            degree=params["degree"],
            max_iter=params["max_iter"],
            random_state=42
        )


# ============================================================
# HYPERPARAMETERS
# ============================================================

def get_classification_params(model_type):
    params = {}

    if model_type == "Logistic Regression":
        c1, c2 = st.columns(2)
        with c1:
            params["C"] = st.slider("Regularization (C)", 0.01, 10.0, 1.0, step=0.01, key="lr_C")
            params["max_iter"] = st.slider("Max Iterations", 100, 1000, 200, step=50, key="lr_iter")
            params["solver"] = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"], key="lr_solver")
        with c2:
            params["penalty"] = st.selectbox("Penalty", ["l2", "l1", "none"], key="lr_penalty")

    elif model_type == "Decision Tree":
        c1, c2 = st.columns(2)
        with c1:
            params["max_depth"] = st.slider("Max Depth", 1, 20, 5, key="dtc_depth")
            params["min_samples_split"] = st.slider("Min Samples Split", 2, 20, 2, key="dtc_split")
            params["min_samples_leaf"] = st.slider("Min Samples Leaf", 1, 20, 1, key="dtc_leaf")
        with c2:
            params["max_features"] = st.selectbox("Max Features", ["sqrt", "log2", None], key="dtc_features")
            params["max_leaf_nodes"] = st.slider("Max Leaf Nodes", 2, 100, 20, key="dtc_leaves")

    elif model_type == "Random Forest":
        c1, c2 = st.columns(2)
        with c1:
            params["n_estimators"] = st.slider("Number of Trees", 10, 300, 100, step=10, key="rfc_trees")
            params["max_depth"] = st.slider("Max Depth", 1, 20, 5, key="rfc_depth")
            params["min_samples_split"] = st.slider("Min Samples Split", 2, 20, 2, key="rfc_split")
        with c2:
            params["min_samples_leaf"] = st.slider("Min Samples Leaf", 1, 20, 1, key="rfc_leaf")
            params["max_features"] = st.selectbox("Max Features", ["sqrt", "log2", None], key="rfc_features")

    elif model_type == "XGBoost":
        c1, c2 = st.columns(2)
        with c1:
            params["n_estimators"] = st.slider("Number of Trees", 10, 300, 100, step=10, key="xgbc_trees")
            params["learning_rate"] = st.slider("Learning Rate", 0.01, 0.5, 0.1, step=0.01, key="xgbc_lr")
            params["max_depth"] = st.slider("Max Depth", 1, 10, 3, key="xgbc_depth")
        with c2:
            params["subsample"] = st.slider("Subsample", 0.1, 1.0, 0.8, step=0.1, key="xgbc_sub")
            params["colsample_bytree"] = st.slider("Col Sample by Tree", 0.1, 1.0, 0.8, step=0.1, key="xgbc_col")

    elif model_type == "SVM":
        c1, c2 = st.columns(2)
        with c1:
            params["C"] = st.slider("Regularization (C)", 0.01, 10.0, 1.0, step=0.01, key="svm_C")
            params["kernel"] = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], key="svm_kernel")
            params["gamma"] = st.selectbox("Gamma", ["scale", "auto"], key="svm_gamma")
        with c2:
            params["degree"] = st.slider("Degree (poly only)", 2, 5, 3, key="svm_degree")
            params["max_iter"] = st.slider("Max Iterations", 100, 2000, 500, step=100, key="svm_iter")

    return params


# ============================================================
# RESULTS
# ============================================================

def show_classification_results(model, model_type, feature_cols, y_test, y_pred):
    # --- Metrics ---
    st.markdown("#### Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    m2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    m3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    m4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

    # --- Classification Report ---
    st.markdown("#### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(4))

    # --- Confusion Matrix ---
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns_like = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(sns_like, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    tick_marks = np.arange(len(np.unique(y_test)))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    st.pyplot(fig)
    download_chart(fig, "confusion_matrix.png")
    plt.close(fig)

    # --- Feature Importance ---
    if model_type in ["Decision Tree", "Random Forest", "XGBoost"]:
        st.markdown("#### Feature Importance")
        imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig, ax = plt.subplots()
        ax.barh(imp_df["Feature"], imp_df["Importance"], color="steelblue")
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance")
        st.pyplot(fig)
        download_chart(fig, "feature_importance.png")
        plt.close(fig)

    # --- Download Model ---
    st.markdown("#### Download Model")
    download_model(model, f"{model_type.lower().replace(' ', '_')}_model.pkl")


# ============================================================
# MAIN FUNCTION
# ============================================================

def classification_trainer(df):
    st.markdown("### Classification")

    model_type = st.radio(
        "Select Classification Model",
        ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "SVM"],
        horizontal=True
    )

    # --- Encode categorical columns ---
    df = df.copy()
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    all_cols = df.columns.tolist()
    numeric_cols = [col for col in df.select_dtypes(include="number").columns
                    if not pd.api.types.is_datetime64_any_dtype(df[col])]

    target_col = st.selectbox("Select Target Column (Y)", all_cols, key="clf_target")
    feature_cols = st.multiselect(
        "Select Feature Columns (X)",
        [col for col in numeric_cols if col != target_col],
        key="clf_features"
    )

    if not feature_cols:
        st.info("Please select at least one feature column.")
        return

    test_size = st.slider("Test Size (%)", min_value=10, max_value=40, value=20, step=5, key="clf_split") / 100

    st.markdown("#### Hyperparameters")
    params = get_classification_params(model_type)

    if st.button("Train Model", key="train_clf"):
        X = df[feature_cols].dropna()
        y = df[target_col].loc[X.index]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        model = get_classification_model(model_type, params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        show_classification_results(model, model_type, feature_cols, y_test, y_pred)




        