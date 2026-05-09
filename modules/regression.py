import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


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
# REGRESSION
# ============================================================

def get_regression_model(model_type, params):
    if model_type == "Linear Regression":
        return LinearRegression()

    elif model_type == "Polynomial Regression":
        return Pipeline([
            ("poly", PolynomialFeatures(
                degree=params["degree"],
                include_bias=False
            )),
            ("linear", LinearRegression())
        ])

    elif model_type == "Decision Tree":
        return DecisionTreeRegressor(
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            max_leaf_nodes=params["max_leaf_nodes"],
            random_state=42
        )

    elif model_type == "Random Forest":
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=42
        )

    elif model_type == "XGBoost":
        return XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            random_state=42
        )


def get_regression_params(model_type):
    params = {}

    if model_type == "Polynomial Regression":
        params["degree"] = st.slider("Polynomial Degree", 2, 4, 2)

    elif model_type == "Decision Tree":
        c1, c2 = st.columns(2)
        with c1:
            params["max_depth"] = st.slider("Max Depth", 1, 20, 5, key="dt_depth")
            params["min_samples_split"] = st.slider("Min Samples Split", 2, 20, 2, key="dt_split")
            params["min_samples_leaf"] = st.slider("Min Samples Leaf", 1, 20, 1, key="dt_leaf")
        with c2:
            params["max_features"] = st.selectbox("Max Features", ["sqrt", "log2", None], key="dt_features")
            params["max_leaf_nodes"] = st.slider("Max Leaf Nodes", 2, 100, 20, key="dt_leaves")

    elif model_type == "Random Forest":
        c1, c2 = st.columns(2)
        with c1:
            params["n_estimators"] = st.slider("Number of Trees", 10, 300, 100, step=10, key="rf_trees")
            params["max_depth"] = st.slider("Max Depth", 1, 20, 5, key="rf_depth")
            params["min_samples_split"] = st.slider("Min Samples Split", 2, 20, 2, key="rf_split")
        with c2:
            params["min_samples_leaf"] = st.slider("Min Samples Leaf", 1, 20, 1, key="rf_leaf")
            params["max_features"] = st.selectbox("Max Features", ["sqrt", "log2", None], key="rf_features")

    elif model_type == "XGBoost":
        c1, c2 = st.columns(2)
        with c1:
            params["n_estimators"] = st.slider("Number of Trees", 10, 300, 100, step=10, key="xgb_trees")
            params["learning_rate"] = st.slider("Learning Rate", 0.01, 0.5, 0.1, step=0.01, key="xgb_lr")
            params["max_depth"] = st.slider("Max Depth", 1, 10, 3, key="xgb_depth")
        with c2:
            params["subsample"] = st.slider("Subsample", 0.1, 1.0, 0.8, step=0.1, key="xgb_sub")
            params["colsample_bytree"] = st.slider("Col Sample by Tree", 0.1, 1.0, 0.8, step=0.1, key="xgb_col")

    return params


def show_regression_results(model, model_type, feature_cols, X, y, y_test, y_pred, degree=2):
    # --- Metrics ---
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.markdown("#### Model Performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", f"{r2:.4f}")
    m2.metric("MAE", f"{mae:.4f}")
    m3.metric("RMSE", f"{rmse:.4f}")

    # --- Actual vs Predicted ---
    st.markdown("#### Actual vs Predicted")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color="steelblue", alpha=0.6, label="Predictions")
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            color="red", linestyle="--", label="Perfect Fit")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    st.pyplot(fig)
    download_chart(fig, "actual_vs_predicted.png")
    plt.close(fig)

    # --- Feature Importance ---
    st.markdown("#### Feature Importance")
    if model_type == "Linear Regression":
        importance = model.coef_
        label = "Coefficient"
    elif model_type in ["Decision Tree", "Random Forest", "XGBoost"]:
        importance = model.feature_importances_
        label = "Importance"
    else:
        importance = None

    if importance is not None:
        imp_df = pd.DataFrame({
            "Feature": feature_cols,
            label: importance
        }).sort_values(label, ascending=False)

        fig, ax = plt.subplots()
        ax.barh(imp_df["Feature"], imp_df[label], color="steelblue")
        ax.set_title(f"Feature {label}")
        ax.set_xlabel(label)
        if model_type == "Linear Regression":
            ax.axvline(x=0, color="red", linestyle="--")
        st.pyplot(fig)
        download_chart(fig, "feature_importance.png")
        plt.close(fig)

    # --- Polynomial Curve ---
    if model_type == "Polynomial Regression" and len(feature_cols) == 1:
        st.markdown("#### Polynomial Fit Curve")
        X_plot = np.linspace(X[feature_cols[0]].min(), X[feature_cols[0]].max(), 300).reshape(-1, 1)
        y_plot = model.predict(pd.DataFrame(X_plot, columns=feature_cols))

        fig, ax = plt.subplots()
        ax.scatter(X[feature_cols[0]], y, color="steelblue", alpha=0.4, label="Data Points")
        ax.plot(X_plot, y_plot, color="red", label=f"Degree {degree} Fit")
        ax.set_xlabel(feature_cols[0])
        ax.set_ylabel("Target")
        ax.set_title(f"Polynomial Regression Fit (Degree {degree})")
        ax.legend()
        st.pyplot(fig)
        download_chart(fig, "polynomial_fit.png")
        plt.close(fig)

    # --- Download Model ---
    st.markdown("#### Download Model")
    download_model(model, f"{model_type.lower().replace(' ', '_')}_model.pkl")


def regression_trainer(df):
    st.markdown("### Regression")

    model_type = st.radio(
        "Select Regression Type",
        ["Linear Regression", "Polynomial Regression", "Decision Tree", "Random Forest", "XGBoost"],
        horizontal=True
    )

    # --- Encode categorical columns ---
    df = df.copy()
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    numeric_cols = [col for col in df.select_dtypes(include="number").columns
                    if not pd.api.types.is_datetime64_any_dtype(df[col])]

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for regression.")
        return

    target_col = st.selectbox("Select Target Column (Y)", numeric_cols, key="reg_target")
    feature_cols = st.multiselect(
        "Select Feature Columns (X)",
        [col for col in numeric_cols if col != target_col],
        key="reg_features"
    )

    if not feature_cols:
        st.info("Please select at least one feature column.")
        return

    test_size = st.slider("Test Size (%)", min_value=10, max_value=40, value=20, step=5) / 100

    st.markdown("#### Hyperparameters")
    params = get_regression_params(model_type)

    if st.button("Train Model", key="train_reg"):
        X = df[feature_cols].dropna()
        y = df[target_col].loc[X.index]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = get_regression_model(model_type, params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        show_regression_results(
            model, model_type, feature_cols,
            X, y, y_test, y_pred,
            degree=params.get("degree", 2)
        )





    