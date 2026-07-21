import streamlit as st
import pandas as pd
from modules.summarizer import summarize_dataset

def inspect_data(df):

    # --- Shape ---
    st.markdown("#### Shape")
    st.write(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")

    # --- Dataset Preview ---
    st.markdown("#### Dataset Preview")
    st.dataframe(df.head(10))

    # --- AI Dataset Summary ---
    st.markdown("####  AI Dataset Summary")
    fingerprint = (tuple(df.columns), df.shape)
    if st.session_state.get("summary_fingerprint") != fingerprint:
        with st.spinner("Summarizing dataset..."):
            st.session_state.dataset_summary = summarize_dataset(df)
            st.session_state.summary_fingerprint = fingerprint
    st.info(st.session_state.dataset_summary)

    # --- Column Types Grouped ---
    st.markdown("#### Column Types")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
    bool_cols = df.select_dtypes(include="bool").columns.tolist()

    grouped_df = pd.DataFrame({
        "Type": ["Numeric", "Categorical", "Datetime", "Boolean"],
        "Count": [
            len(numeric_cols),
            len(categorical_cols),
            len(datetime_cols),
            len(bool_cols)
        ],
        "Columns": [
            ", ".join(numeric_cols) if numeric_cols else "-",
            ", ".join(categorical_cols) if categorical_cols else "-",
            ", ".join(datetime_cols) if datetime_cols else "-",
            ", ".join(bool_cols) if bool_cols else "-"
        ]
    })

    st.dataframe(grouped_df)

    # --- Missing Values ---
    st.markdown("#### Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0].reset_index()
    if missing.empty:
        st.success("No missing values found.")
    else:
        missing.columns = ["Column", "Missing Count"]
        missing["Percentage"] = ((missing["Missing Count"] / len(df)) * 100).round(2)
        st.dataframe(missing)

    # --- Duplicate Rows ---
    st.markdown("#### Duplicate Rows")
    duplicate_count = df.duplicated().sum()
    if duplicate_count == 0:
        st.success("No duplicate rows found.")
    else:
        st.warning(f"Found {duplicate_count} duplicate rows.")
        st.dataframe(df[df.duplicated(keep=False)])

    # --- Summary Statistics ---
    st.markdown("#### Summary Statistics")
    st.dataframe(df.describe())
    st.session_state.global_stats = df.describe(include="all").to_dict()


