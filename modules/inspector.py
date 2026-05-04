import streamlit as st
import pandas as pd

def inspect_data(df):

    # --- Dataset Preview ---
    st.markdown("#### Dataset Preview")
    st.dataframe(df.head(10))

    # --- Shape ---
    st.markdown("#### Shape")
    st.write(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")

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


