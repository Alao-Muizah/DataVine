import pandas as pd
import streamlit as st

def clean_data(df):
    st.subheader("Automated Data Cleaning")
    report = []

    # --- Drop Duplicates ---
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before - after > 0:
        report.append(f"✅ Removed {before - after} duplicate rows.")
    else:
        report.append("✅ No duplicate rows found.")

    # --- Drop ID / High Cardinality Columns ---
    cols_to_drop = []
    for col in df.columns:
        if df[col].nunique() == len(df) and df[col].dtype == "object":
            cols_to_drop.append(col)
        elif df[col].nunique() == len(df) and pd.api.types.is_integer_dtype(df[col]):
            cols_to_drop.append(col)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        report.append(f"✅ Dropped ID/unique columns: {cols_to_drop}")

    # --- Fix Data Types ---
    for col in df.columns:
        if df[col].dtype == "object":
            # Try datetime first
            try:
                df[col] = pd.to_datetime(df[col])
                report.append(f"✅ Converted '{col}' to datetime.")
                continue
            except:
                pass
            # Try numeric
            try:
                df[col] = pd.to_numeric(df[col])
                report.append(f"✅ Converted '{col}' to numeric.")
                continue
            except:
                pass

    # --- Handle Missing Values ---
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    for col in numeric_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            df[col] = df[col].fillna(df[col].median())
            report.append(f"✅ Filled {missing} missing values in '{col}' with median ({df[col].median():.2f}).")

    for col in categorical_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            report.append(f"✅ Filled {missing} missing values in '{col}' with mode ('{mode_val}').")

    # --- Cleaning Report ---
    st.markdown("#### Cleaning Report")
    for item in report:
        st.write(item)

    # --- Preview Cleaned Data ---
    st.markdown("#### Cleaned Data Preview")
    st.dataframe(df.head(10))

    return df
