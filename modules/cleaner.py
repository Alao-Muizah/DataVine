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

    # --- Drop Identifier Columns ---
    identifier_keywords = ["phone", "email", "address", "name", "code", "id", "url", "zip", "postcode"]
    identifier_cols = [
        col for col in df.columns
        if any(keyword in col.lower() for keyword in identifier_keywords)
    ]
    if identifier_cols:
        df = df.drop(columns=identifier_cols)
        report.append(f"✅ Dropped identifier columns: {identifier_cols}")

    # --- Drop ID / High Cardinality Columns ---
    cols_to_drop = []
    for col in df.columns:
        if df[col].nunique() == len(df) and df[col].dtype == "object":
            cols_to_drop.append(col)
        elif df[col].nunique() == len(df) and pd.api.types.is_integer_dtype(df[col]) and len(df) > 100:
            cols_to_drop.append(col)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        report.append(f"✅ Dropped ID/unique columns: {cols_to_drop}")

    # --- Fix Data Types ---
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                converted = pd.to_datetime(df[col])
                df[col] = converted
                report.append(f"✅ Converted '{col}' to datetime.")
                continue
            except:
                pass
            try:
                df[col] = pd.to_numeric(df[col])
                report.append(f"✅ Converted '{col}' to numeric.")
                continue
            except:
                pass

    # --- Handle Missing Values ---
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing == 0:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
            report.append(f"✅ Filled {missing} missing values in '{col}' with median.")
        elif df[col].dtype == "object":
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