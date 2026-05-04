import streamlit as st

def inspect_data(df):
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    st.subheader("Shape")
    st.write(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")

    st.subheader("Column Types")
    st.dataframe(df.dtypes.rename("Data Type").reset_index().rename(columns={"index": "Column"}))

    st.subheader("Missing Values")
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Count"]
    missing["Percentage"] = ((missing["Missing Count"] / len(df)) * 100).round(2)
    st.dataframe(missing)

    st.subheader("Duplicate Rows")
    duplicate_count = df.duplicated().sum()
    st.write(f"Total duplicate rows: {duplicate_count}")
    if duplicate_count > 0:
        st.dataframe(df[df.duplicated(keep=False)])

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())



    