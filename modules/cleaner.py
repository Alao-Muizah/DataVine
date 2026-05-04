import streamlit as st

def clean_data(df):
    st.subheader("Data Cleaning")
    

    # --- Drop Duplicates ---
    st.markdown("#### Remove Duplicate Rows")
    if st.button("Drop Duplicates"):
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        st.success(f"Removed {before - after} duplicate rows.")


    # --- Handle Missing Values ---
    st.markdown("#### Handle Missing Values")
    strategy = st.selectbox("Choose a strategy", [
        "Drop rows with missing values",
        "Fill with mean",
        "Fill with median",
        "Fill with mode",
        "Fill with custom value"
    ])

    if strategy == "Fill with custom value":
        custom_value = st.text_input("Enter custom value")

    if st.button("Apply Missing Value Strategy"):
        if strategy == "Drop rows with missing values":
            before = len(df)
            df = df.dropna()
            st.success(f"Dropped {before - len(df)} rows with missing values.")
        elif strategy == "Fill with mean":
            df = df.fillna(df.mean(numeric_only=True))
            st.success("Filled missing values with column mean.")
        elif strategy == "Fill with median":
            df = df.fillna(df.median(numeric_only=True))
            st.success("Filled missing values with column median.")
        elif strategy == "Fill with mode":
            df = df.fillna(df.mode().iloc[0])
            st.success("Filled missing values with column mode.")
        elif strategy == "Fill with custom value" and custom_value:
            df = df.fillna(custom_value)
            st.success(f"Filled missing values with '{custom_value}'.")


    # --- Drop Columns ---
    st.markdown("#### Drop Columns")
    columns_to_drop = st.multiselect("Select columns to drop", df.columns.tolist())
    if st.button("Drop Selected Columns"):
        df = df.drop(columns=columns_to_drop)
        st.success(f"Dropped columns: {columns_to_drop}")


    # --- Fix Column Data Types ---
    st.markdown("#### Fix Column Data Types")
    col_to_convert = st.selectbox("Select column to convert", df.columns.tolist(), key="convert_col")
    new_type = st.selectbox("Convert to", ["int", "float", "str", "datetime"], key="convert_type")
    if st.button("Convert Column Type"):
        try:
            if new_type == "datetime":
                df[col_to_convert] = pd.to_datetime(df[col_to_convert])
            else:
                df[col_to_convert] = df[col_to_convert].astype(new_type)
            st.success(f"Converted '{col_to_convert}' to {new_type}.")
        except Exception as e:
            st.error(f"Could not convert column: {e}")


    # --- Rename Columns ---
    st.markdown("#### Rename a Column")
    col_to_rename = st.selectbox("Select column to rename", df.columns.tolist())
    new_name = st.text_input("New column name")
    if st.button("Rename Column"):
        df = df.rename(columns={col_to_rename: new_name})
        st.success(f"Renamed '{col_to_rename}' to '{new_name}'.")


    # --- Preview Cleaned Data ---
    st.markdown("#### Cleaned Data Preview")
    st.dataframe(df.head(10))

    return df





