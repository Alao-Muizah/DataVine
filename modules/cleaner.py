import pandas as pd
import streamlit as st


def clean_data(df):
    st.subheader("Data Cleaning")

    df = df.copy()

    # ============================================================
    # STEP 1: SCAN — show issues before doing anything
    # ============================================================

    st.markdown("#### Step 1: Dataset Scan")
    st.caption("DataVine scanned your dataset and found the following issues. Review and decide how to handle each one.")

    # --- Detect missing values ---
    missing_cols = {
        col: int(df[col].isnull().sum())
        for col in df.columns
        if df[col].isnull().sum() > 0
    }

    # --- Detect high cardinality object columns ---
    high_cardinality_cols = [
    col for col in df.columns
    if df[col].dtype == "object"
    and df[col].nunique() > 20
    ]

    # --- Detect identifier columns ---
    identifier_keywords = ["phone", "email", "address", "name", "code", "id", "url", "zip", "postcode"]
    identifier_cols = [
        col for col in df.columns
        if any(keyword in col.lower() for keyword in identifier_keywords)
    ]

    if not missing_cols and not high_cardinality_cols and not identifier_cols:
        st.success(" No issues found. Your dataset looks clean.")
    else:
        if identifier_cols:
            st.info(f" **Identifier columns detected** — these columns appear to be IDs or contact info and are not useful for analysis: `{identifier_cols}`")
        if high_cardinality_cols:
            st.warning(f" **High cardinality columns** — every row has a unique value, making these columns unsuitable for analysis: `{high_cardinality_cols}`")
        if missing_cols:
            st.warning(f" **Columns with missing values** — {len(missing_cols)} column(s) have missing data: `{list(missing_cols.keys())}`")

    st.divider()

    # ============================================================
    # STEP 2: USER DECISIONS
    # ============================================================

    st.markdown("#### Step 2: Your Decisions")
    user_decisions = {}

    # --- Identifier / High Cardinality Columns ---
    flagged_cols = list(dict.fromkeys(identifier_cols + high_cardinality_cols))  # union, no duplicates, order preserved

    if flagged_cols:
        st.markdown("Identifier / High Cardinality Columns")
        for col in flagged_cols:
            if col in identifier_cols and col in high_cardinality_cols:
                reason = f"looks like an identifier, {df[col].nunique()} unique values"
            elif col in identifier_cols:
                reason = "looks like an identifier"
            else:
                reason = f"{df[col].nunique()} unique values, every row is different"

            action = st.radio(
                f"`{col}` — {reason}",
                ["Drop", "Keep"],
                horizontal=True,
                key=f"flag_{col}"
            )
            user_decisions[col] = {"type": "identifier_or_high_cardinality", "action": action}
        st.markdown("---")

    # --- Missing value columns ---
    if missing_cols:
        st.markdown("Missing Value Columns")
        for col, count in missing_cols.items():
            pct = round(count / len(df) * 100, 1)
            st.markdown(f"`{col}` — **{count} missing values** ({pct}% of rows)")

            action = st.radio(
                f"What to do with `{col}`?",
                ["Fill", "Drop rows"],
                horizontal=True,
                key=f"mv_action_{col}"
            )

            strategy = None
            custom = None

            if action == "Fill":
                if pd.api.types.is_numeric_dtype(df[col]):
                    strategy = st.selectbox(
                        "Fill strategy",
                        ["Median", "Mean", "Mode", "Custom value"],
                        key=f"mv_strategy_{col}"
                    )
                else:
                    strategy = st.selectbox(
                        "Fill strategy",
                        ["Mode", "Custom value"],
                        key=f"mv_strategy_{col}"
                    )
                if strategy == "Custom value":
                    custom = st.text_input(
                        f"Enter custom value for `{col}`",
                        key=f"mv_custom_{col}"
                    )

            user_decisions[col] = {
                "type": "missing",
                "action": action,
                "strategy": strategy,
                "custom": custom
            }
            st.markdown("---")


    # ============================================================
    # STEP 3: APPLY CLEANING
    # ============================================================

    if st.button("Apply Cleaning", key="apply_cleaning", use_container_width=True):
        report = []

        # --- Drop duplicates ---
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if before - after > 0:
            report.append(f" Removed {before - after} duplicate rows.")
        else:
            report.append(" No duplicate rows found.")

        # --- Fix data types first ---
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    converted = pd.to_datetime(df[col], errors="coerce")
                    if converted.notna().sum() > len(df) * 0.8:
                        df[col] = converted
                        report.append(f" Converted '{col}' to datetime.")
                        continue
                except:
                    pass
                try:
                    df[col] = pd.to_numeric(df[col])
                    report.append(f" Converted '{col}' to numeric.")
                    continue
                except:
                    pass

        # --- Apply user decisions ---
        for col, decision in user_decisions.items():
            if col not in df.columns:
                continue

            if decision["type"] in ["identifier", "high_cardinality"]:
                if decision["action"] == "Drop":
                    df = df.drop(columns=[col])
                    report.append(f" Dropped column `{col}`.")
                else:
                    report.append(f" Kept column `{col}`.")

            elif decision["type"] == "missing":
                if decision["action"] == "Drop rows":
                    before = len(df)
                    df = df.dropna(subset=[col])
                    report.append(f" Dropped {before - len(df)} rows with missing `{col}`.")
                elif decision["action"] == "Fill":
                    strategy = decision["strategy"]
                    custom = decision["custom"]
                    if strategy == "Median":
                        df[col] = df[col].fillna(df[col].median())
                        report.append(f" Filled `{col}` with median.")
                    elif strategy == "Mean":
                        df[col] = df[col].fillna(df[col].mean())
                        report.append(f" Filled `{col}` with mean.")
                    elif strategy == "Mode":
                        df[col] = df[col].fillna(df[col].mode()[0])
                        report.append(f" Filled `{col}` with mode.")
                    elif strategy == "Custom value" and custom:
                        df[col] = df[col].fillna(custom)
                        report.append(f" Filled `{col}` with '{custom}'.")

        # --- Store cleaned df in session state ---
        st.session_state.df_clean = df

        # --- Cleaning Report ---
        st.markdown("#### Cleaning Report")
        for item in report:
            st.write(item)

        # --- Preview ---
        st.markdown("#### Cleaned Data Preview")
        st.dataframe(df.head(10))

        # --- Download ---
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Cleaned Dataset",
            data=csv,
            file_name="datavine_cleaned.csv",
            mime="text/csv"
        )

    return df
