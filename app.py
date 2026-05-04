import streamlit as st
import pandas as pd
from modules.loader import load_data
from modules.inspector import inspect_data
from modules.cleaner import clean_data
from modules.visualizer import visualize_data

# --- Page Config ---
st.set_page_config(
    page_title="DataVine",
    page_icon="🍇",
    layout="wide"
)

# --- Header ---
st.title(" DataVine")
st.markdown("*A data platform for cleaning, exploring, visualizing and automatically drawing insights from your datasets.*")
st.divider()

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your CSV or Excel file to get started", type=["csv", "xlsx"])

# --- Session State ---
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None

# --- Pipeline ---
if uploaded_file is not None:
    df_raw = load_data(uploaded_file)

    if df_raw is not None:
        st.session_state.df_raw = df_raw

        tab1, tab2, tab3 = st.tabs(["🔍 Inspect", "🧹 Clean", "📊 Visualize"])

        with tab1:
            inspect_data(df_raw)

        with tab2:
            df_clean = clean_data(df_raw.copy())
            st.session_state.df_clean = df_clean

            st.markdown("#### Download Cleaned Dataset")
            csv = df_clean.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="datavine_cleaned.csv",
                mime="text/csv"
            )

        with tab3:
            if st.session_state.df_clean is not None:
                visualize_data(st.session_state.df_clean)
            else:
                st.info("Please go to the Clean tab first before visualizing.")

else:
    st.info("📂 Upload a CSV or Excel file above to get started.")

    