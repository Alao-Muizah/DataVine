import pandas as pd
import streamlit as st

def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        data = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None
    
    return data
    