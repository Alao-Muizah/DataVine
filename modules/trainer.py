import streamlit as st
from modules.regression import regression_trainer
from modules.classification import classification_trainer
from modules.clustering import clustering_trainer


def train_model(df):
    st.subheader("Model Training")
    st.markdown("#### What do you want to do?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📈 Regression**")
        st.caption("Predict continuous values")
        if st.button("Select", key="reg", use_container_width=True):
            st.session_state.task = "Regression"

    with col2:
        st.markdown("**🎯 Classification**")
        st.caption("Predict categories")
        if st.button("Select", key="clf", use_container_width=True):
            st.session_state.task = "Classification"

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**🔵 Clustering**")
        st.caption("Group similar data")
        if st.button("Select", key="clu", use_container_width=True):
            st.session_state.task = "Clustering"

    with col4:
        st.markdown("**📅 Time Series**")
        st.caption("Forecast trends over time")
        if st.button("Select", key="ts", use_container_width=True):
            st.session_state.task = "Time Series"


    if "task" not in st.session_state:
        st.info("Select a task above to get started.")
        return

    st.markdown("---")
    st.markdown(f"#### Selected Task: {st.session_state.task}")

    if st.session_state.task == "Regression":
        regression_trainer(df)
    elif st.session_state.task == "Classification":
        classification_trainer(df)
    elif st.session_state.task == "Clustering":
        clustering_trainer(df)
    elif st.session_state.task == "Time Series":
        st.info("Time Series coming soon.")
     