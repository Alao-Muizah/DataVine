import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import joblib
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ============================================================
# UTILITIES
# ============================================================

def download_chart(fig, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="⬇️ Download Chart",
        data=buf,
        file_name=filename,
        mime="image/png"
    )


def download_model(model, filename):
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    st.download_button(
        label="⬇️ Download Trained Model",
        data=buf,
        file_name=filename,
        mime="application/octet-stream"
    )


# ============================================================
# MODEL BUILDER
# ============================================================

def get_clustering_model(model_type, params):
    if model_type == "K-Means":
        return KMeans(
            n_clusters=params["n_clusters"],
            init=params["init"],
            max_iter=params["max_iter"],
            n_init=params["n_init"],
            random_state=42
        )

    elif model_type == "DBSCAN":
        return DBSCAN(
            eps=params["eps"],
            min_samples=params["min_samples"],
            metric=params["metric"],
            algorithm=params["algorithm"],
            leaf_size=params["leaf_size"]
        )

    elif model_type == "Hierarchical":
        return AgglomerativeClustering(
            n_clusters=params["n_clusters"],
            linkage=params["linkage"],
            metric=params["metric"],
            compute_full_tree=params["compute_full_tree"],
            memory=None
        )


# ============================================================
# HYPERPARAMETERS
# ============================================================

def get_clustering_params(model_type):
    params = {}

    if model_type == "K-Means":
        c1, c2 = st.columns(2)
        with c1:
            params["n_clusters"] = st.slider("Number of Clusters", 2, 15, 3, key="km_clusters")
            params["max_iter"] = st.slider("Max Iterations", 100, 1000, 300, step=50, key="km_iter")
            params["n_init"] = st.slider("Number of Initializations", 5, 20, 10, key="km_ninit")
        with c2:
            params["init"] = st.selectbox("Initialization Method", ["k-means++", "random"], key="km_init")

    elif model_type == "DBSCAN":
        c1, c2 = st.columns(2)
        with c1:
            params["eps"] = st.slider("Epsilon (eps)", 0.1, 5.0, 0.5, step=0.1, key="db_eps")
            params["min_samples"] = st.slider("Min Samples", 2, 20, 5, key="db_min")
            params["metric"] = st.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine"], key="db_metric")
        with c2:
            params["algorithm"] = st.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"], key="db_algo")
            params["leaf_size"] = st.slider("Leaf Size", 10, 100, 30, key="db_leaf")

    elif model_type == "Hierarchical":
        c1, c2 = st.columns(2)
        with c1:
            params["n_clusters"] = st.slider("Number of Clusters", 2, 15, 3, key="hc_clusters")
            params["linkage"] = st.selectbox("Linkage", ["ward", "complete", "average", "single"], key="hc_linkage")
            params["metric"] = st.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine"], key="hc_metric")
        with c2:
            params["compute_full_tree"] = st.selectbox("Compute Full Tree", ["auto", True, False], key="hc_tree")

    return params


# ============================================================
# RESULTS
# ============================================================

def show_clustering_results(model, model_type, X_scaled, X_original, feature_cols, params):
    labels = model.labels_ if hasattr(model, "labels_") else model.predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # --- Metrics ---
    st.markdown("#### Clustering Results")
    m1, m2 = st.columns(2)
    m1.metric("Number of Clusters Found", n_clusters)

    if n_clusters > 1:
        sil_score = silhouette_score(X_scaled, labels)
        m2.metric("Silhouette Score", f"{sil_score:.4f}")
    else:
        m2.metric("Silhouette Score", "N/A")

    if model_type == "DBSCAN":
        noise = np.sum(labels == -1)
        st.write(f"Noise points (outliers): {noise}")

    # --- Cluster Distribution ---
    st.markdown("#### Cluster Distribution")
    unique, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots()
    ax.bar([str(l) for l in unique], counts, color="steelblue")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.set_title("Cluster Distribution")
    ax.bar_label(ax.containers[0], fmt="%d")
    st.pyplot(fig)
    download_chart(fig, "cluster_distribution.png")
    plt.close(fig)

    # --- PCA Visualization ---
    st.markdown("#### Cluster Visualization (PCA)")
    n_components = min(2, X_scaled.shape[1])
    if n_components < 2:
        st.info("Need at least 2 features for PCA visualization.")
    else:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                             c=labels, cmap="Set2", alpha=0.6)
        plt.colorbar(scatter, ax=ax, label="Cluster")
        ax.set_title("Cluster Visualization (PCA)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        st.pyplot(fig)
        download_chart(fig, "cluster_pca.png")
        plt.close(fig)

    # --- Cluster Summary ---
    st.markdown("#### Cluster Summary")
    X_original = X_original.copy()
    X_original["Cluster"] = labels
    summary = X_original.groupby("Cluster").mean().round(2)
    st.dataframe(summary)

    # --- Download ---
    st.markdown("#### Download")
    csv = X_original.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Data with Cluster Labels",
        data=csv,
        file_name="clustered_data.csv",
        mime="text/csv"
    )

    if model_type != "Hierarchical":
        download_model(model, f"{model_type.lower().replace(' ', '_')}_model.pkl")


# ============================================================
# MAIN FUNCTION
# ============================================================

def clustering_trainer(df):
    st.markdown("### Clustering")

    model_type = st.radio(
        "Select Clustering Algorithm",
        ["K-Means", "DBSCAN", "Hierarchical"],
        horizontal=True
    )

    # --- Encode categorical columns ---
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    numeric_cols = [col for col in df.select_dtypes(include="number").columns
                    if not pd.api.types.is_datetime64_any_dtype(df[col])]

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for clustering.")
        return

    feature_cols = st.multiselect(
        "Select Feature Columns",
        numeric_cols,
        key="clu_features"
    )

    if not feature_cols:
        st.info("Please select at least two feature columns.")
        return

    st.markdown("#### Hyperparameters")
    params = get_clustering_params(model_type)

    if st.button("Run Clustering", key="train_clu"):
        X = df[feature_cols].dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = get_clustering_model(model_type, params)
        model.fit(X_scaled)

        show_clustering_results(model, model_type, X_scaled, X.copy(), feature_cols, params)





        
