import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import io
from modules.summarizer import summarize_visuals, stats_corr_matrix, stats_categorical, stats_grouped, stats_numeric, stats_time, stats_two_numeric

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

def visualize_data(df):
    st.subheader("Data Visualization")
    numeric_cols = [
        col for col in df.select_dtypes(include="number").columns.tolist()
        if not (df[col].nunique() == len(df) and df[col].nunique() > 100)
    ]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [
        col for col in df.select_dtypes(include="number").columns.tolist()
        if not pd.api.types.is_datetime64_any_dtype(df[col])
    ]
    categorical_cols = [
        col for col in df.select_dtypes(include="object").columns.tolist()
    ]
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    
    if not numeric_cols and not categorical_cols and not datetime_cols:
        st.warning("No columns available for visualization.")
        return

    # --- Single Column Charts ---
    st.markdown("#### Single Column")
    selected_col = st.selectbox("Select a column", numeric_cols + categorical_cols + datetime_cols)

    if selected_col in numeric_cols:
        chart_type = st.selectbox("Chart Type", ["Histogram", "Box Plot", "Violin Plot"])

        if chart_type == "Histogram":
            fig, ax = plt.subplots()
            sns.histplot(df[selected_col].dropna(), kde=True, ax=ax, color="steelblue", label=selected_col)
            ax.set_title(f"Distribution of {selected_col}")
            ax.set_xlabel(selected_col)
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)
            download_chart(fig, f"histogram_{selected_col}.png")
            plt.close(fig)

            insight = summarize_visuals(
                "Histogram",
                stats_numeric(df[selected_col]),
                st.session_state.dataset_summary,
                st.session_state.global_stats
            )
            st.info(insight)

        elif chart_type == "Box Plot":
            fig, ax = plt.subplots()
            sns.boxplot(y=df[selected_col].dropna(), ax=ax, color="lightblue")
            ax.set_title(f"Box Plot of {selected_col}")
            ax.set_ylabel(selected_col)
            handles = [plt.Line2D([0], [0], color="lightblue", lw=4, label=selected_col)]
            ax.legend(handles=handles)
            st.pyplot(fig)
            download_chart(fig, f"boxplot_{selected_col}.png")
            plt.close(fig)

            insight = summarize_visuals(
                "Box Plot",
                stats_numeric(df[selected_col]),
                st.session_state.dataset_summary,
                st.session_state.global_stats
            )
            st.info(insight)

        elif chart_type == "Violin Plot":
            fig, ax = plt.subplots()
            sns.violinplot(y=df[selected_col].dropna(), ax=ax, color="mediumpurple")
            ax.set_title(f"Violin Plot of {selected_col}")
            ax.set_ylabel(selected_col)
            handles = [plt.Line2D([0], [0], color="mediumpurple", lw=4, label=selected_col)]
            ax.legend(handles=handles)
            st.pyplot(fig)
            download_chart(fig, f"violin_{selected_col}.png")
            plt.close(fig)
        
            insight = summarize_visuals(
                "Violin Plot",
                stats_numeric(df[selected_col]),
                st.session_state.dataset_summary,
                st.session_state.global_stats
            )
            st.info(insight)

    # ---- Categorical column charts --- 
    elif selected_col in categorical_cols:
        chart_type = st.selectbox("Chart Type", ["Bar Chart", "Pie Chart"])
        n_classes = df[selected_col].nunique()
        CATEGORY_LIMIT = 20

        if n_classes > CATEGORY_LIMIT:
            st.warning(
                f"`{selected_col}` has {n_classes} distinct values — too many to "
                f"plot clearly. Showing only an AI summary of the value distribution instead."
            )
            insight = summarize_visuals(
                chart_type,
                stats_categorical(df[selected_col]),
                st.session_state.dataset_summary,
                st.session_state.global_stats
            )
            st.info(insight)

        elif chart_type == "Bar Chart":
            fig, ax = plt.subplots()
            counts = df[selected_col].value_counts()
            bars = counts.plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title(f"Bar Chart of {selected_col}")
            ax.set_xlabel(selected_col)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45)
            ax.bar_label(ax.containers[0], fmt="%d")
            handles = [plt.Line2D([0], [0], color="steelblue", lw=4, label=selected_col)]
            ax.legend(handles=handles)
            st.pyplot(fig)
            download_chart(fig, f"barchart_{selected_col}.png")
            plt.close(fig)

            insight = summarize_visuals(
                "Bar Chart",
                stats_categorical(df[selected_col]),
                st.session_state.dataset_summary,
                st.session_state.global_stats
            )
            st.info(insight)

        elif chart_type == "Pie Chart":
            fig, ax = plt.subplots()
            counts = df[selected_col].value_counts()
            ax.pie(
                counts,
                labels=counts.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=sns.color_palette("Set2", len(counts))
            )
            ax.set_title(f"Pie Chart of {selected_col}")
            ax.legend(counts.index, title=selected_col, loc="best")
            st.pyplot(fig)
            download_chart(fig, f"piechart_{selected_col}.png")
            plt.close(fig)

            insight = summarize_visuals(
                "Pie Chart",
                stats_categorical(df[selected_col]),
                st.session_state.dataset_summary,
                st.session_state.global_stats
            )
            st.info(insight)

    if selected_col in datetime_cols:
        chart_type = st.selectbox("Chart Type", ["Line Chart", "Area Chart"])
        data_over_time = df.set_index(selected_col).resample("ME").size()

        if chart_type == "Line Chart":
            fig, ax = plt.subplots(figsize=(12, 4))
            data_over_time.plot(ax=ax, color="steelblue", label="Record Count")
            ax.set_title(f"Record Count Over Time ({selected_col})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Count")
            ax.legend()
            st.pyplot(fig)
            download_chart(fig, f"linechart_{selected_col}.png")
            plt.close(fig)

            insight = summarize_visuals(
                "Line Chart",
                stats_time(df[selected_col]),
                st.session_state.dataset_summary,
                st.session_state.global_stats
            )
            st.info(insight)

        elif chart_type == "Area Chart":
            fig, ax = plt.subplots(figsize=(12, 4))
            data_over_time.plot(kind="area", ax=ax, color="steelblue", alpha=0.4, label="Record Count")
            ax.set_title(f"Area Chart Over Time ({selected_col})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Count")
            ax.legend()
            st.pyplot(fig)
            download_chart(fig, f"areachart_{selected_col}.png")
            plt.close(fig)

            insight = summarize_visuals(
                "Area chart",
                stats_time(df[selected_col]),
                st.session_state.dataset_summary,
                st.session_state.global_stats
            )
            st.info(insight)

    # --- Numeric vs Categorical ---
    if numeric_cols and categorical_cols:
        st.markdown("---")
        st.markdown("#### Numeric vs Categorical")
        num_col = st.selectbox("Numeric column", numeric_cols, key="num_cat_num")
        cat_col = st.selectbox("Categorical column", categorical_cols, key="num_cat_cat")
        chart_type2 = st.selectbox("Chart Type", ["Grouped Bar Chart", "Violin by Category"], key="num_cat_chart")

        n_classes = df[cat_col].nunique()
        CATEGORY_LIMIT = 20

        if n_classes > CATEGORY_LIMIT:
            st.warning(
                f"`{cat_col}` has {n_classes} distinct values — too many to "
                f"plot clearly. Showing only an AI summary instead."
            )
            grouped = df.groupby(cat_col)[num_col].mean()
            insight = summarize_visuals(
                chart_type2,
                stats_grouped(grouped),
                st.session_state.dataset_summary,
                st.session_state.global_stats
            )
            st.info(insight)

        elif chart_type2 == "Grouped Bar Chart":
            fig, ax = plt.subplots(figsize=(10, 5))
            grouped = df.groupby(cat_col)[num_col].mean()
            grouped.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", len(grouped)))
            ax.set_title(f"Average {num_col} by {cat_col}")
            ax.set_xlabel(cat_col)
            ax.set_ylabel(f"Average {num_col}")
            plt.xticks(rotation=45)
            ax.bar_label(ax.containers[0], fmt="%.2f")
            palette = sns.color_palette("Set2", len(grouped))
            handles = [mpatches.Patch(color=palette[i], label=cat)
                    for i, cat in enumerate(grouped.index)]
            st.pyplot(fig)
            download_chart(fig, f"groupedbar_{num_col}_by_{cat_col}.png")
            plt.close(fig)

            insight = summarize_visuals(
                "Grouped Bar Chart",
                stats_grouped(grouped),
                st.session_state.dataset_summary,
                st.session_state.global_stats
            )
            st.info(insight)

        elif chart_type2 == "Violin by Category":
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.violinplot(x=df[cat_col], y=df[num_col], ax=ax, palette="Set2")
            ax.set_title(f"{num_col} by {cat_col}")
            ax.set_xlabel(cat_col)
            ax.set_ylabel(num_col)
            palette = sns.color_palette("Set2", len(df[cat_col].unique()))
            handles = [mpatches.Patch(color=palette[i], label=cat)
                    for i, cat in enumerate(df[cat_col].unique())]
            plt.xticks(rotation=45)
            st.pyplot(fig)
            download_chart(fig, f"violin_{num_col}_by_{cat_col}.png")
            plt.close(fig)

            grouped = df.groupby(cat_col)[num_col].mean()
            insight = summarize_visuals(
                "Violin by category",
                stats_grouped(grouped),
                st.session_state.dataset_summary,
                st.session_state.global_stats
            )
            st.info(insight)

    else:
        st.warning("Please select two different columns.")

    # --- Correlation Heatmap ---
    if len(numeric_cols) >= 2:
        st.markdown("---")
        st.markdown("#### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        download_chart(fig, "correlation_heatmap.png")
        plt.close(fig)

        insight = summarize_visuals(
                "Correlation Heatmap",
                stats_corr_matrix(df[numeric_cols].corr()),
                st.session_state.dataset_summary,
                st.session_state.global_stats
            )
        st.info(insight)

    


