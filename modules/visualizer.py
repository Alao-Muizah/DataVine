import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import io

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
        if df[col].nunique() <= 20
    ]
    datetime_cols = []

    for col in df.columns:

        # already datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
            continue

        # only try for object/string columns
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):

            try:
                converted = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)

                success_ratio = converted.notna().mean()

                # LOWER threshold (important fix)
                if success_ratio > 0.3:
                    df[col] = converted   # actually convert it
                    datetime_cols.append(col)

            except Exception:
                pass

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

    elif selected_col in categorical_cols:
        chart_type = st.selectbox("Chart Type", ["Bar Chart", "Pie Chart"])

        if chart_type == "Bar Chart":
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

    # --- Numeric vs Categorical ---
    if numeric_cols and categorical_cols:
        st.markdown("---")
        st.markdown("#### Numeric vs Categorical")
        num_col = st.selectbox("Numeric column", numeric_cols, key="num_cat_num")
        cat_col = st.selectbox("Categorical column", categorical_cols, key="num_cat_cat")
        chart_type2 = st.selectbox("Chart Type", ["Grouped Bar Chart", "Violin by Category"], key="num_cat_chart")

        if chart_type2 == "Grouped Bar Chart":
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
        #    for i, cat in enumerate(grouped.index)]
            st.pyplot(fig)
            download_chart(fig, f"groupedbar_{num_col}_by_{cat_col}.png")
            plt.close(fig)

        elif chart_type2 == "Violin by Category":
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.violinplot(x=df[cat_col], y=df[num_col], ax=ax, palette="Set2")
            ax.set_title(f"{num_col} by {cat_col}")
            ax.set_xlabel(cat_col)
            ax.set_ylabel(num_col)
            palette = sns.color_palette("Set2", len(df[cat_col].unique()))
            handles = [mpatches.Patch(color=palette[i], label=cat)
                       for i, cat in enumerate(df[cat_col].unique())]
        #    for i, cat in enumerate(df[cat_col].unique())]
            plt.xticks(rotation=45)
            st.pyplot(fig)
            download_chart(fig, f"violin_{num_col}_by_{cat_col}.png")
            plt.close(fig)

    # --- Two Numerics ---
    if len(numeric_cols) >= 2:
        st.markdown("---")
        st.markdown("#### Two Numeric Columns")
        col1 = st.selectbox("X axis", numeric_cols, key="x_col")
        col2 = st.selectbox("Y axis", numeric_cols, key="y_col")
        chart_type3 = st.selectbox("Chart Type", ["Regression Plot"], key="two_num_chart")
        if col1 != col2:
                fig, ax = plt.subplots()
                sns.regplot(x=df[col1], y=df[col2], ax=ax, color="steelblue",
                            line_kws={"color": "red", "label": "Regression Line"},
                            scatter_kws={"label": f"{col1} vs {col2}"})
                ax.set_title(f"Regression Plot: {col1} vs {col2}")
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)
                ax.legend()
                st.pyplot(fig)
                download_chart(fig, f"regression_{col1}_vs_{col2}.png")
                plt.close(fig)
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

    


