import streamlit as st
import matplotlib.pyplot as plt
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
        if df[col].nunique() < len(df)
    ]
    categorical_cols = [
        col for col in df.select_dtypes(include="object").columns.tolist()
        if df[col].nunique() <= 20
    ]
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

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
            counts.plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title(f"Bar Chart of {selected_col}")
            ax.set_xlabel(selected_col)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45)
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

    elif selected_col in datetime_cols:
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

    


