# modules/summarizer.py
import os
import re
from dotenv import load_dotenv
from groq import Groq, APIConnectionError, APIStatusError

api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
client = Groq(api_key=api_key)

# --- Dataset Summary ---
def summarize_dataset(df) -> str:
    sample = df.head(8).to_dict(orient="records")
    columns_info = [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]

    metadata = {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": columns_info,
        "sample_rows": sample,
    }

    system_prompt = (
        "You are shown metadata and a sample of rows from a dataset. "
        "In 1-2 sentences, describe what this dataset likely represents: "
        "its domain/subject and what each row seems to represent. "
        "Do not state statistics or numbers. Do not use phrases like "
        "'this dataset shows' or 'this dataset contains'. Be direct and specific."
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(metadata)},
            ],
            temperature=0.3,
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()
    except (APIConnectionError, APIStatusError):
        return "AI summary unavailable right now — couldn't reach Groq."
    except Exception:
        return "AI summary unavailable right now."


# --- Plot Summary ---
def summarize_visuals(chart_type: str, stats: dict, dataset_context: str, global_stats: dict) -> str:
    system_prompt = (
        "You are analyzing a chart from a dataset. Dataset context: "
        f"{dataset_context}\n\n"
        f"Overall dataset statistics (for comparison): {global_stats}\n\n"
        "Write exactly one short paragraph (1-2 sentences) describing the trend "
        "in this chart. Only reference the numbers given.\n\n"
        "STRICT RULES:\n"
        "- Do NOT use a numbered or bulleted list.\n"
        "- Do NOT start with phrases like 'Based on', 'This chart shows', "
        "'The data indicates', or any preamble. Start directly with the finding.\n"
        "- Output only the paragraph, nothing else.\n\n"
        "Example output: 'Fiber customers churn at nearly double the rate of "
        "DSL customers, with the gap widening among month-to-month contracts.'"
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Chart type: {chart_type}\nStats: {stats}"},
            ],
            temperature=0.4,
            max_tokens=100,
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)  # strip stray numbering if it slips through
        return text
    except (APIConnectionError, APIStatusError):
        return "AI insight unavailable right now — couldn't reach Groq."
    except Exception:
        return "AI insight unavailable right now."


# --- Stats Helpers ---
def stats_numeric(series) -> dict:
    s = series.dropna()
    return {
        "mean": round(s.mean(), 2), "median": round(s.median(), 2),
        "std": round(s.std(), 2), "min": round(s.min(), 2), "max": round(s.max(), 2),
        "skew": round(s.skew(), 2),
    }

def stats_categorical(series) -> dict:
    counts = series.value_counts()
    return {
        "top_category": counts.index[0], "top_count": int(counts.iloc[0]),
        "bottom_category": counts.index[-1], "bottom_count": int(counts.iloc[-1]),
        "num_categories": len(counts),
    }

def stats_time(series_over_time) -> dict:
    return {
        "start_value": int(series_over_time.iloc[0]), "end_value": int(series_over_time.iloc[-1]),
        "peak_date": str(series_over_time.idxmax().date()), "peak_value": int(series_over_time.max()),
        "trend": "increasing" if series_over_time.iloc[-1] > series_over_time.iloc[0] else "decreasing",
    }

def stats_grouped(grouped_series) -> dict:
    return {
        "highest_group": grouped_series.idxmax(), "highest_value": round(grouped_series.max(), 2),
        "lowest_group": grouped_series.idxmin(), "lowest_value": round(grouped_series.min(), 2),
    }

def stats_two_numeric(df, col1, col2) -> dict:
    corr = df[col1].corr(df[col2])
    return {"correlation": round(corr, 2), "direction": "positive" if corr > 0 else "negative"}

def stats_corr_matrix(corr_df) -> dict:
    pairs = corr_df.where(~corr_df.isna() & (corr_df != 1.0)).unstack().dropna()
    strongest = pairs.abs().idxmax()
    return {"strongest_pair": strongest, "strongest_value": round(pairs[strongest], 2)}
