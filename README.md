#  DataVine

> An end-to-end data platform for cleaning, exploring, visualizing, and modeling your datasets — no code required.

[[Access Streamlit App]](https://datavine-haziumxyzqr.streamlit.app)

---

## What is DataVine?

DataVine is a web-based data platform that enables users to upload, clean, visualize, and apply machine learning models to their data — all in one integrated environment. Designed for both beginners and experienced data practitioners, it offers an intuitive interface with powerful automated workflows.

---

## Workflow

```
Upload Dataset → Inspect → Clean → Visualize → Train Model → Export Results
```

### 1. Upload
- Supports CSV and Excel (.xlsx) files
- Instant preview on upload

### 2. Inspect
- Dataset shape (rows × columns)
- Column types grouped by category (Numeric, Categorical, Datetime, Boolean)
- Missing value counts and percentages
- Duplicate row detection
- Summary statistics

### 3. Clean
DataVine scans your dataset before touching anything and reports:
- Columns with missing values
- High cardinality columns (unique identifiers)
- Identifier columns (emails, phone numbers, IDs)

For each issue, you decide:
- **Missing values** → Fill (Median / Mean / Mode / Custom) or Drop rows
- **High cardinality** → Drop or Keep
- **Identifiers** → Drop or Keep

Nothing is changed until you click **Apply Cleaning**.

### 4. Visualize
Automated chart generation based on column types:

| Column Type | Available Charts |
|---|---|
| Numeric | Histogram, Box Plot, Violin Plot |
| Categorical | Bar Chart, Pie Chart |
| Numeric vs Categorical | Grouped Bar Chart, Violin by Category |
| Two Numerics | Regression Plot |
| All Numerics | Correlation Heatmap |

Every chart includes a download button.

### 5. Train
Select a task and train a model on your cleaned data:

**Regression**
- Linear Regression
- Polynomial Regression
- Decision Tree
- Random Forest
- XGBoost

**Classification**
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- SVM

**Clustering**
- K-Means
- DBSCAN
- Hierarchical (Agglomerative)
  

Each model includes hyperparameter tuning, performance metrics, feature importance charts, and a downloadable trained model (.pkl).

### 6. Export
- Download cleaned dataset as CSV
- Download any chart as PNG
- Download trained model as .pkl
- Download forecast results as CSV

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Model Serialization | Joblib |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Alao-Muizah/DataVine.git
cd DataVine

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Project Structure

```
DataVine/
│
├── app.py                  # Main Streamlit entry point
├── requirements.txt
├── README.md
│
└── modules/
    ├── loader.py           # File upload and loading
    ├── inspector.py        # Data inspection
    ├── cleaner.py          # Data cleaning pipeline
    ├── visualizer.py       # Chart generation
    ├── trainer.py          # Task router
    ├── regression.py       # Regression models
    ├── classification.py   # Classification models      
    └── clustering.py       # Clustering models
```

---

## Status

Active development — v1.0

---

## Author

**Muizah Alao** — [GitHub](https://github.com/Alao-Muizah)
