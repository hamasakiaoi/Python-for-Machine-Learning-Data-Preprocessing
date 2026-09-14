# AI Pioneers Internship — Week 1

## Python for Machine Learning & Data Preprocessing

This project is a reproducible preprocessing and exploratory-analysis study of UCI's Adult Census Income dataset. It prepares a clean, machine-learning-ready tabular dataset without training a model. The focus is on evidence-led preprocessing decisions: quality assessment, missing values, nominal encoding, feature selection, and numerical scaling.

### Internship objectives

1. Understand the fundamentals of Machine Learning using Python.
2. Learn data loading, cleaning, handling missing values, feature selection, encoding categorical variables, normalization, and exploratory data analysis using Pandas and NumPy.
3. Deliverable: Clean and preprocess a sample dataset and document each preprocessing step.

## Dataset

**Adult Census Income** (UCI Machine Learning Repository) contains demographic and employment attributes from 1994 U.S. Census Bureau data, with the income label indicating whether income exceeds $50,000.

- Source: [UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)
- Provenance: donated to UCI by Ronny Kohavi and Barry Becker; based on U.S. Census Bureau data.
- Raw files: `adult.data` and `adult.test`, retained unchanged in `data/raw/`.
- Observed after combining files: 48,842 rows × 15 columns; numeric and categorical variables; `?` markers in three categorical fields.

The raw source is included solely to make this educational project reproducible. Refer to the UCI source for dataset terms and attribution.

## Workflow and decisions

`Raw UCI data → exact-duplicate removal + fnlwgt removal → feature selection → median/mode imputation + one-hot encoding + standardization → processed CSV`

- **Duplicates:** remove only exact duplicate records.
- **Sampling weight (`fnlwgt`):** remove; it represents survey weighting rather than an individual characteristic.
- **Missing values:** parse `?` as missing. Impute retained nominal fields with their most frequent category, avoiding unnecessary row loss.
- **Categoricals:** one-hot encode nominal fields; label encoding would imply an artificial ranking.
- **Feature selection:** remove `education_num` as redundant with `education`; remove `native_country` because it is imbalanced, partially missing, and not required for this compact baseline.
- **Scaling:** standardize four numerical predictors; do not scale categorical indicators or the label.

## Project structure

```text
AI_Pioneers_Week1_ML_Preprocessing/
├── data/raw/                         # Original UCI source files
├── data/processed/                   # Cleaned checkpoint and final CSV
├── docs/                             # Formal internship report
├── notebooks/week1_preprocessing_eda.ipynb
├── outputs/figures/                  # Decision-oriented charts
├── outputs/tables/                   # Auditable before/after summaries
├── src/                              # Modular pipeline code
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup and execution

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.pipeline
python -m src.report
python -m unittest discover -s tests -v
```

Run these commands from the project root. The pipeline fails clearly if the two required raw UCI files are absent. If obtaining a fresh copy, download `adult.data` and `adult.test` from the UCI link above into `data/raw/`; do not rename them.

The standard-library test command provides focused regression coverage for duplicate removal, the removal of the survey-weight field, missing-value imputation, and numerical scaling.

To use the explanatory notebook:

```bash
jupyter notebook notebooks/week1_preprocessing_eda.ipynb
```

## Outputs

- `data/processed/adult_census_income_processed.csv` — 62 transformed features plus the income label.
- `outputs/tables/pipeline_before_after.csv` — stage-by-stage evidence.
- `outputs/tables/scaling_before_after.csv` — numerical statistics before and after standardization.
- `outputs/figures/` — missingness, distribution, category-frequency, and correlation plots.
- `docs/Week_1_ML_Data_Preprocessing_Report.docx` — submission-ready report.

## Limitations

This is a preprocessing project, not a model evaluation. Most-frequent imputation is intentionally simple and may increase the modal category. The historical census data can also reflect social and sampling biases; downstream use requires appropriate fairness and validation work.

## Technical references

- [UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)
- [scikit-learn preprocessing documentation](https://scikit-learn.org/stable/modules/preprocessing.html)
