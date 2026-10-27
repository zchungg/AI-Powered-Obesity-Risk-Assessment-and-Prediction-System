# AI-Powered Obesity Risk Assessment and Prediction System

This repository implements the proposed Sunway University capstone system using the UCI **Estimation of Obesity Levels Based On Eating Habits and Physical Condition** dataset.

## What is implemented

- 80/20 stratified train-test split
- Standard scaling for continuous variables
- Explicit binary/ordinal/categorical encoding
- SMOTE inside the imbalanced-learn pipeline so resampling is confined to training folds
- Model-specific hybrid Borda-count feature selection using:
  - Chi-Square
  - ANOVA F-test
  - RFE using each candidate classifier
  - Initial SHAP global-importance rankings for each candidate classifier
- Model benchmarking:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - CatBoost
  - LightGBM
  - Stacking Ensemble
  - Artificial Neural Network (scikit-learn MLP implementation)
- Accuracy, Macro F1, Log Loss, classification reports and confusion matrices
- Randomized hyperparameter tuning of the two strongest baseline candidates using five-fold stratified cross-validation
- Comparison of every baseline and tuned candidate using cross-validation Macro F1
- Individual metric bar charts for every baseline and tuned model
- SHAP local explanation for the deployed model
- Natural-language explanation labels
- Lifestyle recommendation engine
- Modular Streamlit website in `website_app/` with Home, Risk Assessment, explanations, summary, model comparison and About pages
- Saved model pipeline and metadata for real-time inference

## Project structure

```text
obesity-risk-system/
├── app.py                         # Compatibility launcher
├── download_data.py
├── requirements.txt
├── README.md
├── data/
│   └── ObesityDataSet_raw_and_data_sinthetic.csv
├── models/
│   ├── best_pipeline.joblib
│   └── model_metadata.json
├── results/
│   ├── model_comparison.csv
│   ├── feature_ranking_<model>.csv
│   ├── confusion_matrix_*.png
│   └── model_metrics_*.png         # Per-model metric bar charts
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── models.py
│   ├── predict.py
│   ├── explain.py
│   ├── recommendations.py
│   └── train.py
└── website_app/
    ├── app.py                      # Website entry point
    ├── shared.py                   # Shared layout and result display
    └── pages/                      # One module per website page
```

## Setup

Open a terminal inside the project folder. Create the virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell (activation may be blocked by your Execution Policy):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Or skip activation completely and use the virtual environment Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Download the UCI data in Windows PowerShell:

```bash
.\.venv\Scripts\python.exe download_data.py
```

Train and benchmark the models in Windows PowerShell:

```bash
.\.venv\Scripts\python.exe -m src.train
```

Launch the modular website app:

```bash
.\.venv\Scripts\python.exe -m streamlit run website_app/app.py
```

The legacy root entry point remains available with
`.\.venv\Scripts\python.exe -m streamlit run app.py`.

Each candidate pipeline independently computes feature rankings using that
candidate's own estimator for RFE and SHAP, combines them with Chi-Square and
ANOVA F-test using Hybrid Borda count, selects its features, and trains that model
on those selected features. The resulting feature tables are saved as
`results/feature_ranking_<model>.csv`. The two strongest baseline candidates by
training-set cross-validation Macro F1 are tuned with randomized search and
five-fold stratified CV. Each search explicitly includes the baseline model's
original hyperparameters alongside the random candidates, and compares every
candidate using the same folds and Macro F1 scoring. SMOTE and feature selection
remain inside each fold. The tuned rows in `results/model_comparison.csv` include
`baseline_cv_macro_f1` for comparison. The comparison CSV and Model Comparison
page include every baseline model and both tuned candidates. Deployment is
selected from all baseline and tuned candidates using CV Macro F1; held-out test
metrics do not determine the winner. Per-model metric charts are saved as
`results/model_metrics_<model>.png` and shown on the Model Comparison page.

The website presents a categorical model risk group instead of a single confidence
percentage: Insufficient Weight maps to **Low risk**, Normal Weight to **No risk
(normal weight)**, and Overweight/Obesity categories to **High risk**. These are
groupings of the model's classification output, not clinical risk estimates.

### Windows PowerShell

If PowerShell says `running scripts is disabled`, you can skip activation and run: 

```powershell
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Notes for the capstone report

The deployed model is selected from the SHAP-compatible candidates when possible because explainability is a core system requirement. The selected model, metrics, feature ranking and comparison results are saved under `models/` and `results/`.

The current ANN implementation uses scikit-learn `MLPClassifier` to keep the complete pipeline lightweight and portable. The project can be changed to TensorFlow/Keras later if an exact Batch Normalization + Dropout architecture is required by the final implementation or assessment rubric.

### Target-label handling

The UCI CSV stores class names with underscores, for example `Obesity_Type_I`. The training code converts these to integer IDs 0–6 for model compatibility (especially XGBoost), while the application displays the human-readable category names. This also keeps confusion matrices and SHAP class indexing consistent.

The UCI dataset used here classifies seven obesity/weight categories. Avoid describing the target as a future clinical disease diagnosis. Use wording such as **predicted obesity/weight category** or **model-based obesity classification**.
