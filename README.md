# E-Commerce Conversion Prediction Challenge (Summer Analytics 2026)

This repository contains the final machine learning pipeline, predictions, and report for the E-Commerce Conversion Prediction Challenge. The goal of this challenge is to predict whether a customer will convert (binary target `Converted`) during their e-commerce session.

## Project Overview

- **Objective**: Maximize the classification F1-Score on unseen customer sessions.
- **Model**: Stratified 5-Fold CatBoost Classifier Ensemble.
- **Optimal Decision Threshold**: `0.24` (tuned Out-of-Fold to handle class imbalance).
- **Validation F1-Score**: `0.5626` (a ~43.4% relative improvement over the default 0.50 threshold F1 of 0.3923).
- **Predictions**: 1,668 predicted conversions out of 3,000 test sessions (55.6% conversion rate).

## Repository Contents

* **`notebook.ipynb`**: A clean, fully reproducible Jupyter Notebook containing the data loading, feature engineering, cross-validation, hyperparameter choices, decision threshold optimization, and final test set predictions.
* **`submission.csv`**: The final predictions for the 3,000 private test samples in the required `User_ID` and `Converted` format.
* **`report.pdf`**: A clean, one-page technical summary of the methodology, key exploratory insights, feature engineering choices, and results.
* **`train.csv`**: Training dataset of 10,000 rows.
* **`public_test.csv`**: Public test/validation set of 3,000 rows (used during training & threshold selection).
* **`private_test.csv`**: Private test dataset of 3,000 rows for generating the final predictions.
* **`requirements.txt`**: List of Python packages required to run the notebook.
* **`.gitignore`**: Git configuration to exclude temporary files and python caches.

## Feature Engineering Highlights

We engineered specific behavioral and interaction features to capture session depth, demographic purchasing power, and responses to promotions:
1. **Ratio Features**: Browsing intensity ratios, such as `Pages_Per_Product`, `Time_Per_Page`, and `Time_Per_Product`.
2. **Engagement Score**: Session depth metric calculated as `Pages_Viewed * Products_Viewed * (Time_On_Site + 1)`.
3. **Demographic & Discount Interactions**: Capturing purchasing power (`Income_Age_Prod`, `Income_Age_Ratio`) and discount sensitivity (`Discount_Purchases`, `Discount_Income`).
4. **Missing Value Flags**: Informative binary flags (`Age_is_missing`, `Income_is_missing`, `Time_On_Site_is_missing`) to capture session/user engagement cues since missingness strongly correlated with non-conversion.
5. **High-Cardinality Encoding**: Frequency-based encoding of `Campaign_Code` (6,014 unique values) using the full population count, capturing campaign size without introducing target leakage.

## How to Setup and Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/IqraKhanZ/E-Commerce-Conversion-Prediction.git
   cd E-Commerce-Conversion-Prediction
   ```

2. **Install Dependencies**:
   It is recommended to use a virtual environment:
   ```bash
   python -m venv env
   # On Windows:
   env\Scripts\activate
   # On macOS/Linux:
   source env/bin/activate

   pip install -r requirements.txt
   ```

3. **Run the Notebook**:
   Open Jupyter Notebook or JupyterLab and execute all cells in `notebook.ipynb` to reproduce the model training, threshold tuning, and predictions:
   ```bash
   jupyter notebook notebook.ipynb
   ```
