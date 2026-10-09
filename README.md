# E-Commerce Conversion Prediction Challenge

This repository contains the complete machine learning pipeline, dataset files, trained ensemble model, technical project report, and predictions for the **E-Commerce Conversion Prediction Challenge**.

---

## 📌 Project Overview & Description

Online e-commerce platforms generate rich session and demographic data that can be leveraged to predict user conversion. The objective of this project is to build an accurate binary classification model predicting whether a user will convert (`Converted = 1`) or not (`Converted = 0`) during an e-commerce session.

- **Target Variable**: `Converted` (Binary: `0` = No Conversion, `1` = Conversion)
- **Primary Optimization Metric**: **F1-Score**
- **Core Modeling Technique**: Stratified 5-Fold **CatBoost Classifier Ensemble**
- **Decision Threshold Optimization**: Calibrated decision threshold at `0.24`
- **Final Validation F1-Score**: **0.5626** (a **+43.4% relative gain** over baseline Logistic Regression F1 of `0.3923`)
- **Test Predictions**: 1,668 predicted conversions out of 3,000 private test sessions (`55.6%` conversion rate)

---

## 📊 Dataset Information

The datasets used in this project are included directly in this repository:

| Dataset File | Samples | Description |
| :--- | :--- | :--- |
| **[`train.csv`](train.csv)** | 10,000 | Training set containing customer demographic, browsing, and session records with target label `Converted`. |
| **[`public_test.csv`](public_test.csv)** | 3,000 | Validation set used for cross-validation and decision threshold tuning. |
| **[`private_test.csv`](private_test.csv)** | 3,000 | Unlabeled test set evaluated to produce the final competition submission. |

### Key Features
* **Browsing Behavior**: `Pages_Viewed`, `Products_Viewed`, `Time_On_Site`
* **Promotions & History**: `Discount_Seen`, `Previous_Purchases`, `Campaign_Code`
* **User Demographics & Tech**: `Age`, `Income`, `City_Tier`, `Device_Type`, `Traffic_Source`, `Browser_Version`

---

## 🛠️ Technologies Used

* **Programming Language**: Python (3.9+)
* **Machine Learning & Modeling**: [CatBoost](https://catboost.ai/), [Scikit-Learn](https://scikit-learn.org/), [LightGBM](https://lightgbm.readthedocs.io/), [XGBoost](https://xgboost.readthedocs.io/)
* **Data Processing & Scientific Computing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
* **Interactive Environment**: [Jupyter Notebook / IPython](https://jupyter.org/)
* **Documentation & Reporting**: [python-docx](https://python-docx.readthedocs.io/), [ReportLab](https://www.reportlab.com/)

---

## 💡 Feature Engineering Highlights

1. **Browsing Ratios**: Browsing intensity ratios such as `Pages_Per_Product`, `Time_Per_Page`, and `Time_Per_Product`.
2. **Composite Engagement Score**: Multiplicative session depth indicator: `Pages_Viewed * Products_Viewed * (Time_On_Site + 1)`.
3. **Demographic Purchasing Power**: Interaction terms combining income and age (`Income_Age_Prod`, `Income_Age_Ratio`).
4. **Promotion Sensitivity**: Interaction terms between discounts and past loyalty (`Discount_Purchases`, `Discount_Income`).
5. **Informative Missingness Indicators**: Binary indicator flags (`Age_is_missing`, `Income_is_missing`, `Time_On_Site_is_missing`) preserving the predictive signal that missing attributes correlate with lower conversion rates.
6. **Frequency Encoding**: Encoding high-cardinality `Campaign_Code` (6,014 unique categories) using population occurrence frequencies to eliminate data leakage and avoid overfitting.

---

## 📁 Repository Structure

```
├── IqraKhan_ECommerceConversionPrediction.ipynb  # Primary code notebook (named per submission format)
├── notebook.ipynb                                # Standard Jupyter Notebook pipeline
├── IqraKhan_ProjectReport.docx                   # Complete project documentation (.docx)
├── IqraKhan_ECommerceConversionPrediction_Report.docx # Formatted project report (.docx)
├── report.pdf                                    # One-page technical PDF report
├── submission.csv                                # Final predictions for private_test.csv
├── requirements.txt                              # Required Python libraries
├── train.csv                                     # 10,000 training records
├── public_test.csv                               # 3,000 validation records
├── private_test.csv                              # 3,000 test records
├── README.md                                     # Project overview and instructions
└── .gitignore                                    # Git configuration file
```

---

## 🚀 Setup & Execution Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/IqraKhanZ/E-Commerce-Conversion-Prediction.git
cd E-Commerce-Conversion-Prediction
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv env

# Activate virtual environment
# Windows:
env\Scripts\activate
# macOS / Linux:
source env/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook
Launch Jupyter Notebook and execute all cells in sequence:
```bash
jupyter notebook IqraKhan_ECommerceConversionPrediction.ipynb
```
The notebook executes the end-to-end pipeline: data preprocessing, feature engineering, 5-fold CatBoost training, out-of-fold threshold calibration, and generates `submission.csv`.

---

## 📈 Model Performance Summary

| Model | Decision Threshold | F1-Score | Precision | Recall | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline Logistic Regression | 0.50 | 0.3923 | 0.5050 | 0.3210 | 0.6975 |
| XGBoost Classifier | 0.28 | 0.5288 | 0.4120 | 0.7360 | 0.6120 |
| Random Forest Classifier | 0.28 | 0.5476 | 0.4150 | 0.8047 | 0.6073 |
| LightGBM Classifier | 0.27 | 0.5550 | 0.4421 | 0.7453 | 0.6347 |
| **CatBoost Ensemble (Final)** | **0.24** | **0.5626** | **0.4320** | **0.8064** | **0.6168** |

---

## 👤 Author

* **Iqra Khan** — [GitHub Profile](https://github.com/IqraKhanZ)
