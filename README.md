# 📊 Loan Risk Prediction Model

## 🧾 Overview
This project aims to build a robust and production-ready machine learning pipeline to predict loan applicant risk (`Risk_Flag`). The pipeline includes:

- Data loading
- Exploratory data analysis (EDA)
- Preprocessing
- Feature engineering
- Model training and evaluation
- Fairness analysis

### 🎯 Goal
Classify applicants as either:
- **1 → Risky** (likely to default)
- **0 → Not Risky** (likely to repay)

This solution is designed to support **risk-based decision-making** in financial institutions.

---

## 📁 Dataset Description
The dataset is divided into **training** and **test sets** (JSON format). Each record represents an individual loan applicant described by demographic and financial attributes.

| Feature              | Description                          |
|----------------------|--------------------------------------|
| `Id`                | Unique identifier                    |
| `Income`            | Income level                         |
| `Age`               | Applicant's age                      |
| `Experience`        | Professional experience (years)      |
| `Married/Single`    | Marital status                       |
| `House_Ownership`   | Home ownership status                |
| `Car_Ownership`     | Car ownership status                 |
| `Profession`        | Occupation                           |
| `CITY`              | City of residence                    |
| `STATE`             | State of residence                   |
| `CURRENT_JOB_YRS`   | Years at current job                 |
| `CURRENT_HOUSE_YRS` | Years at current residence           |
| `Risk_Flag`         | Target (1 = Risky, 0 = Not Risky)    |

---

## ⚙️ Project Workflow

### 1. Data Loading
- Loaded using `pandas`
- Parsed and validated JSON input
- Ensured schema consistency

### 2. Exploratory Data Analysis (EDA)
- Used histograms, boxplots, correlation heatmaps
- Explored feature distributions and class imbalance
- Categorical feature significance via Chi-squared tests

### 3. Data Cleaning & Preprocessing
- Handled missing values and outliers
- Encoded categorical variables with `OneHotEncoder`
- Scaled numerical features (`StandardScaler`)
- Addressed class imbalance with **SMOTE** and **ADASYN**

### 4. Feature Engineering
- Created derived features:
  - Income-to-Age Ratio
  - Employment Bands
  - Housing Duration Bins
- Selected features using correlation and importance scores

### 5. Model Development
Models trained:
- Logistic Regression
- Random Forest
- XGBoost
- Stacking Ensemble

> Tuned with `RandomizedSearchCV` and `StratifiedKFold`

### 6. Evaluation Metrics
- Accuracy
- F1-Score
- Precision
- Recall
- ROC AUC

> ⚠️ **Focus on Accuracy and F1-Score** for final selection

### 7. Bias & Fairness Analysis
- Audited predictions by:
  - Age groups
  - Income levels
  - Location (CITY, STATE)
- Applied fairness strategies like:
  - Equal opportunity checks
  - Fair sampling and thresholding

---

## 📦 Installation & Requirements

```bash
pip install -r requirements.txt
Core Libraries:

pandas, numpy, matplotlib, seaborn

scikit-learn

xgboost

imblearn

▶️ How to Run
bash
Copy
Edit
# Clone repository
git clone <your-repo-url>
cd loan-risk-prediction

# Open notebook
jupyter notebook Loan_risk_prediction_model.ipynb
Follow steps to train models and submit predictions on test set.

📈 Results Summary
Model	Accuracy	F1-Score	ROC AUC
Logistic Regression	61%	25%	62%
Random Forest	86%	84%	97.6%
XGBoost	86%	83%	97.5%
Stacking Ensemble	94%	88%	97.7%

🌟 Highlights
End-to-end ML pipeline

Strong visual insights via seaborn and matplotlib

Stacking ensemble for top performance

Cross-validation and hyperparameter tuning

Bias detection and fairness evaluation

🧠 Final Thoughts
This project simulates a real-world financial risk scoring system, integrating:

Model accuracy,

Interpretability,

Fairness in AI decision-making.

🚀 Future Enhancements:
Add SHAP/LIME for model explainability

Deploy via API

Schedule fairness audits in production

📜 License
This project is released under the MIT License.
