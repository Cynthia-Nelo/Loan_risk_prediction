📊 Loan Risk Prediction Model
A Machine Learning Solution for Fair and Transparent Loan Approval Decisions

https://img.shields.io/github/last-commit/Cynthia-Nelo/loan-risk-prediction
https://img.shields.io/github/repo-size/Cynthia-Nelo/loan-risk-prediction
https://img.shields.io/badge/Python-3.8%252B-blue

🚀 Project Overview
Financial institutions need data-driven, fair, and explainable AI models to assess loan applicant risk. This project delivers a production-ready machine learning pipeline that:

✅ Predicts whether an applicant is likely to default (Risk_Flag = 1)
✅ Handles class imbalance using advanced resampling techniques
✅ Ensures fairness across sensitive attributes (Age, Income, Location)
✅ Optimizes for recall to minimize risky loan approvals

Built with Python, Scikit-learn, stacking ensemble, and fairness-aware techniques, this solution is designed for real-world lending scenarios.

🔍 Key Features
📈 High-Performance Predictive Modeling
Trained and optimized Logistic Regression, Random Forest, XGBoost, and Stacking Ensemble

F2-Score Optimization to prioritize recall of risky applicants

ROC AUC: 0.98 | Recall (Risk_Flag=1): 0.88

⚖️ Bias Mitigation & Fairness
Subgroup fairness analysis (Age, Income, City, State)

Smote and ADASYN oversampling to handle class imbalance

Fairness metrics tracked: Demographic Parity, Equalized Odds, Disparate Impact Ratio

🔄 End-to-End Pipeline
Data Cleaning & EDA (Missing value handling, outlier detection)

Feature Engineering (Age bands, income groups, target encoding)

Model Training & Fairness Auditing

Explainability (SHAP, LIME, model cards)

📂 Repository Structure
bash
loan-risk-prediction/  
├── data/                    # Raw and processed datasets  
├── notebooks/               # Jupyter notebooks (EDA, modeling)  
│   ├── 01_Data_Cleaning_EDA_&_ Feature_Engeenering.ipynb    
│   └── 02_Model_Training_Evaluation.ipynb  
├── src/                     # Python scripts for pipeline  
│   ├── preprocessing.py  
│   ├── train.py  
│   └── fairness_analysis.py  
├── models/                  # Saved model artifacts  
├── docs/                    # Fairness reports & model cards  
├── requirements.txt         # Python dependencies  
└── README.md  
🛠️ Installation & Usage
1. Clone the Repository
bash
git clone https://github.com/yourusername/loan-risk-prediction.git  
cd loan-risk-prediction  
2. Set Up a Virtual Environment
bash
python -m venv venv  
source venv/bin/activate  # Linux/Mac  
.\venv\Scripts\activate   # Windows  
3. Install Dependencies
bash
pip install -r requirements.txt  
4. Run the Pipeline
bash
python src/train.py  # Trains model and generates fairness report  
📊 Results & Fairness Insights
Model	F2-Score	ROC AUC	Recall (Risk=1)
XGBoost	0.82	0.92	0.85
Stacking	0.81	0.91	0.83
Fairness Audit Highlights:

No significant bias across age groups (p > 0.05)

<5% disparity in approval rates between income brackets

📜 License
This project is licensed under MIT. See LICENSE for details.

💡 Built for Responsible AI in Finance
https://via.placeholder.com/600x200?text=SHAP+Analysis+Example

🔗 Connect with me: LinkedIn | Portfolio

