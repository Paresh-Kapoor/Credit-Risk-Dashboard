# **🏦 Credit Risk Prediction — Loan Default Classification System**

**An end-to-end Machine Learning pipeline and interactive dashboard that predicts loan defaults and mathematically explains its reasoning to ensure regulatory compliance.**

## **🏢 Business Context: Why Banks Need This**

In the BFSI sector, minimizing loan defaults is critical to maintaining liquidity. However, simply building a highly accurate "Black Box" AI is no longer acceptable. Regulators (like the RBI) require **Explainable AI (XAI)**. If a customer is denied a loan, the bank must be able to explain exactly *why* to avoid discriminatory lending practices.

This project solves both problems: it uses a powerful tuned **XGBoost** algorithm to accurately flag risky borrowers, and utilizes **SHAP (Game Theory)** to provide loan officers with transparent, human-readable reasons for every single decision.

## **📊 Model Performance Results**

To combat the Accuracy Paradox inherent in imbalanced datasets (80% Paid / 20% Default), we utilized **SMOTE** oversampling and tracked **Recall** and **AUC-ROC** as our primary business metrics. Missing a defaulter (False Negative) is financially much more expensive than wrongly rejecting a safe applicant (False Positive).

| Model | AUC-ROC | F1 Score | Precision | Recall |
| :---- | :---- | :---- | :---- | :---- |
| Logistic Regression (Baseline) | 0.7241 | 0.4441 | 0.3417 | 0.6344 |
| Random Forest | 0.7036 | 0.3544 | 0.4196 | 0.3067 |
| **XGBoost (Tuned Final)** | **0.7105** | **0.2820** | **0.4657** | **0.2022** |

*(Note: XGBoost hyperparameters were tuned via 5-fold GridSearchCV to balance precision and recall on unseen test data).*

## **⚙️ How It Works (The Pipeline)**

1. **SQL Data Ingestion:** Processed 500k+ rows of real Lending Club data. Stored and queried via SQLite.  
2. **Feature Engineering:** Created robust financial metrics like loan\_to\_income\_ratio and safely imputed missing values using medians.  
3. **Handling Imbalance:** Applied SMOTE (Synthetic Minority Over-sampling Technique) strictly to the training split.  
4. **Hyperparameter Tuning:** Executed GridSearchCV on XGBoost parameters to find the mathematical peak.  
5. **Interactive Dashboard:** Deployed a Streamlit web app allowing underwriters to input applicant data and instantly receive a risk score and SHAP justification.

## **🧠 Model Explainability (SHAP)**

This system goes beyond predictions by answering "Why?". Using shap.TreeExplainer, the model generates personalized Waterfall charts for every applicant.

**Key Insights Discovered:**

* **Interest Rate:** Evaluated as the \#1 driving factor for default risk across the portfolio.  
* **Debt-to-Income (DTI):** High DTI drastically increased risk, validating standard banking assumptions.  
* **Employment Length:** Surprisingly, the model discovered that longer employment history had a minimal dampening effect on overall risk compared to actual requested loan amounts.

## **🚀 How to Run Locally**

1. **Clone the repository:**  
   git clone \[https://github.com/Paresh-Kapoor/Credit-Risk-Dashboard.git\](https://github.com/Paresh-Kapoor/Credit-Risk-Dashboard.git)  
   cd Credit-Risk-Dashboard

2. **Install dependencies:**  
   pip install \-r requirements.txt

3. **Run the Streamlit Dashboard:**  
   streamlit run app/app.py

## **💻 Tech Stack**

The following libraries and technologies were utilized to build this project:

* **Streamlit** (Interactive web deployment)  
* **XGBoost & Scikit-learn** (Machine Learning & Modeling)  
* **SHAP** (Explainable AI / Game Theory)  
* **Pandas & NumPy** (Data Manipulation)  
* **Matplotlib** (Data Visualization)  
* **SQLite** (Database management)

*Built by [Paresh Kapoor](https://www.linkedin.com/in/pareshkapoor)*