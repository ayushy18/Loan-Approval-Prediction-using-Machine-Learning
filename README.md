# 💰 Loan Approval Prediction using Machine Learning 🤖

> 🏦 **Predicting whether a loan application gets approved or not** using machine learning — a smart way to assist banks and financial institutions in decision-making!  
> This project walks you through **data analysis, feature preprocessing, model building, and evaluation** using Python and Scikit-learn.  

---

## 🧭 Table of Contents
- [✨ Overview](#-overview)
- [📂 Dataset Details](#-dataset-details)
- [📘 Notebook Workflow](#-notebook-workflow)
- [🔍 Data Preprocessing & EDA](#-data-preprocessing--eda)
- [⚙️ Machine Learning Models](#️-machine-learning-models)
- [🧪 Results Summary](#-results-summary)
- [▶️ Run the Project](#️-run-the-project)
- [📁 Project Structure](#-project-structure)
- [🧰 Requirements](#-requirements)
- [🚀 Future Improvements](#-future-improvements)
- [📝 Credits & References](#-credits--references)
- [📜 License & Contact](#-license--contact)

---

## ✨ Overview

This project demonstrates how **Machine Learning** can automate the process of **loan approval prediction** — helping banks assess loan eligibility faster and more accurately.  
The model predicts whether a loan will be **Approved (Y)** or **Not Approved (N)** based on applicant features such as income, employment type, loan amount, and credit history.  

🎯 **Goal:** Build and compare ML models to predict loan approval status.  
💡 **Key Learning:** Data preprocessing, feature encoding, model comparison, and evaluation.

---

## 📂 Dataset Details

📄 **File:** `LoanApprovalPrediction.csv`  
The dataset contains 13 columns describing the applicant and loan attributes:

| Feature | Description |
|----------|-------------|
| Gender | Applicant gender |
| Married | Marital status |
| Dependents | Number of dependents |
| Education | Graduate / Non-Graduate |
| Self_Employed | Employment type |
| ApplicantIncome | Income of the applicant |
| CoapplicantIncome | Income of co-applicant |
| LoanAmount | Loan amount in thousands |
| Loan_Amount_Term | Term of loan in months |
| Credit_History | History of credit repayment |
| Property_Area | Urban / Semi-urban / Rural |
| Loan_Status | Target (Y/N) |
| Loan_ID | Identifier (dropped) |

📚 *Dataset inspired by [GeeksforGeeks: Loan Approval Prediction using ML](https://www.geeksforgeeks.org/machine-learning/loan-approval-prediction-using-machine-learning/)*

---

## 📘 Notebook Workflow

📓 File: `Loan_Approval_Prediction_using_Machine_Learning.ipynb`

The notebook performs the complete ML pipeline:

1. 🧹 **Data Cleaning** — Handle missing values, remove irrelevant columns.  
2. 🔠 **Encoding** — Convert categorical data into numerical form using Label Encoding.  
3. 📊 **Exploratory Data Analysis (EDA)** — Visualize distributions & correlations.  
4. ⚙️ **Model Training** — Fit and test multiple ML models.  
5. 📈 **Evaluation** — Compare model accuracy and performance.

---

## 🔍 Data Preprocessing & EDA

- Dropped `Loan_ID` since it's not predictive.  
- Handled missing values using **mean imputation**.  
- Encoded categorical variables with **LabelEncoder**.  
- Visualized key relationships:
  - 🔸 Credit history vs Loan Status (strong correlation ✅)
  - 🔸 Income distribution among approved vs rejected loans
  - 🔸 Property area’s impact on approval rates

📊 **Insight:** Applicants with a good credit history and higher income have a higher chance of loan approval.

---

## ⚙️ Machine Learning Models

The following ML models were implemented and evaluated:

| Algorithm | Description |
|------------|--------------|
| 🧩 K-Nearest Neighbors (KNN) | Simple distance-based classifier |
| 🌲 Random Forest Classifier | Ensemble of decision trees (best performer) |
| 🔹 Support Vector Classifier (SVC) | Classifies using optimal hyperplane |
| 🧮 Logistic Regression | Statistical model for binary classification |

---

## 🧪 Results Summary

| Model | Test Accuracy | Remarks |
|--------|----------------|----------|
| 🌲 Random Forest | **82.5%** ✅ | Best performer |
| 🧮 Logistic Regression | 80.8% | Consistent and interpretable |
| 🔹 SVC | 69.1% | Needs parameter tuning |
| 🧩 KNN | 63.7% | Sensitive to scaling |

📈 **Conclusion:** Random Forest gave the best balance of accuracy and interpretability for this dataset.

---

## ▶️ Run the Project

### 🟢 Option 1: Run in Google Colab
1. Open [Google Colab](https://colab.research.google.com/).  
2. Upload both:
   - `Loan_Approval_Prediction_using_Machine_Learning.ipynb`
   - `LoanApprovalPrediction.csv`
3. Run all cells sequentially. ✅

💡 *You can also add an “Open in Colab” badge later for direct launch.*

---

### 💻 Option 2: Run Locally
```bash
# Clone the repo
git clone https://github.com/<your-username>/Loan-Approval-Prediction.git
cd Loan-Approval-Prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # (Linux/macOS)
venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

---

## 📁 Project Structure

```
Loan-Approval-Prediction/
├── LoanApprovalPrediction.csv
├── Loan_Approval_Prediction_using_Machine_Learning.ipynb
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 🧰 Requirements

📦 Install the dependencies below (include this in `requirements.txt`):

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 🚀 Future Improvements

✨ Make it even better:
- 🔁 Apply **One-Hot Encoding** instead of Label Encoding.  
- ⚖️ Add **class imbalance handling** (SMOTE, class weights).  
- 🎯 Use **k-fold Cross Validation** for more robust performance.  
- 🧠 Tune hyperparameters using **GridSearchCV**.  
- ⚡ Try **XGBoost**, **LightGBM**, or **CatBoost**.  
- 🌐 Deploy a **Streamlit Web App** for live loan predictions.  
- 🧾 Add **model interpretability** via SHAP / LIME.  

---

## 📝 Credits & References

🙏 **Inspired by:**  
📘 [GeeksforGeeks: Loan Approval Prediction using Machine Learning](https://www.geeksforgeeks.org/machine-learning/loan-approval-prediction-using-machine-learning/)  

💻 **Developed by:** [Your Name]  
📅 **Year:** 2025  

---

## 🧾 Footer
Developed with ️
👨‍💻 Team Members:

Ayush Yadav
🎓 B.Tech Computer Science (Data Science & AI)
202210101150081
Suyash Sharma
🎓 B.Tech Computer Science (Data Science & AI)
202210101150076
---

🎉 *Thanks for visiting this project!*  
