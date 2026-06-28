# Loan Status Prediction using Support Vector Machine (SVM)

A beginner-friendly, end-to-end Machine Learning project that predicts whether a loan application will be **approved** or **rejected** based on applicant information.

---

## Project Overview

Financial institutions receive thousands of loan applications every day. Manually evaluating each one is time-consuming and error-prone. This project builds a **Support Vector Machine (SVM)** classifier trained on historical applicant data to automate and assist loan approval decisions.

The model achieves **81.6% accuracy** on unseen test data.

---

## Features

- Clean exploratory data analysis (EDA) with visualisations
- Handles missing values using mode and median imputation
- Encodes categorical variables with `LabelEncoder`
- Scales features using `StandardScaler` (required for SVM)
- Trains an SVM with an RBF kernel for non-linear classification
- Evaluates with Accuracy Score, Confusion Matrix, and Classification Report
- Saves the trained model and scaler for future inference
- Beginner-friendly comments throughout the notebook

---

## Folder Structure

```
Loan_Status_Prediction/
│
├── dataset/
│   └── loan_status.csv          # Raw dataset
│
├── output/
│   ├── confusion_matrix.png     # Confusion matrix heatmap
│   ├── feature_distribution.png # EDA visualisation
│
├── Loan_Status_Prediction.ipynb # Main Jupyter Notebook
└── README.md                    # Project documentation
```

---

## Libraries Used

| Library | Purpose |
|---|---|
| `numpy` | Numerical operations |
| `pandas` | Data loading and manipulation |
| `matplotlib` | Plotting and visualisation |
| `seaborn` | Statistical visualisations |
| `scikit-learn` | ML model, preprocessing, evaluation |
| `joblib` | Saving and loading the trained model |

---

## Dataset Description

**File:** `dataset/loan_status.csv`  
**Records:** 614 rows | **Columns:** 13 (12 features + 1 target)

| Column | Type | Description |
|---|---|---|
| Loan_ID | ID | Unique loan identifier (dropped during training) |
| Gender | Categorical | Male / Female |
| Married | Categorical | Applicant marital status (Yes / No) |
| Dependents | Numerical | Number of dependents (0, 1, 2, 3+) |
| Education | Categorical | Graduate / Not Graduate |
| Self_Employed | Categorical | Yes / No |
| ApplicantIncome | Numerical | Monthly income of the applicant |
| CoapplicantIncome | Numerical | Monthly income of the co-applicant |
| LoanAmount | Numerical | Loan amount requested (in thousands) |
| Loan_Amount_Term | Numerical | Term of the loan (in months) |
| Credit_History | Binary | 1 = good credit history, 0 = bad |
| Property_Area | Categorical | Urban / Semiurban / Rural |
| **Loan_Status** | **Target** | **Y = Approved, N = Rejected** |

**Missing Values:** Present in Gender, Married, Dependents, Self_Employed, LoanAmount, Loan_Amount_Term, Credit_History — all handled during preprocessing.

---

## Installation Steps

1. **Clone or download** the project folder.

2. **Install dependencies:**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

3. **Place the dataset** inside the `dataset/` folder:
```
dataset/loan_status.csv
```

---

## ▶How to Run the Notebook

```bash
jupyter notebook Loan_Status_Prediction.ipynb
```

Run all cells from top to bottom (`Cell → Run All`).  
The outputs (plots and saved model) will appear in the `output/` folder.

---

## Model Used — Support Vector Machine (SVM)

A **Support Vector Machine** is a supervised learning algorithm that finds the **optimal hyperplane** separating two classes in feature space. It maximises the **margin** between the nearest data points (support vectors) of each class.

We use the **RBF (Radial Basis Function) kernel**, which maps data into a higher-dimensional space to handle non-linearly separable patterns.

| Parameter | Value | Description |
|---|---|---|
| `kernel` | `rbf` | Radial Basis Function — handles non-linear boundaries |
| `C` | `1.0` | Regularisation — penalty for misclassified points |
| `gamma` | `scale` | Influence radius of a single training point |

> **Why scale features?** SVM calculates distances between data points. Without scaling, features with large values (e.g., `ApplicantIncome`) would dominate over smaller ones (e.g., `Credit_History`), hurting model performance.

---

## Evaluation Metrics

| Metric | Value |
|---|---|
| **Accuracy** | **81.6%** |


## Sample Output

### Feature Distribution by Loan Status
Visualises how each feature (Gender, Married, Credit History, Loan Amount, etc.) is distributed across approved and rejected applications.
> See: `output/feature_distribution.png`


---

## Future Improvements

- **Hyperparameter tuning** with `GridSearchCV` to find the best `C` and `gamma`
- **Cross-validation** for more robust performance estimation
- **Try other kernels** (linear, poly, sigmoid) and compare results
- **Feature engineering** — e.g., Total Income = ApplicantIncome + CoapplicantIncome
- **Handle class imbalance** using SMOTE oversampling (approvals outnumber rejections ~2.2:1)
- **Deploy the model** as a web application using Flask or Streamlit

---
