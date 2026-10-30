# 🏠 House Price Prediction System

An end-to-end Machine Learning solution designed to predict residential property prices based on housing attributes, structural dimensions, and quality metrics from the Ames Housing dataset. The project includes full exploratory data analysis (EDA), data preprocessing, feature encoding, hyperparameter-tuned gradient boosting, and an interactive Streamlit web dashboard.

---

## 📌 Project Overview
- **Objective:** Accurate regression modeling to estimate house valuation based on 70+ raw structural and demographic attributes.
- **Trained Model:** Optimized `GradientBoostingRegressor` evaluated against metrics like $R^2$ Score and Mean Absolute Error (MAE).
- **Interactive UI:** Dynamic web application powered by **Streamlit** for real-time user input and automated inference.
- **Robust Pipeline:** Resolves categorical feature mismatch dynamically via encoded feature alignment and missing-value imputation.

---

## 🚀 Key Features
* **Extensive EDA & Data Cleaning:** Missing value imputation, log-transformation of skewed numerical fields, and handling categorical ordinal/nominal features.
* **One-Hot Encoding & Alignment:** Automated schema alignment handling test-time category mismatches across 479 encoded features.
* **Model Benchmarking:** Comparative evaluation across Linear Regression, Ridge, Lasso, SVR, Random Forest, AdaBoost, XGBoost, and Gradient Boosting.
* **Web Deployment:** User-friendly interface with dual-column inputs, default parameter bounds, and instantaneous price estimation.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning & Modeling:** Scikit-Learn, XGBoost, NumPy, SciPy
* **Data Processing & Analysis:** Pandas
* **Web Framework:** Streamlit
* **Model Serialization:** Pickle

---

## ⚙️ How to Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/house-price-prediction.git
cd house-price-prediction

