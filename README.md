<div align="center">

# 🩺 Health Risk Predictive Analytics

### An End-to-End Machine Learning Pipeline with an Interactive Web Application

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Deployed-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://health-classifier-kp.streamlit.app/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

**[🚀 Try the Live App »](https://health-classifier-kp.streamlit.app/)** &nbsp;|&nbsp; **[📓 View the Notebook »](Health_Risk_Classification_Enhanced.ipynb)** &nbsp;|&nbsp; **[👤 Connect on LinkedIn »](https://www.linkedin.com/in/kabir-patil-7a2a9b30b/)**

</div>

---

## 📌 Project Overview

Early detection of health risk is one of the most impactful applications of machine learning in healthcare. This project builds a **complete, production-ready supervised learning pipeline** that predicts whether a patient is `Healthy (0)` or `At-Risk (1)` based on their physiological vitals, lifestyle habits, and medical history.

The pipeline goes beyond a simple notebook — the best performing model (XGBoost, AUC = 0.989) is **deployed as a live interactive web application** accessible to anyone.

**Key Technical Skills Demonstrated:** Feature Engineering · EDA · Hyperparameter Tuning (GridSearchCV) · Model Evaluation · Model Serialization (Joblib) · Streamlit Deployment

---

## 🌐 Live Demo

> **Try the deployed app here:** [**https://health-classifier-kp.streamlit.app/**](https://health-classifier-kp.streamlit.app/)

Input any patient's vitals, lifestyle data, and medical history and the XGBoost model will return a **real-time health risk classification** with a confidence score.

---

## 📊 Model Results & Visualisations

### ROC Curve Comparison — All Models

The ROC Curve directly shows how well each classifier distinguishes between healthy and at-risk patients across all thresholds. XGBoost achieved the highest AUC of **0.989**.

![ROC Curve Comparison](ROC%20curve.jpg)

| Model | AUC Score | Notes |
|---|---|---|
| Logistic Regression | 0.888 | Baseline model |
| Random Forest | 0.985 | Tuned with GridSearchCV |
| **XGBoost** | **0.989** | **Selected for deployment** |

---

### Feature Importance — Top 15 Predictors (XGBoost)

Understanding *which* features drive predictions is critical for clinical interpretability. The chart below shows the relative feature importance scores extracted directly from the trained XGBoost model.

![Feature Importance](Feature%20Importance.jpg)

> **Key Insight:** BMI and Cholesterol are the single strongest predictors of health risk in this dataset, followed by Stress Level and Glucose Level. This aligns strongly with established medical literature on lifestyle-driven chronic disease risk.

---

## 🏗️ Project Architecture

```
Health-Risk-Classifier-ML/
│
├── 📓 Health_Risk_Classification_Enhanced.ipynb  ← Full EDA + Model Training
├── 🌐 app.py                                      ← Streamlit Web Application
├── 🏋️ train_model.py                              ← Model Training & Serialization Script
│
├── 📦 xgboost_health_model.pkl                   ← Serialized XGBoost Model
├── 📦 model_columns.pkl                           ← Feature Column Order (for safe inference)
├── 📄 novagen_dataset.csv                         ← Dataset (9,500+ patient records, 22 features)
│
└── 📋 requirements.txt                            ← Python dependencies
```

---

## 🤖 Machine Learning Pipeline

### 1. Exploratory Data Analysis (EDA)
- Investigated distributions of all 22 features using histograms and KDE plots.
- Analyzed correlation between features and the target variable using a heatmap.
- Detected and treated class imbalance and missing values.

### 2. Feature Engineering
- One-hot encoded categorical variables (`Diet_Type`, `Blood_Group`).
- Standardized numerical features for Logistic Regression compatibility.
- Preserved original scale for tree-based models (RF, XGBoost).

### 3. Model Training & Hyperparameter Tuning
Three classifiers were trained and evaluated:

| Model | Strategy |
|---|---|
| Logistic Regression | Baseline, `C` tuned via GridSearchCV |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split` tuned |
| XGBoost | `learning_rate`, `max_depth`, `n_estimators` tuned |

### 4. Evaluation
Models were compared using **AUC-ROC**, **F1-Score**, **Precision**, and **Recall** on a held-out 20% test set. XGBoost was the clear winner and selected for web deployment.

---

## 🚀 Running Locally

**1. Clone the repository:**
```bash
git clone https://github.com/kabirpatil12676/Health-Risk-Classifier-ML.git
cd Health-Risk-Classifier-ML
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. View the Notebook (EDA + Training):**
```bash
jupyter notebook "Health_Risk_Classification_Enhanced.ipynb"
```

**4. Launch the Streamlit Web App:**
```bash
streamlit run app.py
```

---

## 📁 Dataset

The `novagen_dataset.csv` contains **9,500+ patient records** across **22 features** including:

| Category | Features |
|---|---|
| Physiological | Age, BMI, Blood Pressure, Cholesterol, Glucose Level, Heart Rate |
| Lifestyle | Sleep Hours, Exercise Hours, Water Intake, Stress Level, Smoking, Alcohol |
| Dietary | Diet Quality, Diet Type (Vegan/Vegetarian) |
| Medical | Family Medical History, Mental Health, Allergies, Blood Group |

**Target Variable:** `0 = Healthy`, `1 = At-Risk`

---

## 👨‍💻 Author

**Kabir Patil**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/kabir-patil-7a2a9b30b/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/kabirpatil12676)
