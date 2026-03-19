# 🏥 Health Risk Classifier ML

> An end-to-end Machine Learning pipeline to classify patient health risk (**Healthy vs. High Risk**) using clinical and lifestyle features — with full EDA, model comparison, and XGBoost interpretability.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?style=flat&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green?style=flat)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat)

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Business Value](#-business-value)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Model Results](#-model-results)
- [Feature Importance](#-feature-importance)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [Author](#-author)

---

## 🔍 Project Overview

This project implements a full Machine Learning lifecycle to predict whether a patient is **Healthy (0)** or at **High Health Risk (1)** based on a rich combination of demographic, lifestyle, and medical metrics.

Three classification models are trained and rigorously compared:
- **Logistic Regression** — interpretable baseline
- **Random Forest** — robust ensemble method
- **XGBoost** — state-of-the-art gradient boosting

---

## 💼 Business Value

In healthcare, a **False Negative** (predicting a sick patient as healthy) can be life-threatening. This project prioritizes **Recall** and **ROC-AUC** alongside accuracy to ensure the model is clinically responsible, not just statistically correct.

The XGBoost feature importance analysis also provides actionable insights for medical professionals — surfacing which biomarkers most strongly predict risk.

---

## 📊 Dataset

**File:** `novagen_dataset.csv`

The dataset contains patient records with the following feature categories:

| Category | Features |
|---|---|
| Biometric | BMI, Blood Pressure, Cholesterol, Glucose Level, Heart Rate |
| Lifestyle | Sleep Hours, Exercise Hours, Water Intake, Stress Level |
| Demographic | Age, Blood Group |
| Behavioral | Smoking, Alcohol, Diet Type, Physical Activity |
| Medical | Medical History, Allergies, Mental Health |
| **Target** | `0` = Healthy, `1` = High Health Risk |

---

## 🔄 Project Workflow

```
Raw Data → EDA → Preprocessing → Model Training → Evaluation → Interpretability
```

1. **Data Loading & Inspection** — shape, dtypes, nulls, duplicates
2. **Exploratory Data Analysis** — distributions, correlations, bivariate analysis
3. **Preprocessing** — boolean encoding, stratified train-test split (80/20), StandardScaler
4. **Model Training** — Logistic Regression, Random Forest, XGBoost
5. **Evaluation** — Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
6. **Interpretability** — XGBoost Feature Importance

---

## 🏆 Model Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~85% | 0.888 |
| Random Forest | ~95% | 0.985 |
| **XGBoost** | **~96%** | **0.989** |

> ✅ **XGBoost** achieves the best performance across all metrics, closely followed by Random Forest. Logistic Regression serves as a solid interpretable baseline.

### ROC Curve Comparison

All three models comfortably outperform random classification. XGBoost and Random Forest nearly hug the top-left corner — indicating excellent diagnostic discrimination.

![ROC Curve](ROC_curve.jpg)

---

## 🔬 Feature Importance

XGBoost reveals that **BMI** and **Cholesterol** are the two most powerful predictors of health risk, followed by Stress Level, Glucose Level, and Sleep Hours — offering clinically meaningful, actionable insights.

![Feature Importance](Importance_image.jpg)

---

## 🛠 Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `xgboost` | Gradient boosting classifier |
| `jupyter` | Interactive notebook environment |

---

## ▶️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/kabirpatil12676/Health-Risk-Classifier-ML.git
   cd Health-Risk-Classifier-ML
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
   ```

3. **Launch the notebook**
   ```bash
   jupyter notebook "Health Risk Classification Using Machine Learning.ipynb"
   ```

---

## 👤 Author

**Kabir Patil**

[![GitHub](https://img.shields.io/badge/GitHub-kabirpatil12676-black?style=flat&logo=github)](https://github.com/kabirpatil12676)

---

*If you found this project useful, consider giving it a ⭐ — it helps others discover it!*
