# Health Risk Classification System 🏥

## Project Overview
This project presents an end-to-end Machine Learning pipeline designed to predict health risks based on a comprehensive dataset of numerical and categorical medical markers. Recognizing at-risk patients early is crucial in the healthcare sector, and this notebook demonstrates a robust, predictive approach to solving that challenge.

## Business Value
In healthcare, minimizing **False Negatives** (predicting a sick patient is healthy) is paramount. This project emphasizes critical evaluation metrics such as **Recall** and **F1-Score**, and utilizes interpretable machine learning to extract actionable insights for medical professionals.

## Key Features
- **Exploratory Data Analysis (EDA):** Correlation heatmaps and bivariate analysis mapping dependencies between patient demographics, lifestyle features, and health risk.
- **Data Preprocessing:** Handled boolean conversion, stratified train-test splitting, and feature scaling using standard gradient distance optimizations.
- **Machine Learning Models:** Trained and compared three baseline models:
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier (Gradient Boosting)
- **Advanced Evaluation:** Used Confusion Matrices, Classification Reports, and ROC-AUC Curve Comparisons to properly evaluate diagnostic performance.
- **Interpretability:** Extracted and visualized Feature Importances from the XGBoost model to inform doctors on what biological metrics (e.g., Blood Pressure, BMI, Stress Levels) drive the predictive risk profile.

## Technologies Used
- Python, Pandas, NumPy
- Scikit-Learn, XGBoost
- Matplotlib, Seaborn

## How to Run
1. Clone this repository.
2. Install the necessary dependencies (`pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter`).
3. Open `Health Risk Classification Using Machine Learning.ipynb` in your preferred Jupyter environment.

## Author
Kabir Patil
