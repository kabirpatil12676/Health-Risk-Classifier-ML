# Health Risk Classification & Web App

**Author:** [Kabir Patil](https://www.linkedin.com/in/kabir-patil-7a2a9b30b/)

## Project Overview
Catching health risks early is important for improving patient care. In this project, I built a machine learning pipeline to predict whether individuals are 'Healthy' (0) or 'At-Risk' (1) based on their medical history, lifestyle factors, and physiological data. 

**🌟 Major Update:** I have also deployed the best performing model (XGBoost) into a fully interactive web application using Streamlit! This allows users or doctors to input patient vitals and receive a real-time risk assessment.

## Repository Contents
*   `Health_Risk_Classification_Enhanced.ipynb`: The main Jupyter Notebook with my exploratory data analysis (EDA), data cleaning, model training, and evaluation.
*   `app.py`: The Streamlit Web Application script for the interactive UI.
*   `train_model.py`: The script used to train and serialize the final deployment model.
*   `novagen_dataset.csv`: The dataset containing over 9,500 patient records across 22 different features.
*   `requirements.txt`: The Python packages you need to run this code locally.
*   `xgboost_health_model.pkl`: The serialized XGBoost classifier model.

## Key Findings from the Data
The dataset includes numerical and categorical data like Age, BMI, Blood Pressure, Cholesterol, Diet, and Sleep Hours.

Some interesting initial findings from the EDA:
1.  **Age & BMI:** These showed up as the strongest indicators. As age and BMI go up, the chances of being classified as 'At-Risk' increase significantly.
2.  **Blood Pressure:** This also had a noticeable correlation with the risk classification.

Check the notebook for the actual distribution plots and the feature correlation heatmap.

## Machine Learning Models
I tested a few different models to see what worked best:

1.  **Logistic Regression:** Used this as a simple baseline model.
2.  **Random Forest Classifier:** A stronger tree-based model, tuned using GridSearch to find good hyperparameters.
3.  **XGBoost Classifier:** This is often the go-to for tabular data, so I wanted to see how it performed.

### Results
After tuning, the XGBoost and Random Forest models outperformed the Logistic Regression baseline by a wide margin. They are much better at picking up on the complex relationships in patient data. The notebook includes a chart comparing their F1-Scores. The XGBoost model was selected for the final web deployment!

## Running the Code Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kabirpatil12676/Health-Risk-Classifier-ML.git
   cd Health-Risk-Classifier-ML
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **To view the Data Analysis (Jupyter Notebook):**
   ```bash
   jupyter notebook "Health_Risk_Classification_Enhanced.ipynb"
   ```

4. **To launch the Web Application (Streamlit):**
   ```bash
   streamlit run app.py
   ```

## Contact
**Kabir Patil**
*   **LinkedIn:** [linkedin.com/in/kabir-patil-7a2a9b30b](https://www.linkedin.com/in/kabir-patil-7a2a9b30b/)
*   **GitHub:** [github.com/kabirpatil12676](https://github.com/kabirpatil12676)
