import streamlit as st
import pandas as pd
import joblib

# Set Page Config
st.set_page_config(page_title="Health Risk Predictor", page_icon="🩺", layout="wide")

# Load Model & Columns (Cached for performance)
@st.cache_resource
def load_model():
    model = joblib.load("xgboost_health_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, model_columns

model, model_columns = load_model()

# Sidebar for App Information
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3209/3209995.png", width=100)
    st.title("About the Model")
    st.info("This application uses an **XGBoost Classifier** to predict whether a patient is at risk for severe health conditions based on 22 physiological and lifestyle features.")
    st.markdown("---")
    st.markdown("### Risk Factors Analyzed")
    st.error("""**Critical Predictors**
* Age & Body Mass Index (BMI)
* Systolic Blood Pressure""")
    st.warning("""**Lifestyle Indicators**
* Diet Quality & Hydration
* Sleep & Physical Activity""")
    st.success("""**Medical Context**
* Family Medical History
* Pre-existing Conditions""")
    st.markdown("---")
    st.markdown("**Author:** Kabir Patil")
    st.markdown("[GitHub](https://github.com/kabirpatil12676) | [LinkedIn](https://www.linkedin.com/in/kabir-patil-7a2a9b30b/)")

# Main Content
st.title("🩺 Health Risk Predictive Analytics")
st.write("Provide the patient's medical and lifestyle data below. The system will process the inputs through the XGBoost model to generate a real-time health risk classification.")
st.markdown("---")

# User Input Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📊 Vitals & Demographics")
    with st.container(border=True):
        age = st.slider("Age (Years)", 0, 100, 45)
        bmi = st.number_input("Body Mass Index (BMI)", 10.0, 50.0, 25.0, step=0.5)
        blood_pressure = st.slider("Blood Pressure (Systolic)", 80, 200, 120)
        cholesterol = st.slider("Cholesterol Level (mg/dL)", 100, 300, 200)
        glucose = st.slider("Glucose Level (mg/dL)", 50, 250, 100)
        heart_rate = st.slider("Resting Heart Rate (bpm)", 40, 120, 70)

with col2:
    st.subheader("🏃‍♂️ Lifestyle & Medical History")
    with st.container(border=True):
        col2a, col2b = st.columns(2)
        with col2a:
            sleep = st.number_input("Sleep (Hours/Day)", 0.0, 24.0, 7.0, step=0.5)
            exercise = st.number_input("Exercise (Hours/Week)", 0.0, 20.0, 3.0, step=0.5)
            water = st.number_input("Water Intake (Liters/Day)", 0.0, 10.0, 2.5, step=0.1)
            stress = st.slider("Stress Level (1-10)", 1, 10, 5)
        with col2b:
            smoking = st.selectbox("Smoking Status", options=[0, 1, 2], format_func=lambda x: ["0: Non-Smoker", "1: Current Smoker", "2: Former Smoker"][x])
            alcohol = st.selectbox("Alcohol Consumption", options=[0, 1, 2], format_func=lambda x: ["0: None", "1: Occasional", "2: Frequent"][x])
            diet = st.selectbox("Diet Quality", options=[0, 1, 2], format_func=lambda x: ["0: Poor", "1: Average", "2: Good"][x])
            physical_activity = st.selectbox("Physical Activity", options=[0, 1, 2], format_func=lambda x: ["0: Low", "1: Moderate", "2: High"][x])

st.markdown("### 🏥 Additional Medical Context")
with st.expander("Click to expand Diet, Blood Group, and History"):
    ecol1, ecol2, ecol3 = st.columns(3)
    with ecol1:
        mental_health = st.selectbox("Mental Health Status", options=[0, 1, 2], format_func=lambda x: ["0: Poor", "1: Average", "2: Good"][x])
        history = st.selectbox("Family Medical History", options=[0, 1, 2], format_func=lambda x: ["0: No", "1: Yes", "2: Unknown"][x])
    with ecol2:
        allergies = st.selectbox("Known Allergies", options=[0, 1, 2], format_func=lambda x: ["0: No", "1: Yes", "2: Multiple"][x])
        diet_vegan = st.checkbox("Vegan Diet Follower")
        diet_veg = st.checkbox("Vegetarian Diet Follower")
    with ecol3:
        st.markdown("**Blood Group Type**")
        bg_ab = st.checkbox("Type AB")
        bg_b = st.checkbox("Type B")
        bg_o = st.checkbox("Type O")

st.markdown("---")

# Prediction Button
predict_btn = st.button("🔍 Generate Health Risk Prediction", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Processing data through XGBoost model..."):
        # Compile input mapped perfectly to training data
        input_data = {
            'Age': age, 'BMI': bmi, 'Blood_Pressure': blood_pressure, 'Cholesterol': cholesterol,
            'Glucose_Level': glucose, 'Heart_Rate': heart_rate, 'Sleep_Hours': sleep,
            'Exercise_Hours': exercise, 'Water_Intake': water, 'Stress_Level': stress,
            'Smoking': smoking, 'Alcohol': alcohol, 'Diet': diet, 'MentalHealth': mental_health,
            'PhysicalActivity': physical_activity, 'MedicalHistory': history, 'Allergies': allergies,
            'Diet_Type__Vegan': int(diet_vegan), 'Diet_Type__Vegetarian': int(diet_veg),
            'Blood_Group_AB': int(bg_ab), 'Blood_Group_B': int(bg_b), 'Blood_Group_O': int(bg_o)
        }
        
        input_df = pd.DataFrame([input_data])[model_columns]
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
    # Result Display
    if prediction == 1:
        st.error("## ⚠️ High Health Risk Detected")
        st.markdown(f"**Model Confidence:** {prediction_proba[1]*100:.1f}%")
        st.markdown("The predictive model has classified this profile as **At-Risk**. Based on historical data, these physiological and lifestyle metrics correlate with significant health complications. Immediate medical consultation and preventative lifestyle adjustments are highly recommended.")
    else:
        st.success("## ✅ Healthy Profile Detected")
        st.markdown(f"**Model Confidence:** {prediction_proba[0]*100:.1f}%")
        st.markdown("The predictive model has classified this profile as **Healthy**. Keep up the good work maintaining positive lifestyle and dietary habits!")
