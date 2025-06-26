import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r'C:\Users\Deepika\OneDrive\Documents\ml project\diabetes.csv')

# App title and greeting
st.title('🩺 Diabetes Checkup App')
st.header("👋 Hello! Let's evaluate your health status today.")

# Ask name
username = st.text_input("Enter your name to get started:")
if username:
    st.success(f"Welcome, {username}! 😊")

# Dataset insights
with st.expander("📊 View Dataset Summary"):
    st.subheader('Dataset Overview')
    st.write(df.describe())
    st.bar_chart(df)

# Prepare data
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Sidebar inputs
st.sidebar.header('📋 Enter Health Details')

def user_report():
    Pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 1)
    Glucose = st.sidebar.slider('Glucose Level', 0, 200, 110)
    BloodPressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    SkinThickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    Insulin = st.sidebar.slider('Insulin Level', 0, 846, 79)
    BMI = st.sidebar.slider('BMI', 0.0, 67.0, 25.0)
    DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.5)
    Age = st.sidebar.slider('Age', 10, 100, 30)

    user_data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }

    report_data = pd.DataFrame(user_data, index=[0])
    return report_data

user_data = user_report()

# 🔍 Symptom checklist
st.subheader("🩹 Select Symptoms You Are Experiencing")
st.markdown("_(Choose any that apply)_")
symptoms = {
    "Fatigue 😴": st.checkbox("Fatigue"),
    "Frequent Urination 🚽": st.checkbox("Frequent Urination"),
    "Increased Thirst 🥤": st.checkbox("Increased Thirst"),
    "Blurred Vision 👁️": st.checkbox("Blurred Vision"),
    "Slow Healing Wounds 🩹": st.checkbox("Slow Healing Wounds"),
    "Unexplained Weight Loss ⚖️": st.checkbox("Unexplained Weight Loss"),
    "Tingling or Numbness in Hands/Feet 🖐️🦶": st.checkbox("Tingling/Numbness in Limbs"),
    "Dry Mouth 👄": st.checkbox("Dry Mouth"),
    "Frequent Infections 🦠": st.checkbox("Frequent Infections"),
    "Hunger Even After Eating 🍽️": st.checkbox("Hunger After Meals")
}

# Show selected symptoms
selected_symptoms = [symptom for symptom, checked in symptoms.items() if checked]
if selected_symptoms:
    st.write("### 🧾 You selected the following symptoms:")
    for s in selected_symptoms:
        st.markdown(f"- {s}")
else:
    st.info("No symptoms selected.")

# ✨ Symptom-based insight
st.subheader("🩺 Symptom-based Risk Analysis")
if len(selected_symptoms) >= 3:
    st.warning("⚠️ Based on your symptoms, there's a possible risk of diabetes. Please consult a doctor.")
elif 1 <= len(selected_symptoms) <= 2:
    st.info("🙂 A few symptoms noted. Consider a check-up if they persist.")
else:
    st.success("🎉 No major symptoms reported!")

# Display user input
st.subheader("📋 Health Report")
st.write(user_data)

# Train model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Accuracy
st.subheader('✅ Model Accuracy')
accuracy = accuracy_score(y_test, rf.predict(x_test))
st.write(f"{round(accuracy * 100, 2)}%")

# Prediction
user_result = rf.predict(user_data)

# Diagnosis Output
st.subheader('📢 Your Diagnosis Result')
if user_result[0] == 0:
    st.success(f"🎉 {username}, you are likely **Healthy** based on the prediction!")
else:
    st.error(f"⚠️ {username}, you may be at risk. Please consult a healthcare professional.")

# Tips section
if user_result[0] == 1 or len(selected_symptoms) >= 3:
    with st.expander("💡 Health Tips"):
        st.markdown("""
        - Drink more water and stay hydrated.
        - Eat a balanced diet rich in fiber.
        - Reduce sugar and processed foods.
        - Exercise at least 30 minutes a day.
        - Get regular blood sugar check-ups.
        - Sleep at least 7-8 hours daily.
        """)

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit | Stay healthy!")
