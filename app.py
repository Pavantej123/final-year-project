import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("random_forest_asd_model.pkl")

# Streamlit App Title
st.title("Autism Spectrum Disorder (ASD) Detection")
st.write("""
This application predicts the likelihood of Autism Spectrum Disorder (ASD) 
based on the provided inputs. Please enter the necessary details below.
""")

# Function to handle user input
def user_input_features():
    st.subheader("Enter the Following Features:")
    
    # Numerical features
    case_no = st.number_input("Case Number:", min_value=0, step=1)
    age_mons = st.number_input("Age in Months:", min_value=0, step=1)
    qchat_score = st.number_input("Qchat-10-Score:", min_value=0, max_value=10, step=1)

    # Questions A1 to A10
    a1 = st.selectbox("Does the child make eye contact?", ["Rarely", "Sometimes", "Often"])
    a2 = st.selectbox("Does the child respond to their name?", ["Rarely", "Sometimes", "Often"])
    a3 = st.selectbox("Does the child enjoy playing with peers?", ["Rarely", "Sometimes", "Often"])
    a4 = st.selectbox("Does the child imitate actions (like clapping or waving)?", ["Rarely", "Sometimes", "Often"])
    a5 = st.selectbox("Does the child use gestures to communicate (e.g., pointing)?", ["Rarely", "Sometimes", "Often"])
    a6 = st.selectbox("Does the child engage in repetitive behaviors (e.g., rocking)?", ["Rarely", "Sometimes", "Often"])
    a7 = st.selectbox("Does the child show interest in objects more than people?", ["Rarely", "Sometimes", "Often"])
    a8 = st.selectbox("Does the child show sensitivity to sounds, lights, or textures?", ["Rarely", "Sometimes", "Often"])
    a9 = st.selectbox("Does the child understand and follow simple instructions?", ["Rarely", "Sometimes", "Often"])
    a10 = st.selectbox("Does the child show unusual attachment to objects?", ["Rarely", "Sometimes", "Often"])
    
    # Encode responses into numerical values
    response_mapping = {"Rarely": 0, "Sometimes": 1, "Often": 2}
    a1 = response_mapping[a1]
    a2 = response_mapping[a2]
    a3 = response_mapping[a3]
    a4 = response_mapping[a4]
    a5 = response_mapping[a5]
    a6 = response_mapping[a6]
    a7 = response_mapping[a7]
    a8 = response_mapping[a8]
    a9 = response_mapping[a9]
    a10 = response_mapping[a10]

    # Categorical features
    sex = st.selectbox("Sex (m/f):", ["m", "f"])
    jaundice = st.selectbox("Jaundice History (yes/no):", ["yes", "no"])
    family_mem_with_asd = st.selectbox("Family Member with ASD (yes/no):", ["yes", "no"])
    
    # Map categorical variables
    sex = 1 if sex == "m" else 0
    jaundice = 1 if jaundice == "yes" else 0
    family_mem_with_asd = 1 if family_mem_with_asd == "yes" else 0
    
    # Construct the input DataFrame
    features = {
        "Case_No": case_no,
        "A1": a1,
        "A2": a2,
        "A3": a3,
        "A4": a4,
        "A5": a5,
        "A6": a6,
        "A7": a7,
        "A8": a8,
        "A9": a9,
        "A10": a10,
        "Age_Mons": age_mons,
        "Qchat-10-Score": qchat_score,
        "Sex": sex,
        "Jaundice": jaundice,
        "Family_mem_with_ASD": family_mem_with_asd
    }
    
    return pd.DataFrame([features])

# User input
input_df = user_input_features()

# Display input
st.subheader("Input Data")
st.write(input_df)

# Prediction button
if st.button("Predict ASD"):
    try:
        prediction = model.predict(input_df)
        result = "Likely ASD" if prediction[0] == 1 else "Unlikely ASD"
        st.write(f"### Prediction: {result}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
