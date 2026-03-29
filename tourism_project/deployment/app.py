import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="AnugrahaKenny/TouristPrediction", # Updated repo ID
    filename="best_tourism_prediction_model_v1.joblib" # Updated filename
)
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Purchase Prediction
st.title("Tourism Package Purchase Prediction App")
st.write("""
This application predicts whether a customer will purchase the **Wellness Tourism Package** based on their details.
Please enter the required information below to get a prediction.
""")

# User input for numerical features
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
city_tier = st.selectbox("City Tier", [1, 2, 3]) # Assuming CityTier is 1, 2, or 3
number_of_person_visiting = st.number_input("Number of People Visiting", min_value=1, max_value=10, value=1, step=1)
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
number_of_trips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=5, step=1)
passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
own_car = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
number_of_children_visiting = st.number_input("Number of Children Visiting (below 5)", min_value=0, max_value=10, value=0, step=1)
monthly_income = st.number_input("Monthly Income", min_value=0.0, value=50000.0, step=100.0)
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=0, max_value=5, value=3, step=1)
number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2, step=1)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=15.0, step=0.1)

# User input for categorical features (using LabelEncoder values from prep.py as guidance)
type_of_contact = st.selectbox("Type of Contact", ['Company Invited', 'Self Inquiry'])
occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'FreeLancer'])
gender = st.selectbox("Gender", ['Male', 'Female'])
marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP', 'President', 'Director'])
product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King']) # These are sample values, adjust if needed

# Encoding categorical inputs based on prep.py's LabelEncoder logic
# Note: In a real scenario, you would use the *fitted* LabelEncoders or a mapping.
# For simplicity, we'll hardcode typical mappings if they were seen in training data.
# The model's OneHotEncoder will handle these after the pipeline's preprocessor.
contact_mapping = {'Company Invited': 0, 'Self Inquiry': 1} # Sample mapping, ensure it matches training
occupation_mapping = {'Salaried': 3, 'Small Business': 4, 'Large Business': 1, 'FreeLancer': 0} # Sample mapping
gender_mapping = {'Male': 1, 'Female': 0}
marital_status_mapping = {'Single': 2, 'Married': 1, 'Divorced': 0}
designation_mapping = {'Manager': 3, 'Executive': 1, 'Senior Manager': 5, 'AVP': 0, 'VP': 6, 'President': 4, 'Director': 2} # Sample
product_pitched_mapping = {'Basic': 0, 'Deluxe': 1, 'Standard': 2, 'Super Deluxe': 3, 'King': 4} # Sample

# Assemble input into DataFrame
input_data = pd.DataFrame([
    {
        'Age': age,
        'CityTier': city_tier,
        'NumberOfPersonVisiting': number_of_person_visiting,
        'PreferredPropertyStar': preferred_property_star,
        'NumberOfTrips': number_of_trips,
        'Passport': passport,
        'OwnCar': own_car,
        'NumberOfChildrenVisiting': number_of_children_visiting,
        'MonthlyIncome': monthly_income,
        'PitchSatisfactionScore': pitch_satisfaction_score,
        'NumberOfFollowups': number_of_followups,
        'DurationOfPitch': duration_of_pitch,
        'TypeofContact': contact_mapping.get(type_of_contact, -1), # Use .get with default for safety
        'Occupation': occupation_mapping.get(occupation, -1),
        'Gender': gender_mapping.get(gender, -1),
        'MaritalStatus': marital_status_mapping.get(marital_status, -1),
        'Designation': designation_mapping.get(designation, -1),
        'ProductPitched': product_pitched_mapping.get(product_pitched, -1)
    }
])

# Prediction
if st.button("Predict Purchase Likelihood"):
    prediction_proba = model.predict_proba(input_data)[:, 1][0] # Get probability of positive class
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"The customer is likely to purchase the package (Probability: {prediction_proba:.2f}).")
    else:
        st.info(f"The customer is unlikely to purchase the package (Probability: {prediction_proba:.2f}).")

