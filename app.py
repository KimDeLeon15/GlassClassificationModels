import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load trained model and scaler with error handling
try:
    with open("glass.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Model file 'glass.pkl' not found. Please make sure it's in the same directory.")
    st.stop()

try:
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Scaler file 'scaler.pkl' not found. Please make sure it's in the same directory.")
    st.stop()


# App Title and Description
st.title("🔍 Glass Type Prediction App")
st.write("Predict the type of glass based on chemical composition.")

st.markdown("""
This app uses a trained Random Forest Classifier on the Glass Identification dataset to classify glass types 
based on their chemical properties (RI, Na, Mg, Al, Si, K, Ca, Ba, Fe).
""")


# Glass Type Labels
glass_labels = {
    1: "Building Windows (Float)",
    2: "Building Windows (Non-Float)",
    3: "Vehicle Windows (Float)",
    4: "Vehicle Windows (Non-Float)",
    5: "Containers",
    6: "Tableware",
    7: "Headlamps"
}


# Session Defaults
defaults = {
    "ri": 1.51761, "na": 13.89, "mg": 3.60, "al": 1.36,
    "si": 72.73, "k": 0.48, "ca": 7.83, "ba": 0.0, "fe": 0.0,
    "clear_inputs": False
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# Clear inputs
if st.session_state.clear_inputs:
    for key in defaults:
        if key != "clear_inputs":
            st.session_state[key] = 0.0
    st.session_state.clear_inputs = False


# Example Input Button
if st.button("Use Example Data"):
    example = {
        "ri": 1.51761, "na": 13.89, "mg": 3.60, "al": 1.36,
        "si": 72.73, "k": 0.48, "ca": 7.83, "ba": 0.0, "fe": 0.0
    }
    for key, val in example.items():
        st.session_state[key] = val


# Input Form
with st.form("glass_form"):
    st.subheader("Enter Chemical Composition")
    RI = st.number_input("RI (Refractive Index)", format="%.6f", key="ri")
    Na = st.number_input("Na (Sodium)", format="%.2f", key="na")
    Mg = st.number_input("Mg (Magnesium)", format="%.2f", key="mg")
    Al = st.number_input("Al (Aluminum)", format="%.2f", key="al")
    Si = st.number_input("Si (Silicon)", format="%.2f", key="si")
    K  = st.number_input("K (Potassium)", format="%.2f", key="k")
    Ca = st.number_input("Ca (Calcium)", format="%.2f", key="ca")
    Ba = st.number_input("Ba (Barium)", format="%.2f", key="ba")
    Fe = st.number_input("Fe (Iron)", format="%.2f", key="fe")

    col1, col2 = st.columns(2)
    with col1:
        predict = st.form_submit_button("Predict Glass Type")
    with col2:
        clear = st.form_submit_button("Clear Inputs")

# Handle Clear Input Action
if clear:
    st.session_state.clear_inputs = True
    st.rerun()


# Prediction Output
if predict:
    # Optional warnings for unrealistic inputs
    if RI < 1.5 or RI > 1.55:
        st.warning("⚠️ RI value seems outside the typical range (1.5 - 1.55)")

    input_df = pd.DataFrame([[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]],
                            columns=["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"])
    
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    label = glass_labels.get(prediction, "Unknown")
    st.success(f"🧪 Predicted Glass Type: Type {prediction} - {label}")

    st.subheader("Prediction Probabilities")
    for class_id, prob in zip(model.classes_, probabilities):
        label_i = glass_labels.get(class_id, "Unknown")
        st.write(f"Type {class_id} - {label_i}: {prob:.2%}")


# Feature Importance Plot
with st.expander("🔍 Feature Importance"):
    st.write("Shows which features the model relies on most.")
    importance = model.feature_importances_
    features = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
    fig, ax = plt.subplots()
    ax.barh(features, importance, color="teal")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Random Forest)")
    st.pyplot(fig)

# Model Summary
with st.expander(" Model Summary"):
    st.markdown("""
    - **Algorithm**: Random Forest Classifier  
    - **Accuracy**: ~89%  
    - **Dataset**: UCI Glass Identification  
    - **Features**: RI, Na, Mg, Al, Si, K, Ca, Ba, Fe  
    - **Target Classes**: 7 types of glass
    """)

# Sidebar credits and GitHub link
with st.sidebar:
    st.header("📁 Resources")
    st.markdown("[📌 GitHub Repo](https://github.com/KimDeLeon15/GlassClassificationModels)")
    st.markdown("Created by **Kim De Leon** for Business Analytics")
    st.markdown("Powered by **Streamlit** + **scikit-learn** + **Random Forest**")