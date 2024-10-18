import streamlit as st
from pickle import load
import os

# Load the model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/decision_tree_classifier_default_42.sav")
model = load(open(model_path, "rb"))

# Class dictionary for prediction labels
class_dict = {
    "0": "Iris setosa",
    "1": "Iris versicolor",
    "2": "Iris virginica"
}

# Streamlit App Title
st.title("Iris Flower Prediction App")

# Input fields for the flower measurements
val1 = st.number_input("Petal Width (in cm)", min_value=0.0, format="%.2f")
val2 = st.number_input("Petal Length (in cm)", min_value=0.0, format="%.2f")
val3 = st.number_input("Sepal Width (in cm)", min_value=0.0, format="%.2f")
val4 = st.number_input("Sepal Length (in cm)", min_value=0.0, format="%.2f")

# When the user clicks the 'Predict' button
if st.button("Predict"):
    # Prepare the input data as a 2D list (list of lists)
    data = [[val1, val2, val3, val4]]
    
    # Make the prediction
    prediction = str(model.predict(data)[0])
    
    # Show the predicted class
    pred_class = class_dict[prediction]
    st.write(f"Prediction: {pred_class}")
