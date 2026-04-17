import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(page_title="ML Predictor", page_icon="🤖", layout="centered")

# Title
st.title("🤖 Machine Learning Prediction App")
st.markdown("### Enter input values to get prediction")

# Load model safely
@st.cache_resource
def load_model():
    with open("Model_cleaned.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Try to get feature names (if available)
try:
    feature_names = model.feature_names_in_
except:
    feature_names = [f"Feature {i+1}" for i in range(5)]  # fallback

st.markdown("### 🔢 Input Features")

# Create inputs dynamically
inputs = []
for feature in feature_names:
    val = st.number_input(f"{feature}", value=0.0)
    inputs.append(val)

# Prediction button
if st.button("🚀 Predict"):
    try:
        input_array = np.array(inputs).reshape(1, -1)
        prediction = model.predict(input_array)

        st.success(f"✅ Prediction: {prediction[0]}")

    except Exception as e:
        st.error("❌ Error during prediction")
        st.text(str(e))

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")
