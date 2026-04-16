import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("Model (5).pkl", "rb"))

# Page config
st.set_page_config(page_title="ML Predictor App", layout="centered")

# Title
st.title("🔮 Machine Learning Prediction App")
st.write("Enter the required details below to get predictions")

# Example input fields (EDIT these based on your model)
st.subheader("📥 Input Features")

feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

# Add more fields if needed
# feature4 = st.number_input("Feature 4")

# Prediction button
if st.button("Predict 🚀"):
    try:
        # Prepare input (adjust based on your model)
        input_data = np.array([[feature1, feature2, feature3]])

        prediction = model.predict(input_data)

        st.success(f"✅ Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
