# Plant Disease Detection App with CSV Logging
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Paths
MODEL_PATH = "D:\\ML Projects(TensorFlow)\\plant_disease_cnn\\plant_disease_cnn_model.keras"
LOG_CSV_PATH = "D:\\ML Projects(TensorFlow)\\prediction_log.csv"

# Load the trained model
model = load_model(MODEL_PATH)

# Define class names
class_names = [
    'Apple___Black_rot', 'Apple___healthy', 'Apple___rust', 'Apple___scab',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Full disease advice
suggestions = {
    "Black rot": "Remove infected leaves and avoid overhead watering.",
    "Bacterial spot": "Remove affected leaves and apply copper-based fungicide.",
    "Powdery mildew": "Use sulfur-based fungicides and ensure proper air circulation.",
    "Early blight": "Use fungicide and practice crop rotation.",
    "Late blight": "Remove infected plants and avoid moisture on leaves.",
    "Healthy": "No disease detected. Keep monitoring regularly.",
    "Cercospora leaf spot Gray leaf spot": "Use fungicides and avoid working when plants are wet.",
    "Common rust": "Use resistant varieties and apply fungicide if needed.",
    "Northern Leaf Blight": "Use resistant hybrids and rotate crops yearly.",
    "Esca (Black Measles)": "Prune infected vines and improve vineyard hygiene.",
    "Leaf blight (Isariopsis Leaf Spot)": "Prune and destroy infected leaves; apply fungicide.",
    "Haunglongbing (Citrus greening)": "Remove infected trees and control psyllid vectors.",
    "Septoria leaf spot": "Remove infected leaves and avoid overhead irrigation.",
    "Leaf Mold": "Provide proper ventilation and apply fungicide as needed.",
    "Target Spot": "Avoid excessive nitrogen and apply fungicide.",
    "Tomato Yellow Leaf Curl Virus": "Use resistant varieties and control whiteflies using insecticides.",
    "Tomato mosaic virus": "Disinfect tools and use certified disease-free seeds.",
    "Spider mites Two-spotted spider mite": "Spray miticides and maintain high humidity.",
    "Leaf scorch": "Ensure proper watering and improve drainage.",
    "Rust": "Apply sulfur or copper-based fungicides."
}

# Utility functions
def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def parse_label(label):
    parts = label.split("___")
    plant = parts[0].replace("_", " ")
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    return plant, disease

def log_prediction(timestamp, plant, disease, confidence, filename):
    new_row = pd.DataFrame([[timestamp, plant, disease, confidence, filename]],
                           columns=["Timestamp", "Plant", "Disease", "Confidence (%)", "Filename"])
    if os.path.exists(LOG_CSV_PATH):
        df = pd.read_csv(LOG_CSV_PATH)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
    df.to_csv(LOG_CSV_PATH, index=False)

# Streamlit App
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to detect the plant and possible disease.")

# Sidebar Info
with st.sidebar:
    st.header("ℹ️ App Info")
    st.markdown("""
    **Plant Disease Detector**
    
    - Detects diseases from plant leaf images.
    - Trained on PlantVillage dataset.
    - Provides disease diagnosis and treatment suggestions.

    👨‍🌾 Built with TensorFlow + Streamlit
    """)

uploaded_file = st.file_uploader("📷 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption='📷 Uploaded Image', use_container_width=True)
    with st.spinner('🔍 Analyzing...'):
        img_array = preprocess_image(uploaded_file)
        predictions = model.predict(img_array)
        pred_index = np.argmax(predictions)
        confidence = round(float(np.max(predictions)) * 100, 2)
        predicted_label = class_names[pred_index]

        plant, disease = parse_label(predicted_label)
        suggestion = suggestions.get(disease, "No suggestion available.")

        st.markdown(f"🪴 **Plant**: {plant}")
        st.markdown(f"🩺 **Diagnosis**: {disease}")
        st.markdown(f"📊 **Confidence**: {confidence}%")
        st.markdown(f"💡 **Suggestion**: {suggestion}")

        log_prediction(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), plant, disease, confidence, uploaded_file.name)

if os.path.exists(LOG_CSV_PATH):
    st.markdown("## 🧾 Prediction History Log")
    df_log = pd.read_csv(LOG_CSV_PATH)

    if 'timestamp' in df_log.columns:
        df_log['timestamp'] = pd.to_datetime(df_log['timestamp'], errors='coerce')
        df_log['timestamp'] = df_log['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")

    st.dataframe(df_log[::-1], use_container_width=True)  # Show latest first

    # Optional: Add download button
    csv = df_log.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Prediction Log as CSV",
        data=csv,
        file_name="prediction_log.csv",
        mime="text/csv"
    )
else:
    st.info("No predictions logged yet.")



# .\tensorflow_venv\Scripts\Activate

# streamlit run "plant_disease_cnn/Plant Disease App.py"
