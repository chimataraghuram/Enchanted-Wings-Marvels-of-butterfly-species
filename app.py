import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle

# Load model and class indices
model = tf.keras.models.load_model("Butterfly_classification.keras")

with open("class_indices.pkl", "rb") as f:
    class_indices = pickle.load(f)

# Reverse the class_indices to get label from prediction index
index_to_class = {v: k for k, v in class_indices.items()}

st.title("🦋 Butterfly Species Classifier")
st.write("Upload a butterfly image and I’ll tell you its species!")

st.markdown("📂 **Choose a butterfly image**  \n📌 *(Filename should include the actual species name for comparison)*")
# File uploader with no label
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Get true label from filename
    filename = uploaded_file.name
    true_label = filename.split('_')[0]  # or use full name logic if needed

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = index_to_class[predicted_index]
    confidence = np.max(prediction)

    # Show result
    st.success(f"✅ Predicted: **{predicted_label}**")
    st.info(f"📸 Actual (from filename): **{true_label}**")
    st.write(f"🔍 Confidence: {confidence * 100:.2f}%")
