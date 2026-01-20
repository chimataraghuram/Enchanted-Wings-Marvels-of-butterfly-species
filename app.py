import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Butterfly Classifier",
    page_icon="🦋",
    layout="centered"
)

# Load model and class indices
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), "Butterfly_classification.keras")
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_class_indices():
    try:
        indices_path = os.path.join(os.path.dirname(__file__), "class_indices.pkl")
        with open(indices_path, "rb") as f:
            class_indices = pickle.load(f)
        return class_indices
    except Exception as e:
        st.error(f"Error loading class indices: {e}")
        return None

model = load_model()
class_indices = load_class_indices()

if model and class_indices:
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

        try:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)

            # Preprocess
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            with st.spinner("Classifying..."):
                prediction = model.predict(img_array)
                predicted_index = np.argmax(prediction)
                predicted_label = index_to_class[predicted_index]
                confidence = np.max(prediction)

            # Show result
            st.success(f"✅ Predicted: **{predicted_label}**")
            st.info(f"📸 Actual (from filename): **{true_label}**")
            st.write(f"🔍 Confidence: {confidence * 100:.2f}%")
            
            # Add a progress bar for confidence
            st.progress(float(confidence))

        except Exception as e:
            st.error(f"Error processing image: {e}")
else:
    st.warning("Please ensure 'Butterfly_classification.keras' and 'class_indices.pkl' are in the same directory.")
