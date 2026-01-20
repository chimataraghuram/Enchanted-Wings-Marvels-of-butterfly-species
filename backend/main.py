
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import io
import os

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Butterfly_classification.keras")
INDICES_PATH = os.path.join(BASE_DIR, "class_indices.pkl")

model = None
class_indices = None
index_to_class = None

def load_resources():
    global model, class_indices, index_to_class
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully")
        
        with open(INDICES_PATH, "rb") as f:
            class_indices = pickle.load(f)
        index_to_class = {v: k for k, v in class_indices.items()}
        print("Class indices loaded successfully")
    except Exception as e:
        print(f"Error loading resources: {e}")

# Load resources on startup
load_resources()

@app.get("/species")
def get_species():
    if not class_indices:
        raise HTTPException(status_code=500, detail="Class indices not loaded")
    return {"species": list(class_indices.keys())}

@app.get("/")
def read_root():
    return {"message": "Butterfly Classifier API is running"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not model or not index_to_class:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_label = index_to_class[predicted_index]
        confidence = float(np.max(prediction))
        
        return {
            "label": predicted_label,
            "confidence": confidence,
            "filename": file.filename
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
