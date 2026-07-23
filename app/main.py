from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VEya Model API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ["Cataract", "Conjunctivitis", "Normal", "Pterygium"]
MODEL_PATH = "models/veya_model_final.keras"
IMG_SIZE = (224, 224)

model = None

def load_model():
    global model
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image for model inference"""
    try:
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize(IMG_SIZE)

        img_array = np.array(image)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image format")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def root():
    return {
        "message": "VEya Model API",
        "version": "1.0.0",
        "status": "online",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": CLASS_NAMES
    }

@app.post("/api/analyze")
async def analyze_eye(file: UploadFile = File(...)):
    """
    Analyze eye image for disease detection

    Returns:
        - prediction: predicted class name
        - confidence: confidence score for predicted class
        - probabilities: probabilities for all classes
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()

        img_array = preprocess_image(contents)

        predictions = model.predict(img_array, verbose=0)

        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])

        probabilities = {
            class_name: float(prob)
            for class_name, prob in zip(CLASS_NAMES, predictions[0])
        }

        logger.info(f"Analysis complete: {predicted_class} ({confidence:.2%})")

        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_version": "1.0.0"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
