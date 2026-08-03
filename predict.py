import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load the trained model
MODEL_PATH = os.path.join("model", "plant_model.h5")
model = load_model(MODEL_PATH)

# Class labels (must match your training dataset)
classes = ["Diseased", "Healthy"]


def predict_disease(img_path):
    # Load image
    img = image.load_img(img_path, target_size=(128, 128))

    # Convert image to array
    img_array = image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    predicted_class = classes[predicted_index]

    confidence = round(float(np.max(prediction)) * 100, 2)

    return predicted_class, confidence