import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

MODEL_PATH = "plant_model.h5"
IMG_SIZE = (128, 128)

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Trained model not found. Run plant_classifier.py first.")
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (same order as training)
CLASS_LABELS = ["angular_leaf_spot", "bean_rust", "healthy"]

print("Plant Disease Classifier Ready! Enter image path (Ctrl+C to quit):")
while True:
    img_path = input("> ").strip()
    if not os.path.exists(img_path):
        print("File not found, try again.")
        continue

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)

    print(f"Prediction: {CLASS_LABELS[class_idx]} (confidence: {confidence:.3f})")
