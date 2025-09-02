import tensorflow as tf
import numpy as np
from PIL import Image
import os

from data_loader import class_names   # class_names already loaded in your project
from tensorflow.keras.applications.resnet50 import preprocess_input

MODEL_PATH = "outputs/car_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path, bbox=None):

    image = Image.open(image_path).convert("RGB")

    if bbox:  # crop if bounding box provided
        x1, y1, x2, y2 = bbox
        image = image.crop((x1, y1, x2, y2))

    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32)
    image = preprocess_input(image)  # ResNet50 preprocessing
    return np.expand_dims(image, axis=0)  # add batch dim

def predict_car(image_path, bbox=None):
    image = preprocess_image(image_path, bbox)
    preds = model.predict(image)
    class_id = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds, axis=1)[0]
    return class_names[class_id], float(confidence)

if __name__ == "__main__":
    img_path = "car_img_test.jpg"
    # bbox = (30, 52, 246, 147)  # optional if you want cropping
    car, conf = predict_car(img_path)
    print(f"🚘 Predicted: {car} (confidence: {conf:.2f})")
