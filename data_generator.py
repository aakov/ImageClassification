import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow.keras.applications.resnet50 import preprocess_input


def preprocess_image(image_path, bbox):
    image = Image.open(image_path).convert('RGB')
    x1, y1, x2, y2 = bbox
    if np.random.rand() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Random brightness
    if np.random.rand() < 0.5:
        factor = 1.0 + (np.random.rand() - 0.5) * 0.4  # ±20% brightness
        image = Image.fromarray(np.clip(np.array(image) * factor, 0, 255).astype(np.uint8))
    image = image.crop((x1, y1, x2, y2))
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32)
    image = preprocess_input(image)  # Use ResNet50 preprocessing
    # Random horizontal flip

    return image


def create_dataset(car_images, images_dir, batch_size, shuffle=True):
    def generator():
        indices = list(range(len(car_images)))
        if shuffle:
            np.random.shuffle(indices)

        for i in indices:
            car_image = car_images[i]
            image_path = os.path.join(images_dir, car_image.filename)
            image = preprocess_image(image_path, car_image.bbox)
            yield image, car_image.label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)