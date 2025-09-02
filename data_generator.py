import tensorflow as tf
import os
from tensorflow.keras.applications.resnet50 import preprocess_input

def preprocess_image_tf(image_bytes, bbox):

    image = tf.image.decode_jpeg(image_bytes, channels=3)

    # Crop to bounding box
    x1, y1, x2, y2 = tf.unstack(tf.cast(bbox, tf.int32))
    image = tf.image.crop_to_bounding_box(image, y1, x1, y2 - y1, x2 - x1)

    image = tf.image.resize(image, [224, 224])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)

    # ResNet50 preprocessing
    image = preprocess_input(image)

    return image

def create_dataset(car_images, images_dir, batch_size, shuffle=True):
    # Make them tensors
    filepaths = [os.path.join(images_dir, ci.filename) for ci in car_images]
    bboxes = [ci.bbox for ci in car_images]
    labels = [ci.label for ci in car_images]

    ds = tf.data.Dataset.from_tensor_slices((filepaths, bboxes, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=batch_size * 10)

    # Load & preprocess
    def load_and_preprocess(path, bbox, label):
        image_bytes = tf.io.read_file(path)
        image = preprocess_image_tf(image_bytes, bbox)
        return image, label

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch & prefetch
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
