import tensorflow as tf
import os
from sklearn.model_selection import train_test_split

from data_loader import class_names, train_images
from data_generator import create_dataset
from model import create_car_classifier
from tensorflow.keras import mixed_precision

def setup_gpu():
    """Configure GPU settings"""
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU available: {len(gpus)} device(s)")
            print(f"   Primary GPU: {gpus[0]}")
        except RuntimeError as e:
            print(f"⚠️ GPU setup error: {e}")
    else:
        print("❌ No GPU found - using CPU")
    mixed_precision.set_global_policy('mixed_float16')
    # Use the non-deprecated way to check GPU
    print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")


def main():
    print("🚗 Training Car Classifier")

    setup_gpu()

    # Split data
    train_split, val_split = train_test_split(
        train_images, test_size=0.2, random_state=42,
        stratify=[img.label for img in train_images]
    )

    print(f"Train: {len(train_split)}, Val: {len(val_split)}")

    # Create datasets
    train_ds = create_dataset(
        train_split,
        'stanford-cars-dataset/versions/1/cars_train/cars_train',
        batch_size=64,
        shuffle=True
    )

    val_ds = create_dataset(
        val_split,
        'stanford-cars-dataset/versions/1/cars_train/cars_train',
        batch_size=64,
        shuffle=False
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6)
    ]

    # Create and compile model
    print("Creating model...")
    model = create_car_classifier(num_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    print("Starting training...")
    history = model.fit(
        train_ds,
        batch_size=64,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks,
        use_multiprocessing=True
    )

    # Save
    os.makedirs('outputs', exist_ok=True)
    model.save('outputs/car_model.keras')
    model.trainable = True
    for layer in model.layers[:-30]:  # freeze all but last ~30 layers
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    # Train
    print("Starting training again...")
    history = model.fit(
        train_ds,
        batch_size=64,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks,
        use_multiprocessing=True
    )

    print(f"✅ Final accuracy: {history.history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()