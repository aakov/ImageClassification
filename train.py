import tensorflow as tf
import os
from sklearn.model_selection import train_test_split

from data_loader import class_names, train_images
from data_generator import create_dataset
from model import create_car_classifier


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

    # Create and compile model
    print("Creating model...")
    model = create_car_classifier(num_classes=len(class_names))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ]
    )

    # Save
    os.makedirs('outputs', exist_ok=True)
    model.save('outputs/car_model.keras')

    print(f"✅ Final accuracy: {history.history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()