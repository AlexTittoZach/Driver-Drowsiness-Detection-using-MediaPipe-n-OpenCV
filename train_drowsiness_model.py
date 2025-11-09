import os
import json
import math
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# Paths
DATASET_DIR = "Driver Drowsiness Dataset (DDD)"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "drowsiness_mobilenetv2.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "class_indices.json")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15


def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)


def build_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        shear_range=0.1,
        brightness_range=(0.7, 1.3),
        horizontal_flip=True,
        fill_mode="nearest",
    )

    common_args = dict(
        directory=DATASET_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
    )

    train_gen = train_datagen.flow_from_directory(
        subset="training",
        **common_args,
    )
    val_gen = train_datagen.flow_from_directory(
        subset="validation",
        **common_args,
    )

    return train_gen, val_gen


def build_model():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    # Freeze most layers for a stable start
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.2)(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    ensure_dirs()

    train_gen, val_gen = build_data_generators()

    # Persist class mapping
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    model = build_model()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    steps_per_epoch = math.ceil(train_gen.samples / BATCH_SIZE)
    validation_steps = math.ceil(val_gen.samples / BATCH_SIZE)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    # Final evaluation
    val_loss, val_acc = model.evaluate(val_gen, verbose=1)
    print(f"Validation accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()


