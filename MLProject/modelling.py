"""
modelling.py (untuk MLProject)
Script training model yang digunakan di dalam MLflow Project.

Ini versi yang jalan di GitHub Actions CI — sama persis dengan modelling.py
di Membangun_model, tapi path dataset disesuaikan dengan struktur MLProject.

Author: M. Rohid Rivaldi (rezoku)
"""

import os
import dagshub
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

# ===== KONFIGURASI =====
# path dataset relatif terhadap MLProject folder
DATA_DIR = os.environ.get("DATA_DIR", "./garbage_preprocessing")
IMG_SIZE = (224, 224)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
NUM_CLASSES = 12

KELAS = [
    "battery", "biological", "brown-glass", "cardboard",
    "clothes", "green-glass", "metal", "paper",
    "plastic", "shoes", "trash", "white-glass"
]

# ===== SETUP DAGSHUB =====
# Kita langsung pakai standard MLflow tracking credentials & URI agar non-interaktif dan tidak ada prompt OAuth.
token = os.environ.get("DAGSHUB_TOKEN") or os.environ.get("DAGSHUB_USER_TOKEN") or "50cfb85288aacb94621c6fc91b6fb53dc0e3dd4f"
os.environ["MLFLOW_TRACKING_USERNAME"] = "rohidrivaldi"
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

mlflow.set_tracking_uri("https://dagshub.com/rohidrivaldi/Workflow-CI_M._Rohid_Rivaldi.mlflow")

mlflow.tensorflow.autolog(log_models=True)
print("DagsHub + MLflow ready!")
print("Tracking URI:", mlflow.get_tracking_uri())


def load_dataset(data_dir: str):
    """Load dataset dan split 80/20."""
    
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )
    
    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )
    
    # normalisasi + pipeline optimization
    norm = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (norm(x), y))
    val_ds = val_ds.map(lambda x, y: (norm(x), y))
    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    
    print(f"Train: {len(train_ds)} batches, Val: {len(val_ds)} batches")
    return train_ds, val_ds


def build_model():
    """Build model EfficientNetB0."""
    
    base_model = keras.applications.EfficientNetB0(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


def main():
    print("\n" + "="*60)
    print("  MLProject CI Run — Garbage Classification")
    print("  by M. Rohid Rivaldi (rezoku)")
    print("="*60 + "\n")
    
    train_ds, val_ds = load_dataset(DATA_DIR)
    model = build_model()
    
    with mlflow.start_run(run_name="ci_training_run"):
        mlflow.log_param("data_dir", DATA_DIR)
        mlflow.log_param("base_model", "EfficientNetB0")
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2)
        ]
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        best_val_acc = max(history.history["val_accuracy"])
        mlflow.log_metric("best_val_accuracy", best_val_acc)
        
        print(f"\n✅ Training selesai! Best Val Acc: {best_val_acc:.4f}")
        
        # simpan model
        model.export("./model_output")
        print("Model disimpan ke ./model_output")

        # simpan model dalam format MLflow secara lokal untuk offline Docker build
        import shutil
        if os.path.exists("./mlflow_model"):
            shutil.rmtree("./mlflow_model")
        mlflow.tensorflow.save_model(model, path="./mlflow_model", conda_env="./conda.yaml")
        print("MLflow model disimpan secara lokal ke ./mlflow_model dengan conda.yaml")


if __name__ == "__main__":
    main()

# Artifact step added
