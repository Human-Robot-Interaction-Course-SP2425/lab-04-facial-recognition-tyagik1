"""
Training Script for Facial Emotion Recognition Model

This implementation is based on and adapted from:
    "hand-gesture-recognition-using-mediapipe" by Kazuhito Takahashi (Kazuhito00)
    Original repository: https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe
    Licensed under Apache License 2.0

Improved version:
- Feature normalization (StandardScaler)
- Better MLP architecture (BatchNorm + reduced Dropout)
- Proper train/val/test split
- Class imbalance handling
- Improved learning rate scheduling
"""

import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# Emotion labels
EMOTIONS = ["happy", "sad", "neutral", "surprised"]
NUM_CLASSES = len(EMOTIONS)


def load_dataset(csv_path='./apps/facial_recognition/csv/keypoint.csv'):
    X = []
    y = []

    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if len(row) > 1:
                    label = int(row[0])
                    landmarks = [float(x) for x in row[1:]]
                    X.append(landmarks)
                    y.append(label)
    except FileNotFoundError:
        print(f"Dataset file {csv_path} not found!")
        return None, None

    if not X:
        print("No training data found!")
        return None, None

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('./apps/facial_recognition/csv/confusion_matrix.png')
    plt.show()


def main():
    print("Loading dataset...")
    X, y = load_dataset()

    if X is None:
        return

    print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    print(f"Class weights: {class_weights}")

    model = create_model(X.shape[1], NUM_CLASSES)
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=5,
        min_lr=1e-5
    )

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights,
        verbose=1
    )

    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=EMOTIONS))

    plot_confusion_matrix(y_test, y_pred_classes, EMOTIONS)

    print("Converting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open('./models/emotion_classifier.tflite', 'wb') as f:
        f.write(tflite_model)

    print("Model saved!")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig('./apps/facial_recognition/csv/training_history.png')
    plt.show()

    print("Training complete!")


if __name__ == "__main__":
    main()