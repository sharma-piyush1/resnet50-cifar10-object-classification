import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50


# ------------------------------------------------------
# Load dataset
# ------------------------------------------------------
def load_labels(csv_path):
    labels_df = pd.read_csv(csv_path)
    label_map = {label: i for i, label in enumerate(sorted(labels_df['label'].unique()))}
    y = np.array([label_map[i] for i in labels_df['label']])
    ids = list(labels_df['id'])
    return ids, y, label_map


def load_images(image_folder, ids):
    data = []
    for img_id in ids:
        img_path = os.path.join(image_folder, f"{img_id}.png")
        image = Image.open(img_path)
        image = image.resize((256, 256))   # ResNet50 input size
        image = np.array(image)
        data.append(image)
    return np.array(data)


# ------------------------------------------------------
# Build ResNet-50 based model
# ------------------------------------------------------
def build_model(num_classes=10):
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(256, 256, 3)
    )

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=2e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ------------------------------------------------------
# Train model
# ------------------------------------------------------
def train_model(X_train, y_train):
    model = build_model()
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=10,
        verbose=1
    )
    return model, history


# ------------------------------------------------------
# Plot metrics
# ------------------------------------------------------
def plot_history(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label="Train Accuracy")
    plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.show()


# ------------------------------------------------------
# Main
# ------------------------------------------------------
def main():

    labels_csv = "trainLabels.csv"
    image_folder = "train"

    print("Loading labels...")
    ids, y, label_map = load_labels(labels_csv)

    print("Loading & processing images... (takes time)")
    X = load_images(image_folder, ids)

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2
    )

    # scale
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    print("Training model...")
    model, history = train_model(X_train, y_train)

    print("Evaluating...")
    loss, acc = model.evaluate(X_test, y_test)
    print("Test Accuracy:", acc)

    print("Plotting history...")
    plot_history(history)

    print("Saving model...")
    model.save("resnet50_cifar10_model.h5")
    print("Model saved as resnet50_cifar10_model.h5")


if __name__ == "__main__":
    main()
