# Handwritten Digit Recognition using MNIST Dataset
# ------------------------------------------------
# MICRO PROJECT CODE
# Author: Neil Borikar
# Department: Artificial Intelligence and Data Science
# College: Shah & Anchor Kutchhi Engineering College
# ------------------------------------------------
# This project demonstrates how a simple Neural Network
# can recognize handwritten digits (0–9) using the MNIST dataset.

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report

# ------------------------------------------------
# 1. Load and Explore the MNIST Dataset
# ------------------------------------------------
# MNIST dataset comes preloaded with Keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("Training Data Shape:", x_train.shape)
print("Testing Data Shape:", x_test.shape)

# Display a few sample images
plt.figure(figsize=(6, 3))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")
plt.suptitle("Sample Handwritten Digits from MNIST Dataset")
plt.show()

# ------------------------------------------------
# 2. Data Preprocessing
# ------------------------------------------------
# Normalize pixel values (0-255) to range (0-1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten 28x28 images into a single vector of 784 features
x_train = x_train.reshape((x_train.shape[0], 28 * 28))
x_test = x_test.reshape((x_test.shape[0], 28 * 28))

# Convert labels to one-hot encoded format
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# ------------------------------------------------
# 3. Build the Neural Network Model
# ------------------------------------------------
# Using Sequential model with 2 hidden layers
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes (digits 0–9)
])

# ------------------------------------------------
# 4. Compile the Model
# ------------------------------------------------
model.compile(
    optimizer='adam',                   # Optimization algorithm
    loss='categorical_crossentropy',    # Loss function for multi-class classification
    metrics=['accuracy']                # Track accuracy during training
)

# ------------------------------------------------
# 5. Train the Model
# ------------------------------------------------
history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=32,
    verbose=1
)

# ------------------------------------------------
# 6. Evaluate the Model on Test Data
# ------------------------------------------------
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# ------------------------------------------------
# 7. Visualize Training Progress
# ------------------------------------------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 8. Generate Predictions and Confusion Matrix
# ------------------------------------------------
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:\n", classification_report(y_true, y_pred_classes))

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.colorbar()
plt.show()

# ------------------------------------------------
# 9. Test with Random Samples
# ------------------------------------------------
indices = np.random.choice(len(x_test), 6)
sample_images = x_test[indices]
sample_labels = y_true[indices]
sample_preds = np.argmax(model.predict(sample_images), axis=1)

plt.figure(figsize=(6, 3))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(sample_images[i].reshape(28, 28), cmap="gray")
    plt.title(f"Pred: {sample_preds[i]}, True: {sample_labels[i]}")
    plt.axis("off")
plt.suptitle("Model Predictions on Random Test Images")
plt.show()

# ------------------------------------------------
# 10. Save the Model
# ------------------------------------------------
model.save("mnist_digit_recognition_model.h5")
print("\nModel saved successfully as 'mnist_digit_recognition_model.h5'")

# ------------------------------------------------
# End of Micro Project
# ------------------------------------------------
