import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# CONFIG 
DATASET_DIR = "data/Bean_Dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = "plant_model.h5"

# 1. Data generators (with split)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  # 80% train, 20% val
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False  # keep order for metrics
)

# 2. Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(3, activation="softmax")  # 3 classes
])

# 3. Compile
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 4. Train
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# 5. Evaluate
loss, acc = model.evaluate(val_data)
print(f"\nValidation Accuracy: {acc:.3f}")

# 6. Classification report + confusion matrix
y_true = val_data.classes
y_pred = np.argmax(model.predict(val_data), axis=1)
class_labels = list(val_data.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# 7. Save model
model.save(MODEL_PATH)
print(f"\n✅ Model saved as {MODEL_PATH}")

# 8. Plot training history
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend()
plt.show()
