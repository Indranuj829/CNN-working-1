import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# 1. DOWNLOAD & FIX DATASET PATH
# ==========================================
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

# Fix for nested directory issue: Navigate inside if 'flower_photos' is sub-nested
if (data_dir / 'flower_photos').exists() and (data_dir / 'flower_photos').is_dir():
    data_dir = data_dir / 'flower_photos'

# ==========================================
# 2. LOAD DATASET & VERIFY CLASSES
# ==========================================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names

# MUST PRINT 5 CLASSES: ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
print("\nSuccess! Detected Classes:", class_names)

# Optimize memory pipeline in Colab
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 3. BUILD DEEP CNN ARCHITECTURE
# ==========================================
model = models.Sequential([
    # Fixed syntax: Rescaling 0-255 RGB pixel values down to 0.0-1.0
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),

    # Conv Block 1: Low-level features (edges, colors)
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    # Conv Block 2: Mid-level features (textures, petals)
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    # Conv Block 3: High-level features (full flower patterns)
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    # Classifier Head
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# ==========================================
# 4. TRAIN THE CNN
# ==========================================
print("\nTraining on real images (5 Epochs)...")
history = model.fit(train_ds, validation_data=val_ds, epochs=5)

# ==========================================
# 5. VISUALIZE PREDICTIONS ON REAL TEST IMAGES
# ==========================================
plt.figure(figsize=(10, 10))
for images, labels in val_ds.take(1):
    predictions = model.predict(images)

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)

        # Display image
        plt.imshow(images[i].numpy().astype("uint8"))

        # Predict class & confidence
        predicted_class = class_names[np.argmax(predictions[i])]
        actual_class = class_names[labels[i]]
        confidence = np.max(predictions[i]) * 100

        # Color: Green if correct prediction, Red if wrong
        color = 'green' if predicted_class == actual_class else 'red'

        plt.title(f"Pred: {predicted_class} ({confidence:.1f}%)\nTrue: {actual_class}",
                  color=color, fontsize=10)
        plt.axis("off")

plt.tight_layout()
plt.show()
