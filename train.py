import os
import ssl
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm  # For progress bar

#  Fix SSL certificate issue for downloading model weights
ssl._create_default_https_context = ssl._create_unverified_context

#  Dataset Path
dataset_path = "/Users/suryatopsasmal/Downloads/Projects/Animal Classification/dataset"

#  Data Processing Parameters
img_size = (224, 224)
batch_size = 32
epochs = 10

#  Data Augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% train, 20% validation
)

#  Load Training Data
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

#  Load Validation Data
val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

#  Load Pretrained VGG16 Model (Without Top Layers)
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze pretrained layers

#  Build Custom Classification Model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(15, activation="softmax")  # 15 animal classes
])

#  Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

#  Train the Model (With tqdm Progress Bar)
print(" Training started...")
for epoch in tqdm(range(epochs), desc="Epoch Progress", unit="epoch"):
    model.fit(train_generator, validation_data=val_generator, epochs=1, verbose=1)

#  Save Model
model.save("animal_classifier.h5")
print(" Model saved as 'animal_classifier.h5'")
