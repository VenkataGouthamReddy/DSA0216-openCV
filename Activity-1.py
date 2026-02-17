from google.colab import files
uploaded = files.upload()
import zipfile
import os

with zipfile.ZipFile("archive.zip", 'r') as zip_ref:
    zip_ref.extractall("dataset")

print("Dataset extracted successfully!")
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    "dataset",
    target_size=(150,150),
    batch_size=32,
    class_mode='binary',   # change to 'categorical' if more than 2 classes
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    "dataset",
    target_size=(150,150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)
model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # binary classification
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)
loss, accuracy = model.evaluate(validation_generator)
print("Validation Accuracy:", accuracy)
