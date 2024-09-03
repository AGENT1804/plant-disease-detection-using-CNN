import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the paths
base_dir = r'F:/Mango/CODES/MangoLeafBD Dataset'  # Use raw string to avoid issues with backslashes
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Define classes
classes = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Scooty Mould']

# Create directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

print(f"Base directory: {base_dir}")
print(f"Training data directory: {train_dir}")
print(f"Validation data directory: {validation_dir}")

for class_name in classes:
    class_train_dir = os.path.join(train_dir, class_name)
    class_validation_dir = os.path.join(validation_dir, class_name)
    os.makedirs(class_train_dir, exist_ok=True)
    os.makedirs(class_validation_dir, exist_ok=True)
    
    print(f"Training directory for '{class_name}': {class_train_dir}")
    print(f"Validation directory for '{class_name}': {class_validation_dir}")

print("Directories created successfully.")

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

print(f"Loading training data from: {train_dir}")
print(f"Loading validation data from: {validation_dir}")

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Adjust based on your model's input size
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print("Data generators created successfully.")
