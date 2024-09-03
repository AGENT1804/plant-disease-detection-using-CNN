import os
import sqlite3
from collections import Counter

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = 'F:\Mango\CODES\MangoLeafBD Dataset'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

classes = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Scooty Mould']

os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

for class_name in classes:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, class_name), exist_ok=True)

print("Directories created successfully.")

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

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

def create_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def compile_model(model):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

def train_model(model, train_generator, validation_generator):
    history = model.fit(
        train_generator,
        epochs=2,
        validation_data=validation_generator
    )
    return history

def save_model(model, model_path):
    model.save(model_path)

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def load_trained_model(model_path):
    model = load_model(model_path)
    return model

def predict_image(model, image, class_names):
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

def save_to_db(image_path, predicted_class):
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    
    # Create the table if it does not exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            predicted_class TEXT NOT NULL
        )
    ''')
    
    cursor.execute('''
        INSERT INTO predictions (image_path, predicted_class)
        VALUES (?, ?)
    ''', (image_path, predicted_class))
    
    conn.commit()
    conn.close()

def show_db_contents():
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    
    # Ensure that the table exists before querying it
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            predicted_class TEXT NOT NULL
        )
    ''')
    
    cursor.execute('SELECT predicted_class FROM predictions')
    rows = cursor.fetchall()
    
    if rows:
        predictions = [row[0] for row in rows]
        most_common = Counter(predictions).most_common(1)[0][0]
        print(f"Most Frequent Prediction: {most_common}")
    else:
        print("No predictions found in the database.")
    
    conn.close()

def save_most_frequent_prediction(predictions):
    if not predictions:
        return
    
    most_common = Counter(predictions).most_common(1)[0][0]
    most_common_dir = os.path.join(base_dir, 'most_common_predictions')
    os.makedirs(most_common_dir, exist_ok=True)
    
    with open(os.path.join(most_common_dir, 'most_common_prediction.txt'), 'w') as f:
        f.write(most_common)
    
    print(f"Most frequent prediction saved to {most_common_dir}/most_common_prediction.txt")

def real_time_classification(model, class_names):
    cap = cv2.VideoCapture(0)
    predictions = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        processed_image = preprocess_image(frame)
        prediction = predict_image(model, processed_image, class_names)
        predictions.append(prediction)

        cv2.putText(frame, f'Predicted Class: {prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Real-time Classification', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    save_most_frequent_prediction(predictions)

def main():
    num_classes = len(train_generator.class_indices)
    class_names = list(train_generator.class_indices.keys())
    
    model = create_model(num_classes)
    compile_model(model)
    
    train_model(model, train_generator, validation_generator)
    
    model_path = 'F:/plants/plants 1/model.h5'
    save_model(model, model_path)
    print(f"Model saved to {model_path}")
    
    model = load_trained_model(model_path)
    
    real_time_classification(model, class_names)
    
    # Ensure the table exists before querying
    save_to_db('', '')  # Dummy call to ensure table creation
    show_db_contents()

if __name__ == "__main__":
    main()
