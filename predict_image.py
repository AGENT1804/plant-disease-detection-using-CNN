import cv2
import numpy as np
from tensorflow.keras.models import load_model


# Load and preprocess the image for prediction
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (32, 32))  # Resize to match CIFAR-10 model input size (32x32)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Load the pre-trained model
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

# Predict class
def predict_class(model, image):
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class

# Main function to execute the prediction
def main():
    # Step 1: Load the saved model
    model_path = r'F:\Final year project model\model.h5'  # Path to the saved model
    model = load_trained_model(model_path)
    
    # Step 2: Load and preprocess the image
    image_path = r'C:\Users\dilip\Downloads\Some-leaf-images-from-the-PlantVillage-dataset-The-name-of-the-leaves-and-the-diseases.png'  # Image path
    image = preprocess_image(image_path)
    
    # Step 3: Predict the class of the image
    prediction = predict_class(model, image)
    print(f"Predicted Class: {prediction}")

if __name__ == "__main__":
    main()
