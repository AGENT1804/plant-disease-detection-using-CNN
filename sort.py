import os

# Define paths
base_dir = 'F:\\Mango\\CODES\\MangoLeafBD Dataset'  # Replace with your base directory path
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Define classes
classes = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Scooty Mould']  # Replace with your actual class names

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
