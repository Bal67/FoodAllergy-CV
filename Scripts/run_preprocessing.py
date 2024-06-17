# run_preprocessing.py
from src.data_preprocessing import load_and_preprocess_data

# Define the paths to your data folders
train_folder = '/content/drive/My Drive/FoodAllergyData/train'
test_folder = '/content/drive/My Drive/FoodAllergyData/test'
target_size = (224, 224)  

# Load and preprocess data
train_images, train_labels, test_images, test_labels = load_and_preprocess_data(train_folder, test_folder, target_size)

# Print shapes to verify
print(f"Train images shape: {train_images.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")
