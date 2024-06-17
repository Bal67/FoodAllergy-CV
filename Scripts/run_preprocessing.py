import sys
import os

# Ensure the Scripts directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from data_preprocessing import load_and_preprocess_data

# Define the paths to your data folders and CSV files
train_folder = '/content/drive/My Drive/FoodAllergyData/train'
train_csv = '/content/drive/My Drive/FoodAllergyData/annotations_train.csv'
test_folder = '/content/drive/My Drive/FoodAllergyData/test'
test_csv = '/content/drive/My Drive/FoodAllergyData/annotations_test.csv'
target_size = (224, 224)  # Example target size

# Load and preprocess data
train_images, train_labels, test_images, test_labels = load_and_preprocess_data(train_folder, train_csv, test_folder, test_csv, target_size)

# Print shapes to verify
print(f"Train images shape: {train_images.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")
