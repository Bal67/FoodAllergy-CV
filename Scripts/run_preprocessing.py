import os
from data_preprocessing import load_and_preprocess_data

# Define paths and parameters
train_folder = '/content/drive/MyDrive/FoodAllergyData/train'
test_folder = '/content/drive/MyDrive/FoodAllergyData/test'
train_annotations = '/content/drive/MyDrive/FoodAllergyData/FoodAllergy-CV/Data/annotations_train.csv'
test_annotations = '/content/drive/MyDrive/FoodAllergyData/FoodAllergy-CV/Data/annotations_test.csv'
target_size = (224, 224)

# Load and preprocess data (excluding validation folder)
train_images, train_labels, test_images, test_labels, train_annotations_df, test_annotations_df = load_and_preprocess_data(
    train_folder, test_folder, train_annotations, test_annotations, target_size)

# Print shapes to verify
print(f"Train images shape: {train_images.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")