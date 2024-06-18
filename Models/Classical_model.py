import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from data_preprocessing import load_and_preprocess_data

# Load the dataset
train_images = '/content/drive/My Drive/FoodAllergyData/train'
test_images = '/content/drive/My Drive/FoodAllergyData/test'
train_labels = '/content/drive/My Drive/FoodAllergyData/FoodAllergy-CV/Data/annotations_train.csv' 
test_labels = '/content/drive/My Drive/FoodAllergyData/FoodAllergy-CV/Data/annotations_test.csv'  
model_save_path = '/content/drive/MyDrive/FoodAllergyData/FoodAllergy-CV/Models/deep_learning_model.h5'

target_size = (64, 64)

# Preprocess the data
 # Load and preprocess data
train_images, train_labels, test_images, test_labels, _, _ = load_and_preprocess_data(
        train_images, test_images, train_labels, test_labels, target_size)

# Flatten or reshape images into 2D arrays
train_images_flat = train_images.reshape(train_images.shape[0], -1)
test_images_flat = test_images.reshape(test_images.shape[0], -1)

# Encode the labels
label_encoder = LabelEncoder()
train_labels_enc = label_encoder.fit_transform(train_labels)
test_labels_enc = label_encoder.transform(test_labels)

# Train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(train_images_flat, train_labels_enc)

# Evaluate the model
train_pred = clf.predict(train_images)
test_pred = clf.predict(test_images)

train_accuracy = accuracy_score(train_labels_enc, train_pred)
test_accuracy = accuracy_score(test_labels_enc, test_pred)

train_f1 = f1_score(train_labels_enc, train_pred, average='macro')
test_f1 = f1_score(test_labels_enc, test_pred, average='macro')

print(f'Train Accuracy: {train_accuracy:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')
print(f'Train F1-score: {train_f1:.2f}')
print(f'Test F1-score: {test_f1:.2f}')
