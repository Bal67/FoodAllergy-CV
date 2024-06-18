import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC  # Import SVC from sklearn.svm
from sklearn.metrics import accuracy_score, f1_score
from data_preprocessing import load_and_preprocess_data

# Load the dataset
train_images = '/content/drive/My Drive/FoodAllergyData/train'
test_images = '/content/drive/My Drive/FoodAllergyData/test'
train_labels = '/content/drive/My Drive/FoodAllergyData/FoodAllergy-CV/Data/annotations_train.csv' 
test_labels = '/content/drive/My Drive/FoodAllergyData/FoodAllergy-CV/Data/annotations_test.csv'  
model_save_path = '/content/drive/MyDrive/FoodAllergyData/FoodAllergy-CV/Models/deep_learning_model.h5'

target_size = (64, 64)

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

# Normalize pixel values to improve convergence
scaler = StandardScaler()
train_images_scaled = scaler.fit_transform(train_images_flat)
test_images_scaled = scaler.transform(test_images_flat)

# Parameter grid for GridSearchCV
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'gamma': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

# Create SVM classifier
svm_clf = SVC()

# Perform grid search with cross-validation
grid_search = GridSearchCV(svm_clf, param_grid, cv=3, scoring='accuracy', verbose=2)
grid_search.fit(train_images_scaled, train_labels_enc)

# Print best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-validation Accuracy:", grid_search.best_score_)

# Get the best model from grid search
best_svm_clf = grid_search.best_estimator_


# Predictions on training set
train_pred = best_svm_clf.predict(train_images_scaled)
train_accuracy = accuracy_score(train_labels_enc, train_pred)
train_f1 = f1_score(train_labels_enc, train_pred, average='macro')

# Predictions on test set
test_pred = best_svm_clf.predict(test_images_scaled)
test_accuracy = accuracy_score(test_labels_enc, test_pred)
test_f1 = f1_score(test_labels_enc, test_pred, average='macro')

# Print evaluation metrics
print(f'Train Accuracy: {train_accuracy:.2f}')
print(f'Train F1-score: {train_f1:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')
print(f'Test F1-score: {test_f1:.2f}')