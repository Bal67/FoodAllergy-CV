import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score
from data_preprocessing import load_and_preprocess_data
from keras.preprocessing.image import ImageDataGenerator
from scipy.stats import expon

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

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

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

# Parameter distribution for RandomizedSearchCV
param_dist = {
    'C': expon(scale=1),
    'gamma': expon(scale=0.1),
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'class_weight': [None, 'balanced']
}

# Create SVM classifier
svm_clf = SVC()

# Perform randomized search with cross-validation
random_search = RandomizedSearchCV(svm_clf, param_dist, n_iter=50, cv=3, scoring='accuracy', verbose=2, random_state=42)
random_search.fit(train_images_scaled, train_labels_enc)

# Print best parameters and best score
print("Best Parameters:", random_search.best_params_)
print("Best Cross-validation Accuracy:", random_search.best_score_)

# Get the best model from random search
best_svm_clf = random_search.best_estimator_

# Feature Selection
selector = SelectKBest(f_classif, k=500)
train_images_selected = selector.fit_transform(train_images_scaled, train_labels_enc)
test_images_selected = selector.transform(test_images_scaled)

# Create Ridge classifier
ridge_clf = RidgeClassifier()

# Create ensemble of different classifiers
ensemble_clf = VotingClassifier(
    estimators=[
        ('svm', best_svm_clf),
        ('ridge', ridge_clf)
    ],
    voting='hard'
)

# Split train data into train and validation sets for early stopping
X_train, X_val, y_train, y_val = train_test_split(train_images_selected, train_labels_enc, test_size=0.2, random_state=42)

# Fit the ensemble model with early stopping
ensemble_clf.fit(X_train, y_train)

# Validate the model
val_pred = ensemble_clf.predict(X_val)
val_accuracy = accuracy_score(y_val, val_pred)
val_f1 = f1_score(y_val, val_pred, average='macro')

# Print validation metrics
print(f'Validation Accuracy: {val_accuracy:.2f}')
print(f'Validation F1-score: {val_f1:.2f}')

# Evaluate on test set
test_pred = ensemble_clf.predict(test_images_selected)
test_accuracy = accuracy_score(test_labels_enc, test_pred)
test_f1 = f1_score(test_labels_enc, test_pred, average='macro')

# Print evaluation metrics
print(f'Test Accuracy: {test_accuracy:.2f}')
print(f'Test F1-score: {test_f1:.2f}')
