# src/classical_ml.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_preprocessing import load_and_preprocess_data

def main():
    # Paths
    train_folder = '/content/drive/MyDrive/FoodAllergyData/train'
    test_folder = '/content/drive/MyDrive/FoodAllergyData/test'
    
    # Parameters
    target_size = (224, 224)  # Resize to this size
    
    # Load and preprocess data
    train_images, train_labels, test_images, test_labels = load_and_preprocess_data(train_folder, test_folder, target_size)
    
    # Flatten images for classical ML
    train_images_flat = train_images.reshape(len(train_images), -1)
    test_images_flat = test_images.reshape(len(test_images), -1)
    
    # Train a RandomForest Classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_images_flat, train_labels)
    
    # Predict and evaluate
    predictions = clf.predict(test_images_flat)
    accuracy = accuracy_score(test_labels, predictions)
    print(f'Classical ML Model Accuracy: {accuracy}')
    
if __name__ == "__main__":
    main()
