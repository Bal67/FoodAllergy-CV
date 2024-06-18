import numpy as np
import cv2
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_preprocessing import load_and_preprocess_data

def extract_features(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Compute the color histogram
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], 
                        [0, 180, 0, 256, 0, 256])
    # Normalize the histogram
    cv2.normalize(hist, hist)
    return hist.flatten()

def main():
    # Paths
    train_folder = '/content/drive/My Drive/FoodAllergyData/train'
    test_folder = '/content/drive/My Drive/FoodAllergyData/test'
    train_annotations = '/content/drive/My Drive/FoodAllergyData/FoodAllergy-CV/Data/annotations_train.csv' 
    test_annotations = '/content/drive/My Drive/FoodAllergyData/FoodAllergy-CV/Data/annotations_test.csv' 
    
  # Parameters
    target_size = (224, 224)
    
    # Load and preprocess data
    train_images, train_labels, test_images, test_labels, train_annotations_df, test_annotations_df = load_and_preprocess_data(
        train_folder, test_folder, train_annotations, test_annotations, target_size
    )
    
    # Extract features from images
    train_features = np.array([extract_features(img) for img in train_images])
    test_features = np.array([extract_features(img) for img in test_images])
    
    # Train a Random Forest classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(train_features, train_labels)
    
    # Make predictions
    predictions = classifier.predict(test_features)
    
    # Evaluate the model
    accuracy = accuracy_score(test_labels, predictions)
    print(f'Naive Approach Accuracy: {accuracy}')

if __name__ == "__main__":
    main()