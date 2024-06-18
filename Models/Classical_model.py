import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from data_preprocessing import load_and_preprocess_data  # Adjust the import path as needed

def extract_features(image):
    # Resize image to a fixed size
    image = cv2.resize(image, (128, 128))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract HOG features
    hog = cv2.HOGDescriptor()
    h = hog.compute(gray)
    
    # Flatten the HOG features
    features = h.flatten()
    
    return features

def main():
    # Paths
    train_folder = '/content/drive/My Drive/FoodAllergyData/train'
    test_folder = '/content/drive/My Drive/FoodAllergyData/test'
    train_annotations = '/content/drive/My Drive/FoodAllergyData/FoodAllergy-CV/Data/annotations_train.csv' 
    test_annotations = '/content/drive/My Drive/FoodAllergyData/FoodAllergy-CV/Data/annotations_test.csv' 
    
    # Load and preprocess data
    target_size = (224, 224)  
    train_images, train_labels, test_images, test_labels, _, _ = load_and_preprocess_data(
        train_folder, test_folder, train_annotations, test_annotations, target_size)
    
    # Extract features from images
    train_features = np.array([extract_features(img) for img in train_images])
    test_features = np.array([extract_features(img) for img in test_images])
    
    # Train a simple classifier (e.g., SVM)
    classifier = SVC(kernel='linear', random_state=42)
    classifier.fit(train_features, train_labels)
    
    # Make predictions
    predictions = classifier.predict(test_features)
    
    # Evaluate the classifier
    accuracy = accuracy_score(test_labels, predictions)
    print(f'Naive Approach Accuracy: {accuracy}')

if __name__ == "__main__":
    main()
