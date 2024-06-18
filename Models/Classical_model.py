import numpy as np
import cv2
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from data_preprocessing import load_and_preprocess_data
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray

def extract_features(image):
    # Convert the image to 8-bit unsigned integer format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Compute the color histogram
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], 
                        [0, 180, 0, 256, 0, 256])
    # Normalize the histogram
    cv2.normalize(hist, hist)
    
    # Convert the image to grayscale
    gray = rgb2gray(image)
    # Compute LBP features
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # Combine color histogram and LBP features
    features = np.concatenate((hist.flatten(), lbp_hist))
    
    return features

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
    
# Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Train a Random Forest classifier with GridSearchCV
    classifier = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(train_features, train_labels)
    
    # Use the best estimator from GridSearchCV
    best_classifier = grid_search.best_estimator_
    
    # Make predictions
    predictions = best_classifier.predict(test_features)
    
    # Evaluate the model
    accuracy = accuracy_score(test_labels, predictions)
    print(f'Naive Approach Accuracy: {accuracy}')

if __name__ == "__main__":
    main()