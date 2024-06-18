import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from data_preprocessing import load_and_preprocess_data
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from skimage.filters import sobel
from scipy.stats import randint
from joblib import Parallel, delayed
import multiprocessing

def extract_features(image):
    # Convert the image to 8-bit unsigned integer format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    
    # Convert the image to grayscale
    gray = rgb2gray(image)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    
    # Edge detection using Sobel filter
    edges = sobel(gray)
    edge_hist, _ = np.histogram(edges.ravel(), bins=np.arange(0, 256), range=(0, 255))
    edge_hist = edge_hist.astype("float")
    edge_hist /= (edge_hist.sum() + 1e-6)

    # Combine features
    features = np.concatenate((hist.flatten(), lbp_hist, edge_hist))
    
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
    
    # Extract features from images using multiprocessing
    num_cores = multiprocessing.cpu_count()
    train_features = np.array(Parallel(n_jobs=num_cores)(delayed(extract_features)(img) for img in train_images))
    test_features = np.array(Parallel(n_jobs=num_cores)(delayed(extract_features)(img) for img in test_images))
    
    # Define the parameter distribution for RandomizedSearchCV
    param_dist = {
        'n_estimators': randint(50, 100),
        'max_depth': [3, 5, 7],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5)
    }
    
    # Train a Random Forest classifier with RandomizedSearchCV
    classifier = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(estimator=classifier, param_distributions=param_dist, n_iter=20, cv=2, n_jobs=-1, verbose=2)
    random_search.fit(train_features, train_labels)
    
    # Use the best estimator from RandomizedSearchCV
    best_classifier = random_search.best_estimator_
    
    # Make predictions
    predictions = best_classifier.predict(test_features)
    
    # Evaluate the model
    accuracy = accuracy_score(test_labels, predictions)
    print(f'Naive Approach Accuracy: {accuracy}')

if __name__ == "__main__":
    main()