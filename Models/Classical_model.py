import numpy as np
import cv2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from data_preprocessing import load_and_preprocess_data
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from scipy.stats import randint
from skimage.filters import sobel

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
    
    # Load and preprocess the data
    X_train, y_train = load_and_preprocess_data(train_folder, train_annotations, extract_features)
    X_test, y_test = load_and_preprocess_data(test_folder, test_annotations, extract_features)
    
    target_size = (224, 224)

    # Train the model
    model = GradientBoostingClassifier()
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(1, 10),
        'learning_rate': [0.1, 0.01, 0.001]
    }
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5)
    random_search.fit(X_train, y_train)

    # Evaluate the model
    y_pred = random_search.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Naive Accuracy: {accuracy}')

if __name__ == '__main__':
    main()
