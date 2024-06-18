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
    # Convert the image to grayscale
    gray_image = rgb2gray(image)
    # Extract the LBP features
    lbp = local_binary_pattern(gray_image, 8, 1, method='uniform')
    # Extract the Sobel features
    sobel_x = sobel(gray_image, axis=0)
    sobel_y = sobel(gray_image, axis=1)
    # Concatenate the features
    features = np.hstack([lbp.ravel(), sobel_x.ravel(), sobel_y.ravel()])
    return features

def train_model():
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    # Extract features from the training images
    X_train_features = np.array([extract_features(image) for image in X_train])
    # Train a Gradient Boosting classifier
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train_features, y_train)
    # Extract features from the test images
    X_test_features = np.array([extract_features(image) for image in X_test])
    # Evaluate the model
    y_pred = model.predict(X_test_features)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    return model

def hyperparameter_tuning():
    # Load and preprocess the data
    X_train, _, y_train, _ = load_and_preprocess_data()
    # Extract features from the training images
    X_train_features = np.array([extract_features(image) for image in X_train])
    # Define the hyperparameter grid
    param_dist = {
        'n_estimators': randint(50, 200),
        'learning_rate': [0.01, 0.1, 0.5],
        'max_depth': [3, 5, 7]
    }
    # Train a Gradient Boosting classifier with hyperparameter tuning
    model = GradientBoostingClassifier()
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=3)
    random_search.fit(X_train_features, y_train)
    print(f'Best hyperparameters: {random_search.best_params_}')

if __name__ == '__main__':
    train_model()
    # hyperparameter_tuning()