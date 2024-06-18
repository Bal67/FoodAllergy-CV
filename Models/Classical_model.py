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
    # Paths
    train_folder = '/content/drive/My Drive/FoodAllergyData/train'
    test_folder = '/content/drive/My Drive/FoodAllergyData/test'
    train_annotations = '/content/drive/My Drive/FoodAllergyData/FoodAllergy-CV/Data/annotations_train.csv' 
    test_annotations = '/content/drive/My Drive/FoodAllergyData/FoodAllergy-CV/Data/annotations_test.csv'  
    model_save_path = '/content/drive/MyDrive/FoodAllergyData/FoodAllergy-CV/Models/deep_learning_model.h5'
    
    target_size = (224, 224)
    
    train_images, train_labels, test_images, test_labels, _, _ = load_and_preprocess_data(
        train_folder, test_folder, train_annotations, test_annotations, target_size)

    # Extract features
    X_train = np.array([extract_features(image) for image in X_train])
    X_test = np.array([extract_features(image) for image in X_test])

    # Train the model
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

    return model

if __name__ == '__main__':
    train_model()
