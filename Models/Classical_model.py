import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_preprocessing import load_and_preprocess_data

def extract_features_resnet(image):
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)
    features = resnet_model.predict(np.expand_dims(image, axis=0))
    return features.flatten()


def main():
    # Paths
    train_folder = '/content/drive/My Drive/FoodAllergyData/train'
    test_folder = '/content/drive/My Drive/FoodAllergyData/test'
    train_annotations = '/content/drive/My Drive/FoodAllergyData/FoodAllergy-CV/Data/annotations_train.csv' 
    test_annotations = '/content/drive/My Drive/FoodAllergyData/FoodAllergy-CV/Data/annotations_test.csv' 
    
# Parameters
    target_size = (224, 224)
    num_classes = 30  # Number of categories
    
    # Load and preprocess data
    train_images, train_labels, test_images, test_labels, train_annotations_df, test_annotations_df = load_and_preprocess_data(
        train_folder, test_folder, train_annotations, test_annotations, target_size
    )
    
    # Convert labels to integers
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)
    
    # Normalize image data
    train_images = np.array(train_images) / 255.0
    test_images = np.array(test_images) / 255.0
    
    # Initialize ResNet model for feature extraction
    global resnet_model
    resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Extract features using ResNet
    train_features = np.array([extract_features_resnet(img) for img in train_images])
    test_features = np.array([extract_features_resnet(img) for img in test_images])
    
    # Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    for train_index, val_index in kf.split(train_features):
        X_train, X_val = train_features[train_index], train_features[val_index]
        y_train, y_val = train_labels_encoded[train_index], train_labels_encoded[val_index]

        # Initialize and train the model
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        # Validate the model
        y_val_pred = model.predict(X_val)
        accuracies.append(accuracy_score(y_val, y_val_pred))

    print(f'Mean Cross-Validation Accuracy: {np.mean(accuracies)}')
    
    # Train final model on the entire training set
    final_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    final_model.fit(train_features, train_labels_encoded)
    
    # Evaluate the model on the test set
    test_pred = final_model.predict(test_features)
    test_accuracy = accuracy_score(test_labels_encoded, test_pred)
    print(f'Test Accuracy: {test_accuracy}')

if __name__ == "__main__":
    main()