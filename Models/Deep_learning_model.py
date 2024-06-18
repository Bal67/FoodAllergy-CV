# src/deep_learning.py - Neural Networks
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import accuracy_score
from data_preprocessing import load_and_preprocess_data

def main():
    # Paths
    train_folder = '/content/drive/My Drive/FoodAllergyData/train'
    test_folder = '/content/drive/My Drive/FoodAllergyData/test'
    train_annotations = '/content/drive/My Drive/FoodAllergyData/FoodAllergy-CV/Data/annotations_train.csv'  # Adjust as per your file structure
    test_annotations = '/content/drive/My Drive/FoodAllergyData/FoodAllergy-CV/Data/annotations_test.csv'  # Adjust as per your file structure
    model_save_path = '/content/drive/MyDrive/FoodAllergyData/FoodAllergy-CV/Models/deep_learning_model.h5'
    
    # Parameters
    target_size = (224, 224)  
    num_classes = 30  
    
    # Load and preprocess data
    train_images, train_labels, test_images, test_labels = load_and_preprocess_data(train_folder, test_folder, target_size)
    
    # Convert labels to categorical
    train_labels = to_categorical(train_labels, num_classes)
    test_labels = to_categorical(test_labels, num_classes)
    
    # Create a CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train model
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    
    # Save model
    model.save(model_save_path)
    
    # Evaluate model
    predictions = model.predict(test_images)
    predictions = np.argmax(predictions, axis=1)
    test_labels = np.argmax(test_labels, axis=1)
    accuracy = accuracy_score(test_labels, predictions)
    print(f'Deep Learning Model Accuracy: {accuracy}')
    
if __name__ == "__main__":
    main()

