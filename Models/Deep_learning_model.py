import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_preprocessing import load_and_preprocess_data

def main():
    # Paths
    train_folder = '/content/drive/My Drive/FoodAllergyData/train'
    test_folder = '/content/drive/My Drive/FoodAllergyData/test'
    train_annotations = '/content/drive/My Drive/FoodAllergyData/FoodAllergy-CV/Data/annotations_train.csv' 
    test_annotations = '/content/drive/My Drive/FoodAllergyData/FoodAllergy-CV/Data/annotations_test.csv'  
    model_save_path = '/content/drive/MyDrive/FoodAllergyData/FoodAllergy-CV/Models/deep_learning_model.h5'
    
    # Parameters
    target_size = (224, 224)  
    num_classes = 30  
    
    # Load and preprocess data
    train_images, train_labels, test_images, test_labels, _, _ = load_and_preprocess_data(
        train_folder, test_folder, train_annotations, test_annotations, target_size)
    
    # Encode labels to integers
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)

    # Convert labels to categorical
    train_labels_categorical = to_categorical(train_labels_encoded, num_classes)
    test_labels_categorical = to_categorical(test_labels_encoded, num_classes)
    
    # Normalize image data
    train_images = np.array(train_images) / 255.0
    test_images = np.array(test_images) / 255.0
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels_categorical, test_size=0.2, random_state=42)
        
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,  # increased
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,  # added
        shear_range=0.2)  # added

    # Build the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        BatchNormalization(),  # added
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),  # added
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),  # added
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),  # added
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),  # added
        Dropout(0.5),  # added
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                optimizer=Adam(lr=0.001),
                metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    # Fit the model
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                        epochs=100,  # increased epochs
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping, reduce_lr])
        
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_images, test_labels_categorical)
    print(f'Deep Learning Test Accuracy: {test_accuracy}')

if __name__ == "__main__":
    main()
