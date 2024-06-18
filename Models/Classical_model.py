import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from data_preprocessing import load_and_preprocess_data

# Paths to the dataset
train_images_path = '/content/drive/My Drive/FoodAllergyData/train'
test_images_path = '/content/drive/My Drive/FoodAllergyData/test'
train_labels_path = '/content/drive/My Drive/FoodAllergyData/FoodAllergy-CV/Data/annotations_train.csv' 
test_labels_path = '/content/drive/My Drive/FoodAllergyData/FoodAllergy-CV/Data/annotations_test.csv'  
model_save_path = '/content/drive/MyDrive/FoodAllergyData/FoodAllergy-CV/Models/deep_learning_model.h5'

target_size = (64, 64)

# Load and preprocess data
train_images, train_labels, test_images, test_labels, _, _ = load_and_preprocess_data(
    train_images_path, test_images_path, train_labels_path, test_labels_path, target_size)

# Check shapes of loaded images
print(f'Shape of train images: {train_images.shape}')
print(f'Shape of test images: {test_images.shape}')

# Encode the labels
label_encoder = LabelEncoder()
train_labels_enc = label_encoder.fit_transform(train_labels)
test_labels_enc = label_encoder.transform(test_labels)

# Convert labels to categorical
train_labels_categorical = to_categorical(train_labels_enc)
test_labels_categorical = to_categorical(test_labels_enc)

# Ensure the image data is in the correct shape for VGG16
if len(train_images.shape) != 4 or train_images.shape[3] != 3:
    train_images_cnn = np.repeat(train_images.reshape(train_images.shape[0], 64, 64, 1), 3, axis=3)
else:
    train_images_cnn = train_images

if len(test_images.shape) != 4 or test_images.shape[3] != 3:
    test_images_cnn = np.repeat(test_images.reshape(test_images.shape[0], 64, 64, 1), 3, axis=3)
else:
    test_images_cnn = test_images

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load the VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(np.unique(train_labels)), activation='softmax')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with data augmentation
model.fit(datagen.flow(train_images_cnn, train_labels_categorical, batch_size=32),
          validation_data=(test_images_cnn, test_labels_categorical),
          epochs=50,
          callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images_cnn, test_labels_categorical)
test_pred = model.predict(test_images_cnn)
test_f1 = f1_score(test_labels_enc, np.argmax(test_pred, axis=1), average='macro')


print(f'Test Accuracy: {test_accuracy:.2f}')
print(f'Test F1-score: {test_f1:.2f}')
