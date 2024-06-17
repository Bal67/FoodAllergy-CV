# src/preprocess.py
import os
import cv2
import numpy as np

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            labels.append(int(filename.split('_')[0]))  # Assuming labels are part of the filename
    return images, labels

def preprocess_image(image, target_size):
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def load_and_preprocess_data(train_folder, test_folder, target_size):
    train_images, train_labels = load_images_from_folder(train_folder)
    test_images, test_labels = load_images_from_folder(test_folder)
    
    train_images = [preprocess_image(img, target_size) for img in train_images]
    test_images = [preprocess_image(img, target_size) for img in test_images]
    
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)
