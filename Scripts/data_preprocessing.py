import os
import cv2
import numpy as np
import pandas as pd
import re

def load_images_and_labels_from_folder(folder, csv_file):
    images = []
    labels = []
    annotations = pd.read_csv(csv_file)

    for _, row in annotations.iterrows():
        img_path = os.path.join(folder, row['filename'])
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            labels.append(row['label'])
    
    return images, labels

def preprocess_image(image, target_size):
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def load_and_preprocess_data(train_folder, train_csv, test_folder, test_csv, target_size):
    train_images, train_labels = load_images_and_labels_from_folder(train_folder, train_csv)
    test_images, test_labels = load_images_and_labels_from_folder(test_folder, test_csv)
    
    train_images = [preprocess_image(img, target_size) for img in train_images]
    test_images = [preprocess_image(img, target_size) for img in test_images]
    
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)
