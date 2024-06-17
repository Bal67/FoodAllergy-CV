import os
import cv2
import pandas as pd
import numpy as np
import re

def load_images_from_folder_with_annotations(folder, annotations_file):
    annotations = pd.read_csv(annotations_file)
    images = []
    labels = []
    allergen_labels = []
    
    allergen_map = {
        'egg': 'Ovomucoid',
        'whole_egg_boiled': 'Ovomucoid',
        'milk': 'Lactose/Histamine',
        'icecream': 'Lactose',
        'cheese': 'Lactose',
        'milk_based_beverage': 'Lactose/ Caffeine',
        'chocolate': 'Lactose/Caffeine',
        'non_milk_based_beverage': 'Caffeine',
        'cooked_meat': 'Histamine',
        'raw_meat': 'Histamine',
        'alcohol': 'Histamine',
        'alcohol_glass': 'Histamine',
        'spinach': 'Histamine',
        'avocado': 'Histamine',
        'eggplant': 'Histamine',
        'blueberry': 'Salicylate',
        'blackberry': 'Salicylate',
        'strawberry': 'Salicylate',
        'pineapple': 'Salicylate',
        'capsicum': 'Salicylate',
        'mushroom': 'Salicylate',
        'dates': 'Salicylate',
        'almonds': 'Salicylate',
        'pistachios': 'Salicylate',
        'tomato': 'Salicylate',
        'roti': 'Gluten',
        'pasta': 'Gluten',
        'bread': 'Gluten',
        'bread_loaf': 'Gluten',
        'pizza': 'Gluten'
    }
    
    for _, row in annotations.iterrows():
        img_path = os.path.join(folder, row['filename'])
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            labels.append(row['label'])  # Original label from CSV
            allergen_labels.append(allergen_map.get(row['label'], 'Unknown'))  # Mapped allergen label
    
    annotations['allergen_label'] = allergen_labels  # Add allergen_label column to DataFrame
    
    return images, labels, annotations

def preprocess_image(image, target_size):
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def load_and_preprocess_data(train_folder, test_folder,
                             train_annotations, test_annotations,
                             target_size):
    train_images, train_labels, train_annotations_df = load_images_from_folder_with_annotations(train_folder, train_annotations)
    test_images, test_labels, test_annotations_df = load_images_from_folder_with_annotations(test_folder, test_annotations)
    
    # Preprocess images
    train_images = [preprocess_image(img, target_size) for img in train_images]
    test_images = [preprocess_image(img, target_size) for img in test_images]
    
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels), train_annotations_df, test_annotations_df