import os
import cv2
import pandas as pd
import numpy as np
import random

def load_images_from_folder_with_annotations(folder, annotations_file, num_pairs=200):
    # Load annotations CSV
    annotations = pd.read_csv(annotations_file)
    
    # Initialize lists to store images and labels
    images = []
    labels = []
    allergen_labels = []
    
    # Define allergen mapping based on your provided map
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
    
    # Randomly select num_pairs samples
    random_indices = random.sample(range(len(annotations)), num_pairs)
    selected_annotations = annotations.iloc[random_indices]
    
    # Iterate through each selected row in the annotations DataFrame
    for _, row in selected_annotations.iterrows():
        img_path = os.path.join(folder, row['filename'])
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            labels.append(row['label'])  # Original label from CSV
            allergen_labels.append(allergen_map.get(row['label'], 'Unknown'))  # Mapped allergen label
    
    # Create DataFrame with selected annotations
    selected_annotations['allergen_label'] = allergen_labels
    
    return images, labels, selected_annotations

def preprocess_image(image, target_size):
    # Resize image to target size and normalize
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def load_and_preprocess_data(train_folder, test_folder,
                             train_annotations, test_annotations,
                             target_size):
    # Load and preprocess training data
    train_images, train_labels, train_annotations_df = load_images_from_folder_with_annotations(train_folder, train_annotations)
    train_images = [preprocess_image(img, target_size) for img in train_images]
    
    # Load and preprocess testing data
    test_images, test_labels, test_annotations_df = load_images_from_folder_with_annotations(test_folder, test_annotations)
    test_images = [preprocess_image(img, target_size) for img in test_images]
    
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels), train_annotations_df, test_annotations_df
