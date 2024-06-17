import numpy as np
from sklearn.metrics import accuracy_score
from data_preprocessing import load_and_preprocess_data  # Adjust the import path as needed

def main():
    # Paths
    train_folder = '/content/drive/My Drive/FoodAllergyData/train'
    test_folder = '/content/drive/My Drive/FoodAllergyData/test'
    train_annotations = '/content/drive/My Drive/FoodAllergyData/FoodAllergy-CV/Data/annotations_train.csv'  
    test_annotations = '/content/drive/My Drive/FoodAllergyData/FoodAllergy-CV/Data/annotations_test.csv'  
    model_save_path = '/content/drive/MyDrive/FoodAllergyData/FoodAllergy-CV/Models/classical_model_naive.h5'  # Adjust the save path
    
    # Load and preprocess data
    train_images, train_labels, test_images, test_labels, _, _ = load_and_preprocess_data(
        train_folder, test_folder, train_annotations, test_annotations, target_size=(224, 224))  # Assuming target_size is (224, 224)
    
    # Naive approach (example: Random classifier)
    num_classes = len(np.unique(train_labels))  # Assuming train_labels contain class indices
    
    # Generate random predictions (naive approach)
    np.random.seed(42)  # Ensure reproducibility
    predictions = np.random.randint(0, num_classes, size=len(test_labels))
    
    # Evaluate naive approach
    accuracy = accuracy_score(test_labels, predictions)
    print(f'Naive Approach Accuracy: {accuracy}')

if __name__ == "__main__":
    main()
