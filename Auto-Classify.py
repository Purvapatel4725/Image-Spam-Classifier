import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def classify_images(input_dir, model_path):
    """
    Classify images in a directory as spam or not spam using a pre-trained model.
    Args:
        input_dir (str): Path to the input directory containing images.
        model_path (str): Path to the saved Keras model.
    """
    # Load the pre-trained model
    model = load_model(model_path)
    print("Model loaded successfully.")

    # Define paths for output directories
    spam_dir = os.path.join(input_dir, 'Spam')
    not_spam_dir = os.path.join(input_dir, 'NotSpam')

    # Create output directories if they don't exist
    os.makedirs(spam_dir, exist_ok=True)
    os.makedirs(not_spam_dir, exist_ok=True)

    # Process each file in the input directory
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)

        # Skip if it's not a file or not an image
        if not os.path.isfile(file_path) or not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        try:
            # Load and preprocess the image
            img = load_img(file_path, target_size=(150, 150))
            img_array = img_to_array(img) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict the classification
            prediction = model.predict(img_array)[0][0]

            # Move the image to the appropriate folder
            if prediction > 0.5:
                shutil.move(file_path, os.path.join(spam_dir, file_name))
                print(f"'{file_name}' classified as SPAM and moved to '{spam_dir}'.")
            else:
                shutil.move(file_path, os.path.join(not_spam_dir, file_name))
                print(f"'{file_name}' classified as NOT SPAM and moved to '{not_spam_dir}'.")

        except Exception as e:
            print(f"Error processing file '{file_name}': {e}")

if __name__ == "__main__":
    # Ask for the input directory
    input_directory = input("Enter the path of the input directory: ").strip()

    # Validate the input directory
    if not os.path.exists(input_directory):
        print(f"Error: The directory '{input_directory}' does not exist.")
    else:
        # Path to your trained model
        model_file = 'custom_cnn_model.keras'  # Update if your model is in a different path
        classify_images(input_directory, model_file)
