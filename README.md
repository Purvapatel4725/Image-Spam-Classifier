# Spam Classification Using CNN

This repository contains a complete setup for classifying images as **Spam** or **Not Spam** using a custom Convolutional Neural Network (CNN) model. It includes:

1. A training notebook to build and test the CNN model.
2. An automated script to classify images using the trained model.
3. A structured dataset for training and testing the model.

## Repository Contents

- **dataset/**: Folder containing two subdirectories:
  - `spam`: Place all spam images here.
  - `not_spam`: Place all not spam images here.
- **Auto-Classify.py**: Python script for automated classification of images using the trained model.
- **Modle-Train.ipynb**: Jupyter notebook for training the CNN model and testing it on individual images.
- **requirements.txt**: File listing all necessary dependencies.
- **README.md**: Documentation for the repository.

## Setup Instructions

### Step 1: Install Dependencies

To get started, install the required Python libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 2: Prepare the Dataset

1. Create a folder named `dataset` in the root directory.
2. Inside `dataset`, create two subdirectories:
   - `spam`: Add all images labeled as spam.
   - `not_spam`: Add all images labeled as not spam.

The folder structure should look like this:

```
dataset/
├── spam/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── not_spam/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
```

### Step 3: Train the CNN Model

1. Open the `Modle-Train.ipynb` notebook.
2. Update the `dataset_path` variable in **Cell 1** with the path to your `dataset` folder.
3. Run the notebook to:
   - Preprocess the data.
   - Build, train, and validate the CNN model.
   - Save the trained model as `custom_cnn_model.keras`.
4. The training process will generate accuracy and loss graphs to help evaluate the model’s performance.

### Step 4: Test the Model

Use the second cell of the notebook to test individual images:

1. Update the `image_path` variable with the path to your test image.
2. Run the cell to classify the image as **Spam** or **Not Spam**.

### Step 5: Classify Images Automatically

Use the `Auto-Classify.py` script to classify all images in a directory:

1. Place the images to be classified in a directory.
2. Run the script:

   ```bash
   python Auto-Classify.py
   ```

3. Provide the path to the directory when prompted.
4. The script will classify images and move them into two subdirectories within the input directory:
   - `Spam/`
   - `NotSpam/`

## How It Works

### Model Training

The `Modle-Train.ipynb` notebook builds a CNN with the following architecture:

- Three convolutional layers with ReLU activation and MaxPooling layers.
- Fully connected layers with dropout to prevent overfitting.
- Sigmoid activation in the output layer for binary classification.

The model is compiled using:
- **Optimizer**: Adam (Adaptive Moment Estimation)
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy

Why These Choices?
- Adam: Effective and requires less hyperparameter tuning.
- Binary Crossentropy: Suitable for binary classification tasks like this one.
- Accuracy: Intuitive and directly interprets the model's performance in terms of correct predictions.

### Automation Script

The `Auto-Classify.py` script:

1. Loads the pre-trained model.
2. Processes each image in the input directory by resizing it to 150x150 pixels and normalizing pixel values.
3. Predicts the classification score and moves the image to the appropriate folder (`Spam` or `NotSpam`).

## Author

This repository was created by **Purva Patel**. Please use the provided tools and scripts responsibly and ethically.

---

For any queries or issues, feel free to contact me or open an issue on this repository.

