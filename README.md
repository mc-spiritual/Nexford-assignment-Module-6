# Nexford-assignment-Module-6
# Fashion MNIST CNN Classifier

This notebook implements a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. It's built using TensorFlow and Keras, incorporating best practices like data normalization, metric tracking (precision, recall, F1 score), and visualization of results through accuracy/loss plots and a confusion matrix.

The implementation is encapsulated within a Python class `ApparelClassifier` for better organization and reusability.

## Key Features

- **Data Handling:** Imports and preprocesses the Fashion MNIST dataset (normalization, reshaping, one-hot encoding labels).
- **CNN Architecture:** Defines a sequential CNN model with convolutional layers, pooling, dropout for regularization, and dense layers.
- **Training:** Trains the model with configurable batch size and epochs, tracking key metrics during training.
- **Evaluation:** Evaluates the trained model on the test set, reporting accuracy, precision, recall, F1 score, and a detailed classification report.
- **Visualization:** Generates plots for training/validation accuracy and loss, and displays a confusion matrix to understand per-class performance.
- **Prediction Visualization:** Shows example test images with their predicted and true labels, along with class confidence scores.
- **Model Saving:** Provides functionality to save the trained model.

## Code Breakdown

1.  **Library Imports:** Imports necessary libraries like TensorFlow, Keras, NumPy, Matplotlib, and Seaborn.
2.  **`ApparelClassifier` Class:**
    *   **`__init__`:** Initializes class parameters, including the number of classes, image shape, and placeholders for data and the model. Defines the list of class labels.
    *   **`import_data`:** Loads the Fashion MNIST dataset using `keras.datasets.fashion_mnist.load_data()`.
    *   **`prepare_data`:** Preprocesses the loaded data by normalizing pixel values to the range [0, 1], reshaping images to include the channel dimension, and one-hot encoding the training labels.
    *   **`create_model`:** Defines the CNN model architecture using `keras.Sequential`. The architecture includes:
        *   Two `Conv2D` layers with ReLU activation.
        *   A `MaxPooling2D` layer.
        *   A `Dropout` layer for regularization.
        *   A `Flatten` layer to convert the 2D feature maps to a 1D vector.
        *   Two `Dense` layers with ReLU activation, followed by a final dense layer with softmax activation for classification.
        *   Prints a model summary.
    *   **`setup_model`:** Compiles the model using the Adam optimizer, 'categorical_crossentropy' loss, and tracks 'accuracy', 'Precision', and 'Recall'.
    *   **`run_training`:** Trains the model using `model.fit()`. It splits the training data for validation and records the training history.
    *   **`test_model`:** Evaluates the model on the test dataset using `model.evaluate()` and `model.predict()`. Calculates and prints key metrics and the classification report. Calls helper methods to show the confusion matrix and plot history.
    *   **`plot_history`:** Plots the training and validation accuracy and loss curves over epochs.
    *   **`show_conf_matrix`:** Generates and displays a confusion matrix using Seaborn.
    *   **`visualize_predictions`:** Selects sample test images, makes predictions, and displays the images along with their predicted and true labels, and per-class confidence scores.
    *   **`explain_results`:** Provides a brief explanation of the evaluation metrics used.
    *   **`export_model`:** Saves the trained model to a file using the Keras `.save()` method.
3.  **Script Entry (`if __name__ == "__main__":`)**
    *   Creates an instance of the `ApparelClassifier`.
    *   Calls the methods in sequence: `import_data`, `prepare_data`, `create_model`, `setup_model`, `run_training`.
    *   Performs evaluation and visualization by calling `test_model`.
    *   Calls `explain_results` to print metric explanations.
    *   Calls `visualize_predictions` with specific sample indices.
    *   Calls `export_model` to save the trained model as 'refactored\_fashion\_cnn.keras'.

This notebook provides a complete workflow for building, training, evaluating, and explaining a CNN classifier for image data.
