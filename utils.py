#!/usr/bin/env python3.
import torch
import numpy as np
from np_model.model import NumpyModel
from tf_model.model import models
from pyt_model.model import TorchModel
from PIL import Image

# Default model path
TFmodel_path = 'tf_model/pretrained_model'
NumpyModel_path = 'np_model/weights/model_weights.pkl'
TorchModel_path = 'pyt_model/pretrained_model/mnist_model.pth'


def preprocess_image_for_model(input_image):
    """
    Preprocesses an input image for a deep learning model.

    This function takes an input image and performs the following preprocessing steps:
    1. Normalizes pixel values to the range [0, 1].
    2. Converts the image to grayscale.
    3. Resizes the image to the specified dimensions (28x28).
    4. Reshapes the image to match the input shape expected by the model (1, 28, 28, 1).

    Args:
        input_image (numpy.ndarray): The input image as a NumPy array.

    Returns:
        numpy.ndarray: The preprocessed image as a NumPy array suitable for model inference.

    Example:
        input_img = np.array(Image.open('input.jpg'))
        preprocessed_img = preprocess_image_for_model(input_img)
    """
    # Normalize pixel values to the range [0, 1]
    input_image = input_image / 255.0

    # Convert the image to grayscale
    gray_image = Image.fromarray(input_image).convert('L')

    # Resize the image to the specified dimensions (28x28)
    resized_image = gray_image.resize((28, 28))

    # Reshape the image to match the input shape expected by the model (1, 28, 28, 1)
    reshaped_image = np.array(resized_image).reshape(1, 28, 28, 1)

    return reshaped_image


def load_model_from_model_id(n=2):
    """
    Load a deep learning model based on a command-line argument.

    This function attempts to load a model based on the model ID argument provided.
    1 -> NumpyModel
    2 -> TFmodel
    3 -> pyTorch 

    If no argument is given or the argument is not recognized, it falls back to a default model.

    Returns:
        tf.keras.Model: The loaded deep learning model.

    Example:
        loaded_model = load_model_from_command_line_argument(1)
    """
    if n == 1:
        try:
            print("\nusing NumpyModel\n")
            return NumpyModel(NumpyModel_path)
        except:
            print(f"Error: Model file '{NumpyModel_path}' not found.")
    elif n == 2:
        try:
            print("\nusing TFmodel\n")
            return models.load_model(TFmodel_path)
        except FileNotFoundError:
            print(f"Error: Model file '{TFmodel_path}' not found.")
    elif n == 3:
        try:
            model = TorchModel()
            model.load_state_dict(torch.load(TorchModel_path))
            model.eval()
            print("\nusing TorchModel\n")
            return model
        except FileNotFoundError:
            print(f"Error: Model file '{TorchModel_path}' not found.")
    else:
        print("No command-line argument provided. Using default model.")
