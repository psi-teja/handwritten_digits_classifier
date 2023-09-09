#!/usr/bin/env python3.
import cv2
from using_np.utils import *
from using_np.model import NumpyModel
from using_tf.model import models
import pygame
from PIL import Image
import sys

# Default model path
default_model_path = 'using_tf/tf_model'
NumpyModel_path = 'using_np/weights/model_weights.pkl'

def load_model_from_command_line_argument():
    """
    Load a deep learning model based on a command-line argument.

    This function attempts to load a model based on the command-line argument provided.
    If no argument is given or the argument is not recognized, it falls back to a default model.

    Returns:
        tf.keras.Model: The loaded deep learning model.

    Example:
        loaded_model = load_model_from_command_line_argument()
    """
    if len(sys.argv) > 1:
        argument = sys.argv[1].lower()
        if argument == "np":
            print("\nusing NumpyModel\n")
            return NumpyModel(NumpyModel_path)
        elif argument == "tf":
            try:
                print("\nusing TFmodel\n")
                return models.load_model(default_model_path)
            except FileNotFoundError:
                print(f"Error: Model file '{default_model_path}' not found.")
    else:
        print("No command-line argument provided. Using default model.")
    
    # Fallback to default model if no valid argument is provided
    return models.load_model(default_model_path)


using_model = load_model_from_command_line_argument()


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


drawing = False  # true if mouse is pressed
pt1_x, pt1_y = None, None
pygame.init()

# mouse callback function
def line_drawing(event, x, y, flags, param):
    """
    Mouse event handler for drawing on an image canvas and making digit predictions.

    Args:
        event (int): The type of mouse event (e.g., LBUTTONDOWN, MOUSEMOVE, LBUTTONUP).
        x (int): The x-coordinate of the mouse cursor.
        y (int): The y-coordinate of the mouse cursor.
        flags (int): Additional flags passed by OpenCV.
        param: Additional parameters (not used).

    Returns:
        None

    """
    global pt1_x, pt1_y, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (pt1_x, pt1_y), (x, y), color=0, thickness=25)
            pt1_x, pt1_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (pt1_x, pt1_y), (x, y), color=0, thickness=25)
        input = preprocess_image_for_model(img)
        digit = using_model.predict(input)
        filename = f'soundtrack/{str(digit.argmax())}.wav'
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        print(f"Predicted Digit: {digit.argmax()}")
        print("Press 'r' to retry!")
        print("Press 'q' to close the window.")


if __name__ == "__main__":
    img = np.ones((280, 280), np.uint8) * 255

    cv2.namedWindow('draw a digit here')
    cv2.setMouseCallback('draw a digit here', line_drawing)

    while 1:
        cv2.imshow('draw a digit here', img)
        if cv2.waitKey(1) == ord('r'):
            img = np.ones((280, 280), np.uint8) * 255
        elif cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
