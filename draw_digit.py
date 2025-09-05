#!/usr/bin/env python3.
import cv2
import numpy as np
import pygame
import sys
from utils import *


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
            return load_model_from_model_id(1)
        elif argument == "tf":
            print("\nusing TFmodel\n")
            return load_model_from_model_id(2)
        elif argument == "pyt":
            print("\nusing TorchModel\n")
            return load_model_from_model_id(3)
    else:
        print("No command-line argument provided. Using default model.")

    # Fallback to default model if no valid argument is provided
    return load_model_from_model_id(2)


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
        input = preprocess_image_for_inference(img)
        digit = ocr_model.predict(input)
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

    ocr_model = load_model_from_command_line_argument()

    while 1:
        cv2.imshow('draw a digit here', img)
        if cv2.waitKey(1) == ord('r'):
            img = np.ones((280, 280), np.uint8) * 255
        elif cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
