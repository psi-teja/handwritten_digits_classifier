#!/usr/bin/env python3.
import cv2
from using_np.utils import *
from using_np.model import np_model
from using_tf.model import models
from playsound import playsound
from PIL import Image
import sys

try:
    if sys.argv[1] == "np_model":
        using_model = np_model()
    elif sys.argv[1] == "tf_model":
        using_model = models.load_model('using_tf/tf_model')
except:
    using_model = models.load_model('using_tf/tf_model')

drawing = False  # true if mouse is pressed
pt1_x, pt1_y = None, None


def preprocess(img):
    img = img / 255
    img = Image.fromarray(img)
    new_img = img.resize((28, 28))
    X = np.array(new_img.convert('L'))
    X = X.reshape(1, 28, 28, 1)
    return X


# mouse callback function
def line_drawing(event, x, y, flags, param):
    global pt1_x, pt1_y, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (pt1_x, pt1_y), (x, y), color=0, thickness=20)
            pt1_x, pt1_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (pt1_x, pt1_y), (x, y), color=0, thickness=20)
        input = preprocess(img)
        digit = using_model.predict(input)
        filename = f'soundtrack/{str(digit.argmax())}.wav'
        playsound(filename)
        print(f"Predicted Digit: {digit.argmax()}\nPress 'r' to retry!\nPress 'q' to close the window.\n")


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
