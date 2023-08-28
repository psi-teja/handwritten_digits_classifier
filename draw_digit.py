#!/usr/bin/env python3.
import cv2
from using_np.utils import *
from using_np.model import np_model
from playsound import playsound

np_model = np_model()

drawing = False  # true if mouse is pressed
pt1_x, pt1_y = None, None

# mouse callback function
def line_drawing(event, x, y):
    global pt1_x, pt1_y, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (pt1_x, pt1_y), (x, y), color=0, thickness=30)
            pt1_x, pt1_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (pt1_x, pt1_y), (x, y), color=0, thickness=30)
        digit = np_model.predict(img)
        filename = f'soundtrack/{str(digit[0])}.wav'
        playsound(filename)
        print("press 'r' to retry!")


img = np.ones((280, 280), np.uint8) * 255

cv2.namedWindow('draw a digit here')
cv2.setMouseCallback('draw a digit here', line_drawing)

while 1:
    cv2.imshow('draw a digit here', img)
    if cv2.waitKey(1) & 0xFF == ord('r'):
        img = np.ones((280, 280), np.uint8) * 255
    elif cv2.waitKey(1) & 0xFF == ord('w'):
        img = np.ones((280, 280), np.uint8) * 255
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
