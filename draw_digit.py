#!/usr/bin/env python3.

import cv2
import numpy as np
from PIL import Image
import pickle as pkl
from playsound import playsound

# for playing wav file


# global wh1_b, wh2_b, bh1_b, bh2_b, wout_b, bout_b


wh1_b = pkl.load(open('wh1_b.pkl', 'rb'))
wh2_b = pkl.load(open('wh2_b.pkl', 'rb'))
bh1_b = pkl.load(open('bh1_b.pkl', 'rb'))
bh2_b = pkl.load(open('bh2_b.pkl', 'rb'))
wout_b = pkl.load(open('wout_b.pkl', 'rb'))
bout_b = pkl.load(open('bout_b.pkl', 'rb'))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    S = np.exp(x)
    T = np.sum(S, axis=1, keepdims=True)
    y = S / T
    return y


def train(img, expected, wh1_b, wh2_b, wout_b, bh1_b, bh2_b, bout_b):
    img = (255 - img) / 255
    img = Image.fromarray(img)
    new_img = img.resize((28, 28))
    # img = np.resize(img, (270,270))

    learning_rate = 0.1
    # new_img.save('3.jpg')

    X = np.array(new_img.convert('L'))

    X = X.reshape((1, 784))

    y = int(expected)

    Y = np.zeros((1, 10))

    Y[0][y] = 1

    hl1_in = np.dot(X, wh1_b) + bh1_b
    hl1_a = sigmoid(hl1_in)

    hl2_in = np.dot(hl1_a, wh2_b) + bh2_b
    hl2_a = sigmoid(hl2_in)

    out_in = np.dot(hl2_a, wout_b) + bout_b
    out_in = out_in
    out_a = softmax(out_in)

    E = Y - out_a

    slope_out = d_sigmoid(out_in)
    slope_hl1 = d_sigmoid(hl1_in)
    slope_hl2 = d_sigmoid(hl2_in)

    # d_out = E * slope_out
    d_out = np.multiply(E, slope_out)

    Error_at_hl2 = np.dot(d_out, wout_b.transpose())

    d_hl2 = Error_at_hl2 * slope_hl2

    Error_at_hl1 = np.dot(d_hl2, wh2_b.transpose())

    d_hl1 = Error_at_hl1 * slope_hl1

    wout_b += np.dot(hl2_a.transpose(), d_out) * learning_rate
    wh2_b += np.dot(hl1_a.transpose(), d_hl2) * learning_rate
    wh1_b += np.dot(X.transpose(), d_hl1) * learning_rate

    bout_b += sum(d_out) * learning_rate
    bh2_b += sum(d_hl2) * learning_rate
    bh1_b += sum(d_hl1) * learning_rate


def check(img, wh1_b, wh2_b, wout_b, bh1_b, bh2_b, bout_b):
    img = (255 - img) / 255
    img = Image.fromarray(img)
    new_img = img.resize((28, 28))
    # img = np.resize(img, (270,270))

    # new_img.save('3.jpg')

    X = np.array(new_img.convert('L'))

    X = X.reshape((1, 784))

    hl1 = np.dot(X, wh1_b) + bh1_b
    hl1 = 1 / (1 + np.exp(-hl1))
    hl2 = np.dot(hl1, wh2_b) + bh2_b
    hl2 = 1 / (1 + np.exp(-hl2))

    out = np.dot(hl2, wout_b) + bout_b
    S = np.exp(out)
    T = np.sum(S, axis=1, keepdims=True)
    out = S / T

    max_index = np.argmax(out, axis=1)

    print(max_index)
    filename = f'soundtrack/{str(max_index[0])}.wav'
    playsound(filename)


drawing = False  # true if mouse is pressed
pt1_x, pt1_y = None, None


# mouse callback function
def line_drawing(event, x, y, flags, param):
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
        check(img, wh1_b, wh2_b, wout_b, bh1_b, bh2_b, bout_b)
        print("press 'r' to retry!")


img = np.ones((280, 280), np.uint8) * 255

cv2.namedWindow('draw a digit here')
cv2.setMouseCallback('draw a digit here', line_drawing)

while (1):
    cv2.imshow('draw a digit here', img)
    if cv2.waitKey(1) & 0xFF == ord('r'):
        img = np.ones((280, 280), np.uint8) * 255
    elif cv2.waitKey(1) & 0xFF == ord('w'):
        expected = input('expected output:')
        train(img, expected, wh1_b, wh2_b, wout_b, bh1_b, bh2_b, bout_b)
        img = np.ones((280, 280), np.uint8) * 255
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
