from tensorflow.keras import datasets
import scipy.ndimage as ndimage
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

X_train = (255 - X_train) / 255
X_test = (255 - X_test) / 255

X_train[X_train < 1] = 0
X_test[X_test < 1] = 0

X_train = X_train.reshape(-1, 28, 28, 1)  # training set
X_test = X_test.reshape(-1, 28, 28, 1)  # test set


