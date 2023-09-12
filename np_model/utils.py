import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def preprocess_data(X_train, y_train):
    X_train = (255 - X_train) / 255
    X_train[X_train > 0] = 1

    Y_train = np.zeros((len(y_train), 10))
    for i in range(len(Y_train)):
        Y_train[i][y_train[i]] = 1

    X_train = X_train.reshape((X_train.shape[0], -1))

    return X_train, Y_train