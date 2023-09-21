import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)



def preprocess_data_for_training(X_train, y_train):
    """preprocess_data_for_training 
    convert white pixels to black pixels and vise-versa
    and creates one hot encoded labels

    Args:
        X_train (array): images data
        y_train (array): labels data

    Returns:
        X_train: modified images data
        Y_train: one hot encoded labels
    """
    X_train = (255 - X_train) / 255
    X_train[X_train < 1] = 0

    Y_train = np.zeros((len(y_train), 10))
    for i in range(len(Y_train)):
        Y_train[i][y_train[i]] = 1

    X_train = X_train.reshape((X_train.shape[0], -1))

    return X_train, Y_train