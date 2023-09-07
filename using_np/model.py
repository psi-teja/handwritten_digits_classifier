import numpy as np
import pickle as pkl

class NumpyModel:
    """
    A simple neural network model implemented using NumPy.

    Attributes:
        w1 (numpy.ndarray): Weights for the first hidden layer.
        w2 (numpy.ndarray): Weights for the second hidden layer.
        b1 (numpy.ndarray): Biases for the first hidden layer.
        b2 (numpy.ndarray): Biases for the second hidden layer.
        wo (numpy.ndarray): Weights for the output layer.
        bo (numpy.ndarray): Biases for the output layer.
    """

    def __init__(self, model_weights_path):
        """
        Initialize the model by loading weights and biases from pickle files.
        """
        try:
            with open(model_weights_path, 'rb') as file:
                weights = pkl.load(file)
                self.w1, self.w2, self.wo, self.b1, self.b2, self.bo = weights
        except FileNotFoundError:
            raise FileNotFoundError(f"Model weights file '{model_weights_path}' not found.")
    def predict(self, X):
        """
        Predict the output of the neural network for the given input data.

        Args:
            X (numpy.ndarray): Input data with shape (1, 784).

        Returns:
            numpy.ndarray: Predicted probabilities for each class.
        """
        X = X.reshape((1, 784))
        h1 = np.dot(X, self.w1) + self.b1
        h1 = 1 / (1 + np.exp(-h1))
        h2 = np.dot(h1, self.w2) + self.b2
        h2 = 1 / (1 + np.exp(-h2))
        output = np.dot(h2, self.wo) + self.bo
        softmax_output = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)

        return softmax_output
