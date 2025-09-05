import numpy as np
import pickle as pkl
from tqdm import trange
from keras.datasets import mnist
from utils import preprocess_data_for_training, relu, softmax, d_relu

# loading MNIST dataset
(X, y), (_, _) = mnist.load_data()
# preprocessing training data
X, Y = preprocess_data_for_training(X, y)
X_train = X[:50000]
Y_train = Y[:50000]

X_val = X[50000:]
Y_val = Y[50000:]

N_train = len(X_train)
epochs = 1000
batch_size = 1000
learning_rate = 0.001
lamda_l2 = 0.001

# designing MLPs
n_in = 784
n_hl1 = 128
n_hl2 = 128
n_out = 10

# Random initialization of weights and biases
w1 = np.random.randn(n_in, n_hl1) * np.sqrt(2.0 / n_in)
b1 = np.zeros((1, n_hl1))
w2 = np.random.randn(n_hl1, n_hl2) * np.sqrt(2.0 / n_hl1)
b2 = np.zeros((1, n_hl2))
wo = np.random.randn(n_hl2, n_out) * np.sqrt(2.0 / n_hl2)
bo = np.zeros((1, n_out))

# For batch normalization (simplified approach)
gamma1, beta1 = np.ones((1, n_hl1)), np.zeros((1, n_hl1))
gamma2, beta2 = np.ones((1, n_hl2)), np.zeros((1, n_hl2))
gamma_out, beta_out = np.ones((1, n_out)), np.zeros((1, n_out))

# Running mean and variance for inference
running_mean1, running_var1 = np.zeros((1, n_hl1)), np.ones((1, n_hl1))
running_mean2, running_var2 = np.zeros((1, n_hl2)), np.ones((1, n_hl2))
running_mean_out, running_var_out = np.zeros((1, n_out)), np.ones((1, n_out))

# Define variables for early stopping
best_validation_loss = float("inf")
patience = 10
wait = 0

for j in trange(epochs, desc="Training In Progress", unit="epochs"):
    i = 0
    error = 0
    indices = np.random.permutation(N_train)
    X_train_shuffled = X_train[indices]
    Y_train_shuffled = Y_train[indices]
    
    while i < N_train:
        X_batch = X_train_shuffled[i:i+batch_size]
        Y_batch = Y_train_shuffled[i:i+batch_size]
        
        # Forward pass with batch normalization (training mode)
        # Layer 1
        hl1_in = np.dot(X_batch, w1) + b1
        mean1 = np.mean(hl1_in, axis=0, keepdims=True)
        var1 = np.var(hl1_in, axis=0, keepdims=True)
        hl1_in_norm = (hl1_in - mean1) / np.sqrt(var1 + 1e-8)
        hl1_in_scaled = gamma1 * hl1_in_norm + beta1
        hl1_a = relu(hl1_in_scaled)
        
        # Update running statistics (exponential moving average)
        running_mean1 = 0.9 * running_mean1 + 0.1 * mean1
        running_var1 = 0.9 * running_var1 + 0.1 * var1
        
        # Layer 2
        hl2_in = np.dot(hl1_a, w2) + b2
        mean2 = np.mean(hl2_in, axis=0, keepdims=True)
        var2 = np.var(hl2_in, axis=0, keepdims=True)
        hl2_in_norm = (hl2_in - mean2) / np.sqrt(var2 + 1e-8)
        hl2_in_scaled = gamma2 * hl2_in_norm + beta2
        hl2_a = relu(hl2_in_scaled)
        
        running_mean2 = 0.9 * running_mean2 + 0.1 * mean2
        running_var2 = 0.9 * running_var2 + 0.1 * var2
        
        # Output layer
        out_in = np.dot(hl2_a, wo) + bo
        mean_out = np.mean(out_in, axis=0, keepdims=True)
        var_out = np.var(out_in, axis=0, keepdims=True)
        out_in_norm = (out_in - mean_out) / np.sqrt(var_out + 1e-8)
        out_in_scaled = gamma_out * out_in_norm + beta_out
        out_a = softmax(out_in_scaled)
        
        running_mean_out = 0.9 * running_mean_out + 0.1 * mean_out
        running_var_out = 0.9 * running_var_out + 0.1 * var_out
        
        # Compute loss (cross-entropy)
        loss = -np.sum(Y_batch * np.log(out_a + 1e-8)) / batch_size
        error += loss
        
        # Backward pass
        # Output layer
        d_out = out_a - Y_batch
        
        # Batch norm backprop for output layer
        d_out_in_scaled = d_out
        d_gamma_out = np.sum(d_out_in_scaled * out_in_norm, axis=0, keepdims=True)
        d_beta_out = np.sum(d_out_in_scaled, axis=0, keepdims=True)
        d_out_in_norm = d_out_in_scaled * gamma_out
        
        # Continue backprop through batch norm
        std_inv = 1.0 / np.sqrt(var_out + 1e-8)
        d_out_in = d_out_in_norm * std_inv
        d_var_out = np.sum(d_out_in_norm * (out_in - mean_out) * -0.5 * (var_out + 1e-8)**(-1.5), axis=0)
        d_mean_out = np.sum(d_out_in_norm * -std_inv, axis=0) + d_var_out * np.mean(-2 * (out_in - mean_out), axis=0)
        
        d_out_in += d_var_out * 2 * (out_in - mean_out) / batch_size
        d_out_in += d_mean_out / batch_size
        
        d_wo = np.dot(hl2_a.T, d_out_in)
        d_bo = np.sum(d_out_in, axis=0, keepdims=True)
        d_hl2_a = np.dot(d_out_in, wo.T)
        
        # Layer 2 backprop
        d_hl2_in_scaled = d_hl2_a * d_relu(hl2_a)
        d_gamma2 = np.sum(d_hl2_in_scaled * hl2_in_norm, axis=0, keepdims=True)
        d_beta2 = np.sum(d_hl2_in_scaled, axis=0, keepdims=True)
        d_hl2_in_norm = d_hl2_in_scaled * gamma2
        
        std_inv = 1.0 / np.sqrt(var2 + 1e-8)
        d_hl2_in = d_hl2_in_norm * std_inv
        d_var2 = np.sum(d_hl2_in_norm * (hl2_in - mean2) * -0.5 * (var2 + 1e-8)**(-1.5), axis=0)
        d_mean2 = np.sum(d_hl2_in_norm * -std_inv, axis=0) + d_var2 * np.mean(-2 * (hl2_in - mean2), axis=0)
        
        d_hl2_in += d_var2 * 2 * (hl2_in - mean2) / batch_size
        d_hl2_in += d_mean2 / batch_size
        
        d_w2 = np.dot(hl1_a.T, d_hl2_in)
        d_b2 = np.sum(d_hl2_in, axis=0, keepdims=True)
        d_hl1_a = np.dot(d_hl2_in, w2.T)
        
        # Layer 1 backprop
        d_hl1_in_scaled = d_hl1_a * d_relu(hl1_a)
        d_gamma1 = np.sum(d_hl1_in_scaled * hl1_in_norm, axis=0, keepdims=True)
        d_beta1 = np.sum(d_hl1_in_scaled, axis=0, keepdims=True)
        d_hl1_in_norm = d_hl1_in_scaled * gamma1
        
        std_inv = 1.0 / np.sqrt(var1 + 1e-8)
        d_hl1_in = d_hl1_in_norm * std_inv
        d_var1 = np.sum(d_hl1_in_norm * (hl1_in - mean1) * -0.5 * (var1 + 1e-8)**(-1.5), axis=0)
        d_mean1 = np.sum(d_hl1_in_norm * -std_inv, axis=0) + d_var1 * np.mean(-2 * (hl1_in - mean1), axis=0)
        
        d_hl1_in += d_var1 * 2 * (hl1_in - mean1) / batch_size
        d_hl1_in += d_mean1 / batch_size
        
        d_w1 = np.dot(X_batch.T, d_hl1_in)
        d_b1 = np.sum(d_hl1_in, axis=0, keepdims=True)
        
        # Add L2 regularization
        d_wo += 2 * lamda_l2 * wo
        d_w2 += 2 * lamda_l2 * w2
        d_w1 += 2 * lamda_l2 * w1
        
        # Update parameters
        wo -= learning_rate * d_wo
        bo -= learning_rate * d_bo
        w2 -= learning_rate * d_w2
        b2 -= learning_rate * d_b2
        w1 -= learning_rate * d_w1
        b1 -= learning_rate * d_b1
        
        # Update batch norm parameters
        gamma_out -= learning_rate * d_gamma_out
        beta_out -= learning_rate * d_beta_out
        gamma2 -= learning_rate * d_gamma2
        beta2 -= learning_rate * d_beta2
        gamma1 -= learning_rate * d_gamma1
        beta1 -= learning_rate * d_beta1
        
        i += batch_size

    # Validation (using running statistics for batch norm)
    val_hl1_in = np.dot(X_val, w1) + b1
    val_hl1_in_norm = (val_hl1_in - running_mean1) / np.sqrt(running_var1 + 1e-8)
    val_hl1_in_scaled = gamma1 * val_hl1_in_norm + beta1
    val_hl1_a = relu(val_hl1_in_scaled)

    val_hl2_in = np.dot(val_hl1_a, w2) + b2
    val_hl2_in_norm = (val_hl2_in - running_mean2) / np.sqrt(running_var2 + 1e-8)
    val_hl2_in_scaled = gamma2 * val_hl2_in_norm + beta2
    val_hl2_a = relu(val_hl2_in_scaled)

    val_out_in = np.dot(val_hl2_a, wo) + bo
    val_out_in_norm = (val_out_in - running_mean_out) / np.sqrt(running_var_out + 1e-8)
    val_out_in_scaled = gamma_out * val_out_in_norm + beta_out
    val_out_a = softmax(val_out_in_scaled)

    val_loss = -np.sum(Y_val * np.log(val_out_a + 1e-8)) / len(X_val)

    # Check if validation loss improved
    if val_loss < best_validation_loss:
        best_validation_loss = val_loss
        wait = 0
        # Save the model weights
        weights = [w1, w2, wo, b1, b2, bo]
        pkl.dump(weights, open('np_model/weights/model_weights_best.pkl', 'wb'))
    else:
        wait += 1

    # Check if patience is exhausted
    if wait >= patience:
        print(f"Early stopping after {j} epochs.")
        break

    # Print progress
    if j % 10 == 0:
        print(f"Epoch {j}, Training Loss: {error/(N_train/batch_size)}, Validation Loss: {val_loss}")

# Save the final model
weights = [w1, w2, wo, b1, b2, bo]
pkl.dump(weights, open('np_model/weights/model_weights_final.pkl', 'wb'))