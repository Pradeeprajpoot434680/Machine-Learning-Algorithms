# Neural Network From Scratch using NumPy

import numpy as np

# -----------------------------------
# Input Data (XOR Problem)
# -----------------------------------

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# -----------------------------------
# Activation Function
# -----------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# -----------------------------------
# Initialize Weights and Biases
# -----------------------------------

np.random.seed(42)

# Input layer -> Hidden layer
weights_input_hidden = np.random.rand(2, 2)
bias_hidden = np.random.rand(1, 2)

# Hidden layer -> Output layer
weights_hidden_output = np.random.rand(2, 1)
bias_output = np.random.rand(1, 1)

# -----------------------------------
# Training Parameters
# -----------------------------------

learning_rate = 0.1
epochs = 10000

# -----------------------------------
# Training
# -----------------------------------

for epoch in range(epochs):

    # -----------------------------
    # Forward Propagation
    # -----------------------------

    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(final_input)

    # -----------------------------
    # Calculate Error
    # -----------------------------

    error = y - predicted_output

    # -----------------------------
    # Backpropagation
    # -----------------------------

    d_predicted_output = error * sigmoid_derivative(predicted_output)

    hidden_error = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_output = hidden_error * sigmoid_derivative(hidden_output)

    # -----------------------------
    # Update Weights and Biases
    # -----------------------------

    weights_hidden_output += hidden_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += X.T.dot(d_hidden_output) * learning_rate
    bias_hidden += np.sum(d_hidden_output, axis=0, keepdims=True) * learning_rate

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss}")

# -----------------------------------
# Final Output
# -----------------------------------

print("\nFinal Predictions:")
print(predicted_output)