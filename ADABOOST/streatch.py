# AdaBoost From Scratch using NumPy

import numpy as np

# -----------------------------------
# Sample Dataset
# -----------------------------------

X = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Labels
# -1 = Negative Class
# +1 = Positive Class

y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

# -----------------------------------
# Initialize Weights
# -----------------------------------

n_samples = len(X)

weights = np.ones(n_samples) / n_samples

# Number of weak learners
n_estimators = 3

# Store alpha values
alphas = []

# Store weak learner thresholds
thresholds = []

# -----------------------------------
# AdaBoost Training
# -----------------------------------

for estimator in range(n_estimators):

    best_threshold = None
    best_predictions = None
    min_error = float('inf')

    # --------------------------------
    # Try different thresholds
    # --------------------------------

    for threshold in X:

        predictions = np.ones(n_samples)

        # Simple Decision Stump
        predictions[X < threshold] = -1

        # Calculate weighted error
        misclassified = predictions != y

        error = np.sum(weights * misclassified)

        # Find best threshold
        if error < min_error:
            min_error = error
            best_threshold = threshold
            best_predictions = predictions.copy()

    # --------------------------------
    # Calculate Alpha
    # --------------------------------

    alpha = 0.5 * np.log((1 - min_error) / (min_error + 1e-10))

    # Store values
    alphas.append(alpha)
    thresholds.append(best_threshold)

    # --------------------------------
    # Update Weights
    # --------------------------------

    weights = weights * np.exp(-alpha * y * best_predictions)

    # Normalize weights
    weights = weights / np.sum(weights)

    # --------------------------------
    # Print Iteration Details
    # --------------------------------

    print(f"\nEstimator {estimator + 1}")
    print("Best Threshold:", best_threshold)
    print("Error:", min_error)
    print("Alpha:", alpha)
    print("Updated Weights:", weights)

# -----------------------------------
# Prediction Function
# -----------------------------------

def predict(X_test):

    final_prediction = np.zeros(len(X_test))

    for alpha, threshold in zip(alphas, thresholds):

        prediction = np.ones(len(X_test))
        prediction[X_test < threshold] = -1

        final_prediction += alpha * prediction

    return np.sign(final_prediction)

# -----------------------------------
# Test Predictions
# -----------------------------------

X_test = np.array([1.5, 3.5, 6.5, 7.5])

predictions = predict(X_test)

print("\nTest Data:", X_test)
print("Predictions:", predictions)