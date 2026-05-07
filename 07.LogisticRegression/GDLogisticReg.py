import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 🔹 Logistic Regression from scratch
class LogisticRegressionGD:

    def __init__(self, lr=0.5, epochs=5000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):

        # add bias column
        X = np.insert(X, 0, 1, axis=1)

        # initialize weights
        self.weights = np.ones(X.shape[1])

        # gradient descent
        for _ in range(self.epochs):
            z = np.dot(X, self.weights)
            y_hat = self.sigmoid(z)

            gradient = np.dot((y - y_hat), X) / X.shape[0]
            self.weights += self.lr * gradient

    def predict(self, X):

        X = np.insert(X, 0, 1, axis=1)
        z = np.dot(X, self.weights)
        probs = self.sigmoid(z)

        return np.array([1 if i >= 0.5 else 0 for i in probs])


# 🔹 Generate dataset (FIXED)
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=1,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=1,  
    random_state=41,
    class_sep=20
)

# 🔹 Train model
model = LogisticRegressionGD()
model.fit(X, y)

# 🔹 Predictions
y_pred = model.predict(X)

# 🔹 Accuracy (optional but useful)
accuracy = np.mean(y == y_pred)
print("Accuracy:", accuracy)

# 🔹 Plot results
plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y_pred, cmap='winter', s=100)
plt.title("Logistic Regression (GD from scratch)")
plt.show()