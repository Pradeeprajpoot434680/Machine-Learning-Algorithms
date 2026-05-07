import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



class BaggingClassifier:

    def __init__(self, n_models=10):
        self.n_models = n_models
        self.models = []

    def fit(self, X, y):
        n_samples = X.shape[0]

        for _ in range(self.n_models):

            # bootstrap sample
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[idx]
            y_sample = y[idx]

            # base model
            model = DecisionTreeClassifier()
            model.fit(X_sample, y_sample)

            self.models.append(model)

    def predict(self, X):

        predictions = np.array([model.predict(X) for model in self.models])

        # majority voting
        final_pred = []

        for i in range(X.shape[0]):
            counts = np.bincount(predictions[:, i])
            final_pred.append(np.argmax(counts))

        return np.array(final_pred)
    



bag = BaggingClassifier(n_models=10)
bag.fit(X_train, y_train)

y_pred = bag.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))