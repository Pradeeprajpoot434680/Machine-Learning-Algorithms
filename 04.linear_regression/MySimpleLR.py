import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class MySimpleLR:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.b = None
        self.m = None  

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train  

        num = np.sum((X_train - np.mean(X_train)) * (y_train - np.mean(y_train)))
        den = np.sum((X_train - np.mean(X_train)) ** 2)

        self.m = num / den
        self.b = np.mean(y_train) - self.m * np.mean(X_train)

    def predict(self, X_test):
        return self.m * X_test + self.b


df = pd.read_csv('placement.csv')

X = df.iloc[:, 0].values
y = df.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

myLr = MySimpleLR()
myLr.train(X_train, y_train)

print(myLr.predict(X_test))