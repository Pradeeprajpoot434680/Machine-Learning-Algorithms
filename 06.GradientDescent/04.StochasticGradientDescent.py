from sklearn.datasets import load_diabetes

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import random
X,y = load_diabetes(return_X_y=True)

# print(X.shape)
# print(y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

reg = LinearRegression()
reg.fit(X_train,y_train)

print(reg.coef_)
print(reg.intercept_)

y_pred = reg.predict(X_test)
print("Sklearn r2 score=> ",r2_score(y_test,y_pred))

print("##############################################")


class SGDRegressior:
    def __init__(self,learning_rate,epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self,X_train,y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])
        for i in range(self.epochs):
            for j in range(X_train.shape[0]):
            # for updaing the B0
                idx = np.random.randint(0,X_train.shape[0])
                y_hat = np.dot(X_train[idx],self.coef_)+self.intercept_

                intercept_derivative = -2*(y_train[idx] - y_hat)
                self.intercept_ = self.intercept_ - self.learning_rate*intercept_derivative  

                coef_der = -2*np.dot((y_train[idx] - y_hat),X_train[idx])

                self.coef_ = self.coef_ - (self.learning_rate*coef_der)

        print(self.coef_,self.intercept_)

    def predict(self,X_test):
        return np.dot(X_test,self.coef_) + self.intercept_
    

gdr = SGDRegressior(epochs=600,learning_rate=0.3)
gdr.fit(X_train,y_train)



y_pred = gdr.predict(X_test)
print("My Model R2 Score => " ,r2_score(y_test,y_pred))