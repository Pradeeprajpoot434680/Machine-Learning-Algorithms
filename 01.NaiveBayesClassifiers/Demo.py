
import math


X = [
    [1.0, 20.0],
    [2.0, 21.0],
    [3.0, 22.0],
    [8.0, 30.0],
    [9.0, 31.0],
    [10.0, 32.0]
]

y = [0, 0, 0, 1, 1, 1]

class NaiveBayes :
    def fit(self,X,y):
        self.classes = set(y) # uni values
        self.data = {}

        for c in self.classes:
            rows=[]
            for i in range(len(X)):
                if c == y[i]:
                    rows.append(X[i])
            self.data[c] = list(zip(*rows))
        
    def mean(self,value):
        return sum(value)/len(value)
    
    def varience(self,value):
        m = self.mean(value)
        return sum((x-m)**2 for x in value)/len(value)
    
    def gaussian(self,x,mean,var):
         return (1 / math.sqrt(2 * math.pi * var)) * math.exp(-(x - mean) ** 2 / (2 * var))



    def predict(self,rows):
        probs={}
        for c in self.classes:
            prob = 1
            for i , column in enumerate(self.data[c]):
                m = self.mean(column)
                v = self.varience(column)
                prob *= self.gaussian(rows[i], m, v)
            probs[c] = prob
        return max(probs, key=probs.get)
