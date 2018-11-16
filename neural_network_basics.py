import numpy as np 
import matplotlib.pyplot as plt 

X = np.array(([3,5],[5,1],[10,2]),dtype= float)
y = np.array(([80],[75],[56]),dtype = float)

X = X/ np.amax(X, axis = 0)
y = y/100

class Neural_Network(object):
    def __init__(self):
        self.inputLayerSize = 2
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1
        self.W1 = np.random.rand(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.rand(self.hiddenLayerSize,self.outputLayerSize)
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def forward(self,X):
        self.z2 = np.dot(X,self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat
NN = Neural_Network()
yhat = NN.forward(X)
print(yhat)
print(y)
    
    