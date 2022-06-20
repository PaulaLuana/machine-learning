import numpy as np
class BinayLogitRegression() :
    def __init__(self, learning_rate, iterations):        
        self.learning_rate = learning_rate        
        self.iterations = iterations
          
    def fit( self, X, Y ) :        
        self.m, self.n = X.shape        
        self.W = np.zeros(self.n)        
        self.b = 0        
        self.X = X        
        self.Y = Y
                  
        for i in range(self.iterations):            
            self.update_weights()            
        return self
      
      
    def update_weights(self):           
        A = 1 / (1 + np.exp(-(self.X.dot(self.W) + self.b)))
        tmp = (A - self.Y.T)        
        tmp = np.reshape(tmp, self.m)        
        dW = np.dot(self.X.T, tmp) / self.m         
        db = np.sum(tmp) / self.m 
          
        self.W = self.W - self.learning_rate * dW    
        self.b = self.b - self.learning_rate * db
          
        return self
      
      
    def predict( self, X ) :    
        Z = 1 / (1 + np.exp(-(X.dot(self.W) + self.b)))        
        Y = np.where(Z > 0.5, 1, 0)        
        return Y
