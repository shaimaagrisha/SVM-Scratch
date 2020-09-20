import numpy as np

class svm :
    def __init__ ( self,lr=.01,n_iter=200,lamda_param = .01):
        self.lr = lr
        self.n_iter = n_iter
        self.lamda_param = lamda_param
        self.w = None
        self.b = 0
    def fit (self,x,y):
        if len(x.shape) == 1 :
            no_observation = x.shape
            n_features = 1
        else:
            no_observation,n_features = x.shape
            
        self.w = np.zeros(n_features)
        y_ = np.where(x<=0,-1,1)
        
        for i in range(self.n_iter):
            for idx,x_i in enumerate(x):
                condition = [y_[idx]*(np.dot(x_i,self.w)-self.b) >=1]
                if condition == True :
                    self.w = self.lr*(2*self.lamda_param*self.w)
                else : 
                    self.w -= self.lr*(2*self.lamda_param*self.w - np.dot(x_i,y_[idx]))
                    self.b -=self.lr * y_[idx]
                    
    def predict (self,x):
        output = np.dot(x,self.w)-self.b
        if output[0] <0 : 
            output = 0 
        
        return np.sign(output)
         