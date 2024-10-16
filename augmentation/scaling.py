import numpy as np
import torch

class Scaling():
    def __init__(self, mu:float=1.0, std:float=0.03) :
        self.mu = mu
        self.std = std
        
    def callxs(self, data):
        scalar = np.random.normal(loc=1.0, scale=self.std, size=(1, data.shape[1]))
        noise = np.matmul(np.ones((data.shape[0], 1)), scalar)
        scaling = data * noise
        scaling = scaling.float()
        
        return scaling    
    
    def call(self, data):     
        scalar = np.random.normal(loc=1.0, scale=self.std, size=(1, 1, data.shape[2]))
        noise = np.matmul(np.ones((data.shape[0], 1, 1)), scalar)
        scaling = data * noise
        scaling = scaling.float()
            
        return scaling
