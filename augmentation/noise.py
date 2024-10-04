import torch
import numpy as np

class Noise():
    def __init__(self, mu: float=0.0, std:float=1.0, std_y:float=0.9): 
        super().__init__()
        self.mu = mu
        self.std = std
        
    def call(self, input):
        noise = torch.normal(mean=self.mu, std=self.std, size=input.shape)
        input = input + noise
        
        return input   
           
