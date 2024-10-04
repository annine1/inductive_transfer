import torch
import numpy as np
import random

class IncreaseTemp():
    def __init__(self): #-> None:
        super().__init__()
    
    def call(self, X_batch):
        temp_increase = torch.randint(1, 4, (X_batch.size(0), 1, 1)).float()  
        X_batch[:, :, temp_feature_index] += temp_increase
        
        return X_batch
    
    def call_us(self, X_batch):
        temp_increase = torch.randint(1, 4, (X_batch.size(0), 1, 1)).float()  
        X_batch[:, :, temp_feature_index] += temp_increase
        
        return X_batch
    
