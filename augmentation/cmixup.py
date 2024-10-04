import torch
import numpy as np
from neuralhydrology.augmentation.utils import stats_values
from neuralhydrology.utils.config import Config
from sklearn.neighbors import KernelDensity
from sklearn.metrics import pairwise_distances

# from neuralhydrology.augmentation.dtwdist import DTWD
import dtaidistance
from dtaidistance import dtw
from scipy.spatial.distance import euclidean


class CMixup():
    def __init__(self, cfg:Config, bw:float = 1.0, alpha:float = 0.2):
        super().__init__()
        self.cfg = cfg
        self.bw = bw
        self.alpha = alpha

    def rate(self, y):
        
        y_reshape = y.reshape(-1, y.shape[0]) # get 2D
        
        y_reshape = y_reshape.nan_to_num()
              
        kd = KernelDensity(bandwidth=self.bw, algorithm='auto', kernel='linear', 
                     metric="euclidean").fit(y_reshape)  
                
        log_density = kd.score_samples(y_reshape)
        each_rate = torch.exp(torch.from_numpy(log_density))
        
           
        sum_prob = torch.sum(each_rate)
        each_rate = each_rate / sum_prob  
        
        return each_rate    

    def callxs(self, X, y):
        n = X.shape[0]
        y_reshape = y.reshape(-1, y.shape[1]) # get 2D
        
        y_reshape = y_reshape.nan_to_num()
        
        kd = KernelDensity(bandwidth=self.bw, algorithm='auto', kernel='linear', 
                           metric="euclidean").fit(y_reshape)  
                
        log_density = kd.score_samples(y_reshape)
        
        each_rate = torch.exp(torch.from_numpy(log_density))
           
        sum_prob = torch.sum(each_rate)
        each_rate = each_rate / sum_prob   
        each_rate = each_rate.numpy()
        m = len(each_rate)
        
        j = np.random.choice(m, size=n, p=each_rate)
        new_X = X[j]
        
        lam = np.random.beta(self.alpha, self.alpha)
        
        mixed_X = lam * X + (1 - lam) * new_X   
        
        return mixed_X
    
    def call(self, data, y):
        n_seq = data.shape[1]
        m1 = data.shape[1]
        batch_size = data.shape[0]      
            
        iteration = m1 // batch_size         
            
        batch_rate = self.rate(y)
        batch_rate = batch_rate.numpy()
            
        m = len(batch_rate)
        sum_prob = np.sum(batch_rate)
        j = np.random.choice(m, size=n_seq, p=batch_rate)  
        new_X = data[:,j,:]                 
        new_y = y[:,j,:]
        lam = np.random.beta(self.alpha, self.alpha)            
        mixed_X = lam * data + (1 - lam) * new_X   
        mixed_y = lam * y + (1 - lam) * new_y
                                
        return mixed_X , mixed_y


