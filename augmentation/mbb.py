import numpy as np
import torch

from arch.bootstrap import MovingBlockBootstrap
from recombinator.block_bootstrap import moving_block_bootstrap

class MBB():
    """
        Moving Block Bootstrapping data augmentation technique.
        Generates new data by resampling blocks of data and concatenating them.
    
        Parameters:
           X (numpy.ndarray): Features array of shape (batch_size, sequence_length, n_features).
           y (numpy.ndarray): Labels array of shape (batch_size, sequence_length, n_f).
           block_size (int): Size of each block to use (l=n_seq/k).
           n_blocks (int): Number of blocks to resample( n_blocks = n_seq/l). 
        """
    def __init__(self, block_size, seed, n_Bootstrap):
        self.block_size = block_size
        self.seed = seed
        self.n_Bootstrap = n_Bootstrap  
        
        
    def call(self, input):
        input_arr = input.numpy()
        
        mbb = MovingBlockBootstrap(self.block_size, input_arr, seed=self.seed)
        
        for data in mbb.bootstrap(self.n_Bootstrap):
            mbb_data = data[0][0]         
        
        output = torch.from_numpy(mbb_data)
            
        return output  
















# class MBB():
#     """
#         Moving Block Bootstrapping data augmentation technique.
#         Generates new data by resampling blocks of data and concatenating them.
    
#         Parameters:
#            X (numpy.ndarray): Features array of shape (batch_size, sequence_length, n_features).
#            y (numpy.ndarray): Labels array of shape (batch_size, sequence_length, n_classes).
#            block_size (int): Size of each block to use (k=n_seq/n_block)
#            n_blocks (int): Number of blocks to resample( n_blocks = n_seq/k)
        
#         Returns:
#            X_new (numpy.ndarray): Augmented features array of shape (batch_size * num_blocks, sequence_length, n_features).
#            y_new (numpy.ndarray): Augmented labels array of shape (batch_size * num_blocks, sequence_length, n_classes).
#         """
#     def __init__(self, block_size):
#         self.block_size = block_size  
#         # self.n_block = n_block
        
#     def callxd(self, data):
#         batch_size, seq_length, n_features = data.shape
        
#         n_block = int(seq_length / self.block_size)   # block length
#         block_starts = np.random.randint(0, seq_length - self.block_size + 1, size=n_block)
        
#         data_new = np.empty((batch_size * n_block, seq_length, n_features))
        
#         for i, start_idx in enumerate(block_starts):
#             end_idx = start_idx + self.block_size
#             block_data = data[:, start_idx:end_idx, :]
#             block_indices = np.arange(start_idx, end_idx)
#             block_indices_new = np.arange(seq_length * i, seq_length * (i + 1))
#             np.random.shuffle(block_indices_new)
            
#             data_new[i * batch_size:(i + 1) * batch_size, block_indices_new, :] = block_data[:, block_indices, :]
#             print(data_new.dtype, data.dtype)
            
#         return data_new
    
    
#     def callxs(self, data):
#         data = torch.unsqueeze(data, dim=0)
#         batch_size, seq_length, n_features = data.shape
        
#         n_block = int(seq_length / self.block_size)
#         block_starts = np.random.randint(0, seq_length - self.block_size + 1, size=n_block)
        
#         data_new = np.empty((batch_size * n_block, seq_length, n_features))
        
#         for i, start_idx in enumerate(block_starts):
#             end_idx = start_idx + self.block_size
#             block_data = data[:, start_idx:end_idx, :]
#             block_indices = np.arange(start_idx, end_idx)
#             block_indices_new = np.arange(seq_length * i, seq_length * (i + 1))
#             np.random.shuffle(block_indices_new)
            
#             data_new[i * batch_size:(i + 1) * batch_size, block_indices_new, :] = block_data[:, block_indices, :]
        
        
#         data_new = data_new[-1, :, :]
#         print(data_new.dtype, data.dtype)    
            
#         return data_new    
            
    
        
