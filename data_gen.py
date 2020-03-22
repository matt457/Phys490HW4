import numpy as np
import torch

class Data():
    def __init__(self, data_path):
        '''Data generation
        x_* attributes contain 14x14 MNIST images of digits 0,2,4,6,8
        y_* attributes are the digit labels'''
        
        data = np.loadtxt(data_path)
        data_length = len(data)
        y = np.zeros((data_length,1,5))
        x = np.zeros((data_length,1,14,14))
        
        for i in range(data_length):
            data_i = data[i]
            index = int(data_i[-1]/2)
            y[i,0,index] = 1
            x[i,0,:,:] = np.reshape(data_i[:-1],(14,14))
        
        x_train= torch.tensor(np.array(x, dtype= np.float32))
        y_train= torch.tensor(np.array(y, dtype= np.float32))

        self.x_train= x_train
        self.y_train= y_train
