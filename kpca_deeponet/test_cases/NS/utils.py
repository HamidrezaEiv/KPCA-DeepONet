import numpy as np
from mat73 import loadmat

class prep():
    def __init__(self):
        super(prep, self).__init__()
        self.mean = 0
        self.std = 1
        
    def get_data(self, filename, ntrain, ntest):
        SEED = 24
        np.random.seed(SEED)
        data = loadmat(filename)

        u = data['u']
        
        x_train = u[:ntrain, :, :, :10]
        y_train = u[:ntrain, :, :, 19].reshape((ntrain, -1))
        
        x_test = u[-ntest:, :, :, :10]
        y_test = u[-ntest:, :, :, 19].reshape((ntest, -1))
        
        x = np.linspace(0, 1, 64, endpoint=True)
        y = np.linspace(0, 1, 64, endpoint=True)
        xx, yy = np.meshgrid(x, y)
        grid = np.stack((xx.flatten(), yy.flatten()), axis = 1)
        return x_train, y_train, x_test, y_test, grid
        
    