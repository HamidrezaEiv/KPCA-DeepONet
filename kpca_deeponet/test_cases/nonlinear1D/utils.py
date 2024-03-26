import numpy as np
from scipy import io
import h5py

class prep():
    def __init__(self):
        super(prep, self).__init__()
        self.mean = 0
        self.std = 1
        
    def fun(self, x, mu):
        return (1-x) * np.cos(3*np.pi*mu*(x + 1)) * np.exp(-(1 + x)*mu)
        
    def get_data(self):
        SEED = 123
        np.random.seed(SEED)
        x = np.linspace(-1, 1, 100, endpoint=True)
        x_train = np.linspace(1, np.pi, 51, endpoint=True)
        y_train = np.stack([self.fun(x, mu) for mu in x_train], 0)
        
        x_test = np.random.uniform(low=1, high=np.pi, size=(51,))
        y_test = np.stack([self.fun(x, mu) for mu in x_test], 0)
        
        return x_train, y_train, x_test, y_test, x[..., None]
        
    