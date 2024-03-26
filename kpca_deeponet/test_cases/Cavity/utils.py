import numpy as np

class prep():
    def __init__(self):
        super(prep, self).__init__()
        self.mean = 0.0
        self.std = 1.0
        
    def get_data(self, datadir):
        B = np.linspace(100, 2080, 100, endpoint=True, dtype=int)
        train_files_u = [datadir + 'train/u{0:0=5d}'.format(b) for b in B]
        train_files_v = [datadir + 'train/v{0:0=5d}'.format(b) for b in B]
        
        U = []
        i = 0
        for f in train_files_u:
            u = np.loadtxt(f)
            U.append(u)
            i += 1
            
        U = np.array(U)
        
        V = []
        i = 0
        for f in train_files_v:
            v = np.loadtxt(f)
            V.append(v)
            i += 1
        
        V = np.array(V)
        data = np.stack((U, V), axis = 1)
        x = np.linspace(0, 1, 257, endpoint=True)
        y = np.linspace(0, 1, 257, endpoint=True)
        grid = np.meshgrid(x, y)
        
        return B[..., None], data.reshape((data.shape[0], -1)), np.stack((grid[0].flatten(), grid[1].flatten()), axis = 1)
    
    def get_data_test(self, datadir):
        B = np.array([215, 503, 530, 580, 687, 1168, 1234, 1460, 1648, 1965])
        train_files_u = [datadir + 'test/u{0:0=5d}'.format(b) for b in B]
        train_files_v = [datadir + 'test/v{0:0=5d}'.format(b) for b in B]
        
        U = []
        i = 0
        for f in train_files_u:
            u = np.loadtxt(f)
            U.append(u)
            i += 1
            
        U = np.array(U)
        
        V = []
        i = 0
        for f in train_files_v:
            v = np.loadtxt(f)
            V.append(v)
            i += 1
            
        V = np.array(V)
        data = np.stack((U, V), axis = 1)

        
        return B[..., None], data.reshape((data.shape[0], -1))
        
    