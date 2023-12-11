import numpy as np
import jax.numpy as jnp

class DEIM():
    def __init__(self, rank = None):
        self.rank = rank
        
    def fit(self, Psi):
        n = len(Psi)
        r = self.rank # select rank truncation
        
        # First DEIM point
        nmax = np.argmax(np.abs(Psi[:,0]))
        XI_m = Psi[:,0].reshape(n, 1)
        z = np.zeros((n, 1))
        P = np.copy(z)
        P[nmax] = 1
        
        # DEIM points 2 to r
        for jj in range(1, r):
            c = np.linalg.solve(P.T @ XI_m, P.T @ Psi[:, jj].reshape(n, 1))
            res = Psi[:,jj].reshape(n,1) - XI_m @ c
            nmax = np.argmax(np.abs(res))
            XI_m = np.concatenate((XI_m, Psi[:,jj].reshape(n,1)),axis=1)
            P = np.concatenate((P,z),axis=1)
            P[nmax, jj] = 1
                        
        return P
