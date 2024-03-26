import numpy as np
from scipy.linalg import qr

class QDEIM():
    def __init__(self, rank = None):
        self.rank = rank
        
    def fit(self, Psi):
        n = len(Psi)
        r = self.rank # select rank truncation
            
        Q, R, pivot = qr(Psi.T, pivoting=True)
        P_qr = pivot[:r]
        P = np.zeros((n, r))
        for jj in range(r):
            P[P_qr[jj], jj] = 1
        return P