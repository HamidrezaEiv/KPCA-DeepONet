import numpy as np
from scipy import linalg
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.extmath import svd_flip
from kpca_deeponet.qdeim import QDEIM

class kpca():
    def __init__(self, kernel, gamma=None, degree=2, coef0=1.0, alpha=1.0, sparse=False):
        super(kpca, self).__init__()
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        
        self.sparse = sparse
        
    def get_kernelx(self, X, Y=None):
        params = {"kernel": self.kernel, "gamma": self.gamma[0], "degree": self.degree[0], "coef0": self.coef0[0]}
        self.params = params
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)
    
    def get_kernelz(self, X, Y=None):
        params = {"kernel": self.kernel, "gamma": self.gamma[1], "degree": self.degree[1], "coef0": self.coef0[1]}
        self.params = params
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)
    
    def fit(self, X, n_components):
        """
        Perform Kernel principal component analysis (KPCA) on the input data X.
    
        :param X: Input data (n_samples, n_features)
        :param n_components: Number of dominant modes to retain
        """
        
        self.c = X.mean(0)[None, ...]
        X = X - self.c
        
        if self.sparse:
            X, self.deim_modes = self.sparsify(X)
        
        self.X = X
        K = self.get_kernelx(X)
        # Eigenvalue decomposition of the kernel matrix
        eigenvalues, eigenvectors = np.linalg.eigh(K)
        
        # flip eigenvectors' sign to enforce deterministic output
        eigenvectors, _ = svd_flip(
            eigenvectors, np.zeros_like(eigenvectors).T
        )

        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        eigenvectors = eigenvectors[:, :n_components]
        eigenvalues = eigenvalues[:n_components]
        
        eigenvectors = eigenvectors[:, eigenvalues > 0] 
        eigenvalues = eigenvalues[eigenvalues > 0]
        
        self.modes = eigenvectors / np.sqrt(eigenvalues) 
        self.eigenvalues = eigenvalues
        
        self.z = eigenvectors * np.sqrt(eigenvalues)
        Kz = self.get_kernelz(self.z)
        
        n_samples = self.z.shape[0]
        Kz.flat[:: n_samples + 1] += self.alpha
        self.dual_coef = linalg.solve(Kz, X, overwrite_a=True)
        
        if self.sparse:
            self.dual_coef = self.dual_coef @ self.deim_modes.T
            
        return self.dual_coef, self.z
    
    def transform(self, X):
        Xs = X - self.c
        
        if self.sparse:
            Xs = Xs @ self.P
            
        K = self.get_kernelx(Xs, self.X)
        z = np.dot(K, self.modes)
        Kz = self.get_kernelz(z, self.z)
        Xr = np.dot(Kz, self.dual_coef) + self.c

        err = np.mean(np.linalg.norm(X - Xr, axis=1) / np.linalg.norm(X, axis=1)) * 100
        return Xr, z, err
    
    def sparsify(self, X):
        modes, sigma, vh = np.linalg.svd(X.T, full_matrices=False)
        s = np.cumsum(sigma) / np.sum(sigma)
        rr = np.argmax(s > 0.9999) + 1
        
        modes = modes[:, :rr]
        deim = QDEIM(rank = rr)
        
        self.P = deim.fit(modes)
        modes = modes @ np.linalg.inv(self.P.T @ modes)

        return X @ self.P, modes
        
