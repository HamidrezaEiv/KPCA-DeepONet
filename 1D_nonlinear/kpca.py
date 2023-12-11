import numpy as np
from scipy import linalg
from sklearn.metrics.pairwise import pairwise_kernels
# from sklearn.preprocessing import KernelCenterer as kc
from sklearn.utils.extmath import svd_flip
# from sklearn.linear_model import ridge_regression as ridge

class kpca():
    def __init__(self, kernel, gamma=None, degree=3, coef0=1.0, alpha=1.0):
        super(kpca, self).__init__()
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha

    def get_kernelx(self, X, Y=None):
        params = {"gamma": self.gamma[0], "degree": self.degree[0], "coef0": self.coef0[0]}
        self.params = params
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)
    
    def get_kernelz(self, X, Y=None):
        params = {"gamma": self.gamma[1], "degree": self.degree[1], "coef0": self.coef0[1]}
        self.params = params
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)
    
    def fit(self, X, n_components):
        """
        Perform Kernel Proper Orthogonal Decomposition (Kernel POD) on the input data X.
    
        :param X: Input data (n_samples, n_features)
        :param n_components: Number of dominant modes to retain
        :param gamma: Kernel parameter (bandwidth) for the Gaussian (RBF) kernel
        :return: Dominant modes and corresponding coefficients
        """
        self.X = X
        K = self.get_kernelx(X)
        # K = kc().fit_transform(K)
        # Eigenvalue decomposition of the kernel matrix
        eigenvalues, eigenvectors = np.linalg.eigh(K)
        
        # eigenvalues = np.flip(eigenvalues)
        # eigenvectors = np.fliplr(eigenvectors)
        
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

        # self.dual_coef = np.linalg.pinv(Kz, rcond=1e-9, hermitian=True) @ X
        
        # self.dual_coef = ridge(Kz, X, alpha=self.alpha).T
        
        # p = np.dot(X.T, self.z)
        # q, r = np.linalg.qr(p)
            
        return self.dual_coef, self.z
    
    def transform(self, X, c=0):
        gamma = self.gamma[1]
        coef0 = self.coef0[1]
        degree = self.degree[1]
        
        X = X - c
        K = self.get_kernelx(X, self.X)
        z = np.dot(K, self.modes)

        Kz = self.get_kernelz(z, self.z)
        # if self.kernel == 'poly':
        #     Kz = (gamma * np.dot(z, self.z.T) + coef0)**degree
        # elif self.kernel == 'rbf':
        #     dist = -2 * np.dot(z, self.z.T) + np.sum(z**2, axis=1, keepdims=True) + np.sum(self.z**2, axis=1, keepdims=True).T
        #     Kz = np.exp(-gamma * dist)
        # else:
        #     Kz = np.tanh(gamma *  np.dot(z, self.z.T) + coef0)
        
        Xr = np.dot(Kz, self.dual_coef) + c
        X = X + c
        err = np.mean(np.linalg.norm(X - Xr, axis=1) / np.linalg.norm(X, axis=1)) * 100
        return Xr, z, err
