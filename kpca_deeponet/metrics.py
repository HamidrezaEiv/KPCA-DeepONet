import numpy as np

def l2_norm_error(y_ref, y_pred):
    return np.mean(np.linalg.norm(y_ref - y_pred, axis=1) / np.linalg.norm(y_ref, axis=1)) * 100
        
    