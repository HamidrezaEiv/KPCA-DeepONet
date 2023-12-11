#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras import models
from utils import prep
import joblib

SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

tf.config.set_visible_devices(gpus, 'GPU')
#%%
seed = 4
d = 11
test_name = f'seed{seed}_d{d}'
model_names = [
                # 'deeponet',
                'pod_rescale_True',
                'kpod_rescale_True',
               # 'kpca',
               # 'deim_pod',
               # 'deim_kpod',
               ]

pp = prep()
x_train, y_train, x_test, y_test, grid = pp.get_data()
#%%
for model_name in model_names:
    sdir = f'./res/{model_name}/{test_name}/'

    model = models.load_model(sdir + 'model')
    pps = joblib.load(sdir + 'pp')
    y_pred = model.predict(x_test)
    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]
    
    ae = np.abs(y_test - y_pred)
    print('mean:', ae.mean(), 'max:', ae.max())
    l2 = np.mean(np.linalg.norm(y_test - y_pred, axis=1) / np.linalg.norm(y_test, axis=1)) * 100
    print(model_name + ':\t', np.round(l2, 2))

