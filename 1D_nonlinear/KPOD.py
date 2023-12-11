#%%
import sys
import os

import random
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras import optimizers, losses, models, layers
from utils import prep
import joblib
from time import time

from nns import kpod_deepnet as PDN
from kpca import kpca
from qdeim import QDEIM

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

tf.config.set_visible_devices(gpus, 'GPU')
#%%
rescale = False
verbose = 0
def training(seed, d):
    SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    test_name = f'kpod_rescale_{rescale}'
    model_name = f'seed{seed}_d{d}'
    
    if not os.path.exists(f'./res/{test_name}'):
        os.mkdir(f'./res/{test_name}')
        
    if not os.path.exists(f'./res/{test_name}/{model_name}'):
        os.mkdir(f'./res/{test_name}/{model_name}')
    
    sdir = f'./res/{test_name}/{model_name}/'
    
    pp = prep()
    x_train, y_train, x_test, y_test, grid = pp.get_data()
    #%%
    Xs = y_train
    c = Xs.mean(0)[None, ...]
    Xs = Xs - c
    
    modes, sigma, vh = np.linalg.svd(Xs, full_matrices=False)
    s = np.cumsum(sigma) / np.sum(sigma)
    rr = np.argmax(s > 0.9999999) + 1
    modes = modes[:, :rr]
    
    deim = QDEIM(rank = rr)
    P = deim.fit(modes)
    
    # Xs = P.T @ Xs
    r = d
    
    # kern = kpca('poly', gamma=[1.0, 1.0], degree=[1.0, 2.0], coef0=[1.0, 1.0], alpha=0.01)
    kern = kpca('poly', gamma=[1.0, 1.0], degree=[1.0, 2.0], coef0=[0.0, 0.0], alpha=0.001)

    modes, z = kern.fit(Xs, d)
    _, z_train, err_rec = kern.transform(y_train, c)
    y_pred, z_test, err = kern.transform(y_test, c)
    print(err)
    #%%
    def gen_models(nf, nv, act, nn, nl, n_out):
        inp = layers.Input((nf,))
        x = layers.Dense(nn, activation = act)(inp)
        for i in range(nl - 1):
            x = layers.Dense(nn, activation = act)(x)
            
        outs = []
        for i in range(n_out):
            outs.append(layers.Dense(nv)(x))
            
        model = models.Model(inp, outs)
        return model
    
    m = 1
    act = tf.keras.activations.tanh
    branch = gen_models(m, r, act, 64, 4, 1)
    print(branch.summary())
    #%%
    p = 1
    if rescale:
        p = d**0.5
        
    n_batches = 1
    batch_size = int(len(x_train) / n_batches)
    
    initial_learning_rate = 1e-3
    lr_schedule = optimizers.schedules.InverseTimeDecay(
        initial_learning_rate,
        decay_steps=1,
        decay_rate=1e-3)
    
    model = PDN(branch)
    model.compile_grid_modes(grid, modes, z, kern.params, c, p)
    model.compile(
        optimizer = optimizers.Adam(learning_rate=lr_schedule),
        loss = losses.MeanSquaredError(),
        )
    start_time = time()
    hist = model.fit(x_train, [y_train, z_train], 
              epochs=100000, 
              batch_size = batch_size, 
              validation_data=(x_test, [y_test, z_test]), 
              verbose = verbose)
    end_time = time()
    cp_time = end_time - start_time
    
    model.save(sdir + 'model', save_format = 'tf')
    hist = hist.history
    
    hist = np.array(hist)
    np.savez_compressed(sdir + 'res', hist = hist, cp_time = cp_time)
    joblib.dump(modes, sdir + 'U')
    joblib.dump(pp, sdir + 'pp')
    #%%
    y_pred, z_pred = model.predict(x_test)
    
    print(np.mean(np.linalg.norm(y_test - y_pred, axis=1) / np.linalg.norm(y_test, axis=1)) * 100)

for seed in range(3, 5):
    for d in range(6, 20):
        print(seed, d)
        training(seed, d)