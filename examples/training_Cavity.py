import numpy as np
import random

import tensorflow as tf
from tensorflow.keras import optimizers, losses, models, layers

from kpca_deeponet.nns import konet
from kpca_deeponet.kpca import kpca
from kpca_deeponet.test_cases.Cavity.utils import prep
from kpca_deeponet.metrics import l2_norm_error

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

tf.config.set_visible_devices(gpus, 'GPU')

def gen_models(nf, nv, act, nn, nl):
    inp = layers.Input((nf,))
    x = layers.Dense(nn, activation = act)(inp)
    for i in range(nl - 1):
        x = layers.Dense(nn, activation = act)(x)
    out = layers.Dense(nv)(x)
    model = models.Model(inp, out)
    return model

def main():
    verbose = 2
    seed, d = 0, 6
    
    SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    pp = prep()
    ddir = '../../data/cavity_steady/'
    x_train, y_train, grid = pp.get_data(ddir)
    x_test, y_test = pp.get_data_test(ddir)
    
    kern = kpca('poly', gamma=[1.0, 1e-2], degree=[1.0, 2.0], coef0=[1.0, 1.0], alpha=1e-6, sparse=False)
    
    modes, z = kern.fit(y_train, d)
    _, z_train, err_rec = kern.transform(y_train)
    y_pred, z_test, err_rec_test = kern.transform(y_test)
    print('relative l2-norm error of reconstruction for the test data: ', err_rec_test)    
    
    m = 1
    act = tf.keras.activations.tanh
    branch = gen_models(m, d, act, 64, 4)
    print(branch.summary())

    n_batches = 1
    batch_size = int(len(x_train) / n_batches)
    
    initial_learning_rate = 1e-3
    lr_schedule = optimizers.schedules.InverseTimeDecay(
        initial_learning_rate,
        decay_steps=1,
        decay_rate=1e-3)
    
    model = konet(branch)
    model.compile_grid_modes(grid, modes, z, kern.params, kern.c, p = 1)
    model.compile(
        optimizer = optimizers.Adam(learning_rate=lr_schedule),
        loss = losses.MeanSquaredError(),
        )

    model.fit(x_train, [y_train, z_train],  
            epochs=20000, 
            batch_size = batch_size, 
            validation_data=(x_test, [y_test, z_test]), 
            verbose = verbose)
    
    y_pred, z_pred = model.predict(x_test)
    print('relative l2-norm error: ', l2_norm_error(y_test, y_pred))
    
if __name__ == "__main__":
    main()