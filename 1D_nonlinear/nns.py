import tensorflow as tf
from tensorflow.keras import models, layers, metrics

class DotProd(layers.Layer):
    def __init__(self, units, *args, **kwargs):
        super(DotProd, self).__init__(*args, **kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True)
    def call(self, x):
        b, phi = x
        return tf.einsum("bi,ni->bn", b, phi) + self.bias
    
class deeponet(models.Model):
    def __init__(self, branch, trunk, *args, **kwargs):
        super(deeponet, self).__init__(*args, **kwargs)
        self.branch = branch
        self.trunk = trunk
        self.loss_tracker_1 = metrics.Mean(name="loss")
        self.loss_tracker_2 = metrics.Mean(name="l2-norm error")
        
        self.dotprod_1 = DotProd(1)
    
    def compile_grid_modes(self, grid, modes = None):
        self.x = tf.convert_to_tensor(grid, dtype=tf.float32)
    
    @tf.function
    def call(self, inputs):
        b = self.branch(inputs)
        phi = self.trunk(self.x)
        phi = tf.concat(phi, 0)        
        u = self.dotprod_1([b, phi])
        return u
   
    @tf.function
    def train_step(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape:
            u = self(inputs, training=True)
            loss = self.loss(outputs, u)
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        error = tf.norm(outputs - u, axis=-1) / tf.norm(outputs, axis=-1) * 100
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}

    @property
    def metrics(self):
        return [self.loss_tracker_1, self.loss_tracker_2]
    
    @tf.function
    def test_step(self, data):
        inputs, outputs = data
        u = self(inputs, training=False)
        loss = self.loss(outputs, u)
        
        error = tf.norm(outputs - u, axis=-1) / tf.norm(outputs, axis=-1) * 100
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}
    
class pod_deepnet(models.Model):
    def __init__(self, branch, *args, **kwargs):
        super(pod_deepnet, self).__init__(*args, **kwargs)
        self.branch = branch
        self.loss_tracker_1 = metrics.Mean(name="loss")
        self.loss_tracker_2 = metrics.Mean(name="l2-norm error")
            
    def compile_grid_modes(self, grid, modes = None, c = None, p = 1):
        self.x = tf.convert_to_tensor(grid, dtype=tf.float32)
        self.modes = tf.convert_to_tensor(modes, dtype=tf.float32)
        self.c = tf.convert_to_tensor(c, dtype=tf.float32)
        self.p = tf.convert_to_tensor(p, dtype=tf.float32)
    
    @tf.function
    def call(self, inputs):
        b = self.branch(inputs)        
        u = (self.modes @ tf.transpose(b))/self.p + self.c
        return tf.transpose(u), b
   
    @tf.function
    def train_step(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape:
            u, b = self(inputs, training=True)
            loss = self.loss(outputs, u)
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        error = tf.norm(outputs - u, axis=-1) / tf.norm(outputs, axis=-1) * 100
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}

    @property
    def metrics(self):
        return [self.loss_tracker_1, self.loss_tracker_2]
    
    @tf.function
    def test_step(self, data):
        inputs, outputs = data
        u, b = self(inputs, training=False)
        loss = self.loss(outputs, u)
        
        error = tf.norm(outputs - u, axis=-1) / tf.norm(outputs, axis=-1) * 100
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}
    
    
class deim_deepnet(models.Model):
    def __init__(self, branch, *args, **kwargs):
        super(deim_deepnet, self).__init__(*args, **kwargs)
        self.branch = branch
        self.loss_tracker_1 = metrics.Mean(name="loss")
        self.loss_tracker_2 = metrics.Mean(name="l2-norm error")
            
    def compile_grid_modes(self, grid, modes = None, PT = None):
        self.x = tf.convert_to_tensor(grid, dtype=tf.float32)
        self.modes = tf.convert_to_tensor(modes, dtype=tf.float32)
        self.PT = tf.convert_to_tensor(PT, dtype=tf.float32)
    
    @tf.function
    def call(self, inputs):
        b = self.branch(inputs)
        u = self.modes @ tf.transpose(b)
        return tf.transpose(u), b
   
    @tf.function
    def train_step(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape:
            u, b = self(inputs, training=True)
            loss = self.loss(outputs, u)
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        error = tf.norm(outputs - u, axis=-1) / tf.norm(outputs, axis=-1) * 100
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}

    @property
    def metrics(self):
        return [self.loss_tracker_1, self.loss_tracker_2]
    
    @tf.function
    def test_step(self, data):
        inputs, outputs = data
        u, b = self(inputs, training=False)
        loss = self.loss(outputs, u)
        
        error = tf.norm(outputs - u, axis=-1) / tf.norm(outputs, axis=-1) * 100
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}
    
class kpod_deepnet(models.Model):
    def __init__(self, branch, *args, **kwargs):
        super(kpod_deepnet, self).__init__(*args, **kwargs)
        self.branch = branch
        self.loss_tracker_1 = metrics.Mean(name="loss")
        self.loss_tracker_2 = metrics.Mean(name="l2-norm error")
        
    def compile_grid_modes(self, grid, modes = None, z = None, params = None, c = None, p = 1):
        self.x = tf.convert_to_tensor(grid, dtype=tf.float32)
        self.modes = tf.convert_to_tensor(modes, dtype=tf.float32)
        self.z = tf.convert_to_tensor(z, dtype=tf.float32)
        self.gamma = tf.constant([params['gamma']],  dtype=tf.float32)
        self.coef0 = tf.constant([params['coef0']],  dtype=tf.float32)
        self.degree = tf.constant([params['degree']],  dtype=tf.float32)
        self.c = tf.convert_to_tensor(c, dtype=tf.float32)
        self.p = tf.convert_to_tensor(p, dtype=tf.float32)
        
    @tf.function
    def call(self, inputs):
        b = self.branch(inputs)
        
        ## rbf
        # kb = -2 * tf.linalg.matmul(b, self.z, transpose_b=True) + tf.reduce_sum(b**2, axis=1, keepdims=True) + tf.transpose(tf.reduce_sum(self.z**2, axis=1, keepdims=True))
        # kb = tf.exp(-self.gamma * kb)
        
        ## poly
        kb = tf.linalg.matmul(b, self.z, transpose_b=True)
        kb = (self.gamma * kb + self.coef0)**self.degree
        
        ## sigmoid
        # kb = tf.linalg.matmul(b, self.z, transpose_b=True)
        # kb = tf.tanh(self.gamma * kb + self.coef0)
        
        u = tf.linalg.matmul(kb, self.modes, transpose_b=False)/self.p + self.c
        return u, b
   
    @tf.function
    def train_step(self, data):
        inputs, outputs = data
        with tf.GradientTape() as tape:
            u, b = self(inputs, training=True)
            loss = self.loss(outputs[1], b) 
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        error = tf.norm(outputs[0] - u, axis=-1) / tf.norm(outputs[0], axis=-1) * 100
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}

    @property
    def metrics(self):
        return [self.loss_tracker_1, self.loss_tracker_2]
    
    @tf.function
    def test_step(self, data):
        inputs, outputs = data
        u, b = self(inputs, training=False)
        loss = self.loss(outputs[1], b)
        
        error = tf.norm(outputs[0] - u, axis=-1) / tf.norm(outputs[0], axis=-1) * 100
        self.loss_tracker_1.update_state(loss)
        self.loss_tracker_2.update_state(error)
        return {"loss": self.loss_tracker_1.result(), "l2-norm error": self.loss_tracker_2.result()}
    
import numpy as np 
def svdp(y, modes, c=None):
    x = y.T - c
    z = np.linalg.pinv(modes) @ x
    yr = (modes @ z + c).T
    print(np.mean(np.linalg.norm(y - yr, axis=1) / np.linalg.norm(y, axis=1)) * 100)
    return yr, z.T
    