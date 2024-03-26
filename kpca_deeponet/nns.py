import tensorflow as tf
from tensorflow.keras import models, layers, metrics
    
class konet(models.Model):
    def __init__(self, branch, *args, **kwargs):
        super(konet, self).__init__(*args, **kwargs)
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
        
        if params['kernel'] == 'poly':
            self.kernel_func = self.poly_kernel
        elif params['kernel'] == 'rbf':
            self.kernel_func = self.rbf_kernel
        elif params['kernel'] == 'sigmoid':
            self.kernel_func = self.sigmoid_kernel
    
    @tf.function
    def poly_kernel(self, b):
        kb = tf.linalg.matmul(b, self.z, transpose_b=True)
        kb = (self.gamma * kb + self.coef0)**self.degree
        return kb
    
    @tf.function
    def rbf_kernel(self, b):
        kb = -2 * tf.linalg.matmul(b, self.z, transpose_b=True) + tf.reduce_sum(b**2, axis=1, keepdims=True) + tf.transpose(tf.reduce_sum(self.z**2, axis=1, keepdims=True))
        kb = tf.exp(-self.gamma * kb)
        return kb
    
    @tf.function
    def sigmoid_kernel(self, b):
        kb = tf.linalg.matmul(b, self.z, transpose_b=True)
        kb = tf.tanh(self.gamma * kb + self.coef0)
        return kb

    @tf.function
    def call(self, inputs):
        b = self.branch(inputs)
        kb = self.kernel_func(b)
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

    