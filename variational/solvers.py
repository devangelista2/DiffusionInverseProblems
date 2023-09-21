import numpy as np
import tensorflow as tf

class HardGD:
    def __init__(self, G, K, lmbda, optimizer, maxit=200, verbose=False, metric=None):
        self.G = G
        self.K = K
        self.lmbda = lmbda

        self.maxit = maxit

        self.optimizer = optimizer
        self.verbose = verbose

        if metric is not None:
            self.metric = metric

    def __call__(self, y_delta, z=None, x_true=None):
        # Change z0 to a variable.
        if z is not None:
            z = tf.Variable(z, name='z')
        else:
            z = tf.zeros_like(y_delta, name='z')

        # Initialization
        if x_true is not None:
            metric_vec = np.zeros((self.maxit + 1, ))

        # Compute the starting iterate and the corresponding loss
        x_k = self.G(z)
        if x_true is not None:
            metric_vec[0] = self.metric(x_k, x_true)

        # Run the Gradient Descent
        for k in range(1, self.maxit + 1):
            with tf.GradientTape() as tape:
                tape.watch(z)
                x_k = self.G(z)
                d = tf.reduce_sum(tf.square(self.K(x_k) - y_delta)) + self.lmbda * tf.square(tf.norm(z, 'euclidean'))
            gradient = tape.gradient(d, z)

            if x_true is not None:
                metric_vec[k] = self.metric(x_k, x_true)
                if self.verbose:
                    print(f"Metric at step {k}: {metric_vec[k]:0.4f}. Res: {tf.reduce_sum(tf.square(self.K(x_k) - y_delta)):.3f}, Reg: {tf.square(tf.norm(z, 'euclidean')):.3f}.")

            # Update z
            self.optimizer.apply_gradients(zip([gradient], [z]))

        if x_true is not None:
            return z, metric_vec
        return z