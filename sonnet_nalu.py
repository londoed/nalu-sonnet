#################################################################
########## IMPLEMENTATION OF NALU IN DEEPMIND'S SONNET ##########
#################################################################

import tensorflow as tf
import sonnet as snt

class NAC(snt.Module):
    """Neural accumulator cell.

    The implementation is based on https://arxiv.org/abs/1808.00508.

    Attributes:
        input_shape: Number describing the shape of the input tensor.
        num_outputs: Integer describing the number of outputs.

    """
    def __init__(self, input_shape, num_outputs, name=None):
        super(NAC, self).__init__(name=name)
        self.num_outputs = num_outputs

    @snt.once
    def _initialize(self, input_shape):
        shape = [int(input_shape[-1]), self.num_outputs]
        self.W_hat = tf.Variable("W_hat", shape=shape, initializer=tf.initialziers.GlorotUniform())
        self.M_hat = tf.Variable("M_hat", shape=shape, initializer=tf.initializers.GlorotUniform())

    def __call__(self, x):
        self._initialize(x)
        W = tf.nn.tanh(self.W_hat) * tf.nn.sigmoid(self.M_hat)
        return tf.matmul(x, tf.cast(W, 'float64'))

class NALU(snt.AbstractModule):
    """Neural arithmetic logic unit.

    The implementation is based on https://arxiv.org/abs/1808.00508.

    Attributes:
        input_shape: Number describing the shape of the input tensor.
        num_outputs: Integer describing the number of outputs.

    """
    def __init__(self, input_shape, num_outputs, name='nalu'):
        super(NALU, self).__init__(name=name)
        self.num_outputs = num_outputs
        self.nac = NAC(input_shape, self.num_outputs)
        self.eps = 1e-7

    @snt.once
    def _initialize(self, input_shape):
        shape = [int(input_shape[-1]), self.num_outputs]
        self.G = tf.Variable("G", shape=shape, initializer=tf.initializers.GlorotUniform())

    def __call__(self, x):
        self._initialize(x)
        g = tf.nn.sigmoid(tf.matmul(x, self.G))
        y1 = g * self.nac(x)
        y2 = (1 - g) * tf.exp(self.nac(tf.log(tf.abs(x) + self.eps)))
        return y1 + y2
