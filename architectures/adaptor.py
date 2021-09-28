#  Copyright 2019 Gabriele Valvano
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import tensorflow as tf
from tensorflow import layers
from idas.tf_utils import get_shape


# He initializer for the layers with ReLU activation function:
he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
b_init = tf.zeros_initializer()


class Adaptor(object):

    def __init__(self, n_filters=16, n_layers=3, trainable=True, name='Adaptor'):
        """
        Class for building the image adaptor. For additional details, refer to:
            Karani, Neerav, et al. "Test-time adaptable neural networks for robust medical image segmentation."
            Medical Image Analysis 68 (2020): 101907

        :param n_filters: (int) number of filters at the first convolutional layer
        :param n_layers: (int) number of convolutional layer
        :param name: (str) name for the variable scope

        - - - - - - - - - - - - - - - -
        Notice that:
          - output is linear
        - - - - - - - - - - - - - - - -
        """
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.name = name
        self.trainable = trainable

        self.prediction = None

    @staticmethod
    def activation(incoming, sigma, eps=1e-16):
        return tf.exp(-tf.math.square(incoming) / (tf.math.square(sigma) + eps))

    def build(self, incoming, reuse=tf.AUTO_REUSE):
        """
        Build the model.
        :param incoming: (list) list of input tensors (for each block)
        :param reuse: (bool) if True, reuse trained weights
        """
        n_out = get_shape(incoming)[-1]

        with tf.variable_scope(self.name, reuse=reuse):
            _incoming = incoming
            for i in range(self.n_layers - 1):
                conv = layers.conv2d(_incoming, filters=self.n_filters, kernel_size=3, strides=1, padding='same',
                                     trainable=self.trainable, kernel_initializer=he_init, bias_initializer=b_init,
                                     name='conv_{0}'.format(i))

                init_value = tf.random_normal([1, 1, 1, self.n_filters], mean=0.2, stddev=0.05)
                sigma = tf.Variable(initial_value=init_value, name='sigma_{0}'.format(i))
                conv_act = self.activation(conv, sigma)
                _incoming = conv_act

            pred = layers.conv2d(conv_act, filters=n_out, kernel_size=3, strides=1, padding='same',
                                 trainable=self.trainable, kernel_initializer=he_init, bias_initializer=b_init,
                                 name='conv_{0}'.format(i + 1))
            init_value = tf.random_normal([1, 1, 1, n_out], mean=0.2, stddev=0.05)
            sigma = tf.Variable(initial_value=init_value, name='sigma_{0}'.format(i + 1))

            # activation + residual connection:
            self.prediction = self.activation(pred, sigma) + incoming

        return self

    def get_prediction(self):
        """ Get discriminator prediction. """
        return self.prediction
