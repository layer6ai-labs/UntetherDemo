import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, initializers, regularizers
from constants import Constants

class ResidualBlock(models.Model):

    def __init__(self, dilation_rate, nb_filters, kernel_size, padding, dropout_rate=0.0, conv_regularization=0.08):
        """
        Defines the residual block for TCN
        :param x: The previous layer in the model
        :param dilation_rate: The dilation rate for this residual block
        :param nb_filters: The number of convolutional filters to use in this block
        :param kernel_size: The size of the convolutional kernel
        :param padding: The padding used in the convolutional layers, 'same' or 'causal'.
        :param dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        :param conv_regularization: L2 regularization coefficient.
        :return: A tuple where the first element is the residual model layer, and the second is the skip connection.
        """
        super(ResidualBlock, self).__init__()

        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.conv_regularization = conv_regularization

        self.con1 = layers.SeparableConv1D(filters=self.nb_filters,
                                           kernel_size=self.kernel_size,
                                           dilation_rate=self.dilation_rate,
                                           padding=self.padding,
                                           kernel_regularizer=regularizers.l2(conv_regularization),
                                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01))
        self.dropout1 = layers.SpatialDropout1D(self.dropout_rate)
        self.con2 = layers.SeparableConv1D(filters=self.nb_filters,
                                           kernel_size=self.kernel_size,
                                           dilation_rate=self.dilation_rate,
                                           padding=self.padding,
                                           kernel_regularizer=regularizers.l2(conv_regularization),
                                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01))
        self.dropout2 = layers.SpatialDropout1D(self.dropout_rate)
        self.conv_matching = layers.Conv1D(filters=nb_filters, kernel_size=1, padding='same',
                                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01))

    def call(self, x):
        prev_x = x

        x = self.con1(x)
        x = tf.nn.relu(x)
        x = self.dropout1(x)

        x = self.con2(x)
        x = tf.nn.relu(x)
        x = self.dropout2(x)

        if prev_x.shape[-1] != x.shape[-1]:
            prev_x = self.conv_matching(prev_x)

        x = x + prev_x
        x = tf.nn.relu(x)
        return x


class TCN(models.Model):

    def __init__(self, nb_filters=64, kernel_size=3, nb_stacks=1, dilations=None, padding='same',
                 use_skip_connections=False, dropout_rate=0.0, conv_regularization=0.08):

        """Creates a TCN layer.
               Input shape:
                   A tensor of shape (batch size, time steps, features).
               Args:
                   nb_filters: The number of filters to use in the convolutional layers.
                   kernel_size: The size of the kernel to use in each convolutional layer.
                   nb_stacks : The number of stacks of residual blocks to use.
                   dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
                   padding: The padding to use in the convolutional layers, 'causal' or 'same'.
                   use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
                   dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
                   conv_regularization: Float, L2 regularization coefficient for conv layers.
               Returns:
                   A TCN layer.
           """
        super(TCN, self).__init__()
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding
        self.conv_regularization = conv_regularization

        self.tcn_blocks = []

        for s in range(self.nb_stacks):
            for d in self.dilations:
                self.tcn_blocks.append(
                    ResidualBlock(
                        dilation_rate=d,
                        nb_filters=self.nb_filters,
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                        dropout_rate=self.dropout_rate,
                        conv_regularization=self.conv_regularization)
                )
        self.fc1 = layers.Dense(self.nb_filters // 2, kernel_initializer=initializers.RandomNormal(0, 0.01))
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.fc2 = layers.Dense(self.nb_filters // 4, kernel_initializer=initializers.RandomNormal(0, 0.01),
                                bias_initializer=initializers.RandomNormal(0, 0.000001))
        self.dropout2 = layers.Dropout(self.dropout_rate)
        self.reshape = layers.Reshape([Constants._TCN_LENGTH * self.nb_filters // 4])
        self.fc = layers.Dense(Constants._NUM_TARGETS, kernel_initializer=initializers.RandomNormal(0, 0.01),
                                bias_initializer=initializers.RandomNormal(0, 0.000001))

    def call(self, x):

        for tcn_blk in self.tcn_blocks:
            x = tcn_blk(x)

        x = self.fc1(x)
        x = self.dropout1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = tf.nn.relu(x)
        x = self.reshape(x)
        x = self.fc(x)

        return x
