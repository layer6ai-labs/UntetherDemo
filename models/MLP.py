from tensorflow.keras import layers, models, initializers


class MLP(models.Model):

    def __init__(self, num_inputs, num_layers, num_dims, num_outputs, dropout_rate):
        super(MLP, self).__init__()
        model = models.Sequential()
        for n_layer in range(num_layers):
            if n_layer == 0:
                model.add(layers.Dense(num_dims, input_shape=(num_inputs,),
                                       kernel_initializer=initializers.RandomNormal(0, 0.01),
                                       bias_initializer=initializers.RandomNormal(0, 0.000001)))
            else:
                model.add(layers.Dense(num_dims, kernel_initializer=initializers.RandomNormal(0, 0.01),
                                       bias_initializer=initializers.RandomNormal(0, 0.000001)))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.Activation('relu'))

        model.add(layers.Dense(num_outputs, kernel_initializer=initializers.RandomNormal(0, 0.01),
                               bias_initializer=initializers.RandomNormal(0, 0.000001)))

        self.model = model

    def call(self, x):
        x = self.model(x)

        return x
