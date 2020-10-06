# import the necessary packages
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


class Simplenn:
    @staticmethod
    def build(input_shape, n_classes, n_layers, activation_type):
        # define the architecture using Keras
        model = Sequential()
        first = True
        for i in range(len(n_layers)):
            if first:
                model.add(Dense(n_layers[i], input_shape=input_shape, activation=activation_type['input']))
                first = False
            else:
                model.add(Dense(n_layers[i], activation=activation_type['hidden']))

        model.add(Dense(n_classes, activation=activation_type['output']))

        # return the constructed network architecture
        return model
