from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

# Implement as a multilayer object instead of a layer itself
def GeneratorCell0(self, h0, v, r, z, c0, u0):
    # 1) concatenate h0, v, r, z
    input = keras.layers.Input(shape=()) # TODO: what is shape of input???

    # (2a) sig0 aka forget gate
    forget = keras.layers.Conv2D(
                    filters = ?,
                    kernel_size=(5,5),
                    strides=(1,1),
                    activation='sigmoid'
            )(input)

    # (2b) sig1 aka input gate
    inp_gate = keras.layers.Conv2D(
                filters = ?,
                kernel_size=(5,5),
                strides=(1,1),
                activation='sigmoid'
            )(input)

    # (2c) tanh aka canditates
    candiates = keras.layers.Conv2D(
                filters = ?,
                kernel_size=(5,5),
                strides=(1,1),
                activation='tanh'
            )(input)

    # (2d) sig2 aka output
    output = keras.layers.Conv2D(
                filters = ?,
                kernel_size=(5,5),
                strides=(1,1),
                activation='sigmoid'
            )(input)

    # (3) update context/state
    c = c0 * forget + inp_gate * candidates

    # (4) update output/h
    h = tanh(context) + output

    # (5) u = u0 + delta?(h) (kernel 4x4, stride 4x4)
    delta = keras.layers.Conv2D(
                filters = ?,
                kernel_size=(4,4),
                strides=(4,4)
            )(output)# delta?, and no activation???

    u = u0 + delta

    # return outputs for the next unit
    return h, c, u



def GeneratorCellLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim                    # specify output dimensions
        super(GeneratorCell, self).__init__(**kwargs)   # init with kwargs

    # Define weights
    def build(self, input_shape):
        # Create a trainable weight variable for this layer
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end


    # Logic of the layer: x is input tensor
    # NOTE: This is just brainstorming, syntax makes no sense
    #        More specifically, need to figure out how convolutional layers
    #         work in this LSTM context
    # 1) concatenate h0, v, r, z
    # 2a) sigmoid of hvrz (kernel 5x5, stride 1x1) = sig0
    # 2b) sigmoid of hvrz (kernel 5x5, stride 1x1) = sig1
    # 2c) tanh of hvrz (kernel 5x5, stride 1x1) = tan1
    # 2d) sigmoid of hvrz (kernel 5x5, stride 1x1) = sig2
    # 3) c = c0 + sig0 + sig1*tan1
    # 4) h = tanh(c) x sig2
    # 5) u = u0 + delta?(h) (kernel 4x4, stride 4x4)
    def call(self, h0, v, r, z, c, u0):
        # 1) concatenate h0, v, r, z
        input =

        # (2a) sig0 aka forget gate
        forget = keras.layers.Conv2D(
                        filters = ?,
                        kernel_size=(5,5),
                        strides=(1,1),
                        activation='sigmoid'
                )(input)

        # (2b) sig1 aka input gate
        inp_gate = keras.layers.Conv2D(
                    filters = ?,
                    kernel_size=(5,5),
                    strides=(1,1),
                    activation='sigmoid'
                )(input)

        # (2c) tanh aka canditates
        candiates = keras.layers.Conv2D(
                    filters = ?,
                    kernel_size=(5,5),
                    strides=(1,1),
                    activation='tanh'
                )(input)

        # (2d) sig2 aka output
        output = keras.layers.Conv2D(
                    filters = ?,
                    kernel_size=(5,5),
                    strides=(1,1),
                    activation='sigmoid'
                )(input)

        # (3) update context/state
        context = c * forget + inp_gate * candidates

        # (4) update output/h
        output = tanh(context) + output

        # (5) u = u0 + delta?(h) (kernel 4x4, stride 4x4)
        delta = keras.layers.Conv2D(
                    filters = ?,
                    kernel_size=(4,4),
                    strides=(4,4)
                )(output)# delta?, and no activation???
        u = u0 + delta

        return (output, context, u)

    # Specify the shape of transformation logic
    # Allows Keras to do shape inference
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
