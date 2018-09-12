import keras
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

# Individual ConvLSTM cell as described on p38 of paper
# NOTE: I include the conv of h into z in the cell instead of passing conv'd input
class SkipConvLSTM:
    def __init__(self):
        # Cell Inputs
        vq = keras.layers.Input(shape=(16,16,256), name="vq")
        r = keras.layers.Input(shape=(16,16,256), name="r")

        # INIT CONVLSTM CELL VALUES
        h1= keras.layers.Input(shape=(16,16,256), name="h1") # p26 of paper indicates this is initialized as 0
        c0 = keras.layers.Input(shape=(16,16,256), name="c0")
        u0 = keras.layers.Input(shape=(4,4,256), name="u0")

        print("hi")
        # (0) get z from h0
        # TODO: IS IT JUST CONV'D ON THE FIRST LAYERS LIKE P38 SUGGESTS?
        z = keras.layers.Conv2D(
                        filters = 256,
                        kernel_size = (5,5),
                        strides = (1,1),
                        padding = 'same'
                        # what is activation?
                    )(h1)

        # (1) concatenate h0, v, r, z
        concat = keras.layers.concatenate([h1, vq, r, z], axis=0)

        # (2a) sig0 aka forget gate
        forget = keras.layers.Conv2D(
                        filters = 256, # hmm
                        kernel_size=(5,5),
                        strides=(1,1),
                        activation='sigmoid',
                        padding="same"
                )(concat)

        # (2b) sig1 aka input gate
        inp_gate = keras.layers.Conv2D(
                    filters = 256, # hmm
                    kernel_size=(5,5),
                    strides=(1,1),
                    activation='sigmoid',
                    padding='same'
                )(concat)

        # (2c) tanh aka canditates
        candidates = keras.layers.Conv2D(
                    filters = 256, # hmm
                    kernel_size=(5,5),
                    strides=(1,1),
                    activation='tanh',
                    padding='same'
                )(concat)

        # (2d) sig2 aka output
        output = keras.layers.Conv2D(
                    filters = 256, # hmm
                    kernel_size=(5,5),
                    strides=(1,1),
                    activation='sigmoid',
                    padding='same'
                )(concat)

        # (3) update context/state
        tmp1 = keras.layers.multiply([c0, forget])
        tmp2 = keras.layers.multiply([inp_gate, candidates])
        c = keras.layers.add([keras.layers.multiply([c0, forget]), keras.layers.multiply([inp_gate, candidates])])

        # (4) update output/h
        h = keras.layers.add([keras.layers.Activation('tanh')(c), output])

        # (5) u = u0 + delta?(h) (kernel 4x4, stride 4x4)
        delta = keras.layers.Conv2D( # TODO: should be transpose
                    filters = 256,
                    kernel_size=(4,4),
                    strides=(4,4)
                )(output) # what is delta symbol?, and no ACTIVATION???

        u = keras.layers.add([u0, delta])

        self.model = keras.Model(inputs=[vq, r, h1, c0, u0], outputs=[h, c, u])
