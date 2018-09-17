import keras
import numpy as np
import tensorflow as tf
from keras import backend as K

def Generator():
    # INPUTS
    vq = keras.layers.Input(shape=(16,16,256), name="vq")
    r = keras.layers.Input(shape=(16,16,256), name="r")

    # INIT CONVLSTM CELL VALUES
    h0 = keras.layers.Input(tensor=tf.constant(np.zeros((1,16,16,256)), dtype='float32'), name="h0") # p26 of paper indicates this is initialized as 0
    c0 = keras.layers.Input(tensor=tf.constant(np.zeros((1,16,16,256)), dtype='float32'), name="c0")
    u0 = keras.layers.Input(tensor=tf.constant(np.zeros((1,64,64,256)), dtype='float32'), name="u0")

    # TODO: How many cells in paper????
    # TODO: LOOP?
    cell0 = GeneratorCell()([vq, r, h0, c0, u0])
    cell1 = GeneratorCell()(inputs=[vq,r]+cell0)
    cell2 = GeneratorCell()(inputs=[vq, r] + cell1)

    # last conv on u (kernel 1x1, stride 1x1)
    # WHAT IS THIS OUTPUT ?
    something = keras.layers.Conv2D(
                            filters = 256, # ???
                            kernel_size = (1,1),
                            strides = (1,1)
                            # activation?
                        )(cell2[2])

    model = keras.Model(inputs=[vq, r, h0, c0, u0], outputs=something)
    return model



# Individual ConvLSTM cell as described on p38 of paper
# NOTE: I include the conv of h into z in the cell instead of passing conv'd input
def GeneratorCell():
    # Cell Inputs
    vq = keras.layers.Input(shape=(16,16,256), name="vq")
    r = keras.layers.Input(shape=(16,16,256), name="r")
    h0 = keras.layers.Input(shape=(16,16,256), name="h_i") # p26 of paper indicates this is initialized as 0
    c0 = keras.layers.Input(shape=(16,16,256), name="c_i")
    u0 = keras.layers.Input(shape=(64,64,256), name="u_i")

    # (0) get z from h0
    z = keras.layers.Conv2D(
                    filters = 256,
                    kernel_size = (5,5),
                    strides = (1,1),
                    padding = 'same'
                    # what is activation?
                )(h0)

    # (1) concatenate h0, v, r, z
    concat = keras.layers.concatenate([h0, vq, r, z])

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
    c = keras.layers.add([keras.layers.multiply([c0, forget]), keras.layers.multiply([inp_gate, candidates])])

    # (4) update output/h
    h = keras.layers.add([keras.layers.Activation('tanh')(c), output])

    # (5) u = u0 + delta?(h) (kernel 4x4, stride 4x4)
    delta = keras.layers.Conv2DTranspose( # TODO: should be transpose
                filters = 256,
                kernel_size=(4,4),
                strides=(4,4)
            )(output) # what is delta symbol?, and no ACTIVATION???

    u = keras.layers.add([u0, delta])

    return keras.Model(inputs=[vq, r, h0, c0, u0], outputs=[h, c, u])


inp = np.zeros((1,16,16,256))
model = Generator()
keras.utils.plot_model(model, to_file='generator_net.png')
print(model.predict([inp, inp], verbose=1))
