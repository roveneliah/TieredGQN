import keras
import numpy as np
import tensorflow as tf
from skipConvLSTM import SkipConvLSTM

# Generator model takes the state representation r
#  and the _____ vq as inputs
#  and returns u, the ___________
def Generator():
    # INPUTS
    vq = keras.layers.Input(shape=(16,16,256), name="vq")
    r = keras.layers.Input(shape=(16,16,256), name="r")

    # INIT CONVLSTM CELL VALUES
    h0 = keras.layers.Input(tensor=tf.constant(np.zeros((1,16,16,256)), dtype='float32'), name="h0") # p26 of paper indicates this is initialized as 0
    c0 = keras.layers.Input(tensor=tf.constant(np.zeros((1,16,16,256)), dtype='float32'), name="c0")
    u0 = keras.layers.Input(tensor=tf.constant(np.zeros((1,4,4,256)), dtype='float32'), name="u0")


    # TODO: How many cells in paper????
    cell0 = GeneratorCell()([vq, r, h0, c0, u0])
    print(cell0)
    # cell1 = GeneratorCell()(inputs=[vq, r] + cell0)
    # cell2 = GeneratorCell()(inputs=[vq, r] + cell1)

    # last conv on u (kernel 1x1, stride 1x1)
    # WHAT IS THIS OUTPUT ?
    # something = keras.layers.Conv2D(
    #                         filters = 256, # ???
    #                         kernel_size = (1,1),
    #                         strides = (1,1)
    #                         # activation?
    #                     )(cell0[2])

    model = keras.Model(inputs=[vq, r], outputs=cell0)
    return model



# Individual ConvLSTM cell as described on p38 of paper
# NOTE: I include the conv of h into z in the cell instead of passing conv'd input
def GeneratorCell():
    # Cell Inputs
    vq = keras.layers.Input(shape=(16,16,256), name="vq")
    r = keras.layers.Input(shape=(16,16,256), name="r")
    h0 = keras.layers.Input(shape=(16,16,256), name="h_i") # p26 of paper indicates this is initialized as 0
    c0 = keras.layers.Input(shape=(16,16,256), name="c_i")
    u0 = keras.layers.Input(shape=(4,4,256), name="u_i")

    print("hi")
    # (0) get z from h0
    # TODO: IS IT JUST CONV'D ON THE FIRST LAYERS LIKE P38 SUGGESTS?
    z = keras.layers.Conv2D(
                    filters = 256,
                    kernel_size = (5,5),
                    strides = (1,1),
                    padding = 'same'
                    # what is activation?
                )(h0)

    # (1) concatenate h0, v, r, z
    concat = keras.layers.concatenate([h0, vq, r, z], axis=0)

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
    # tmp1 = keras.layers.multiply([c0, forget])
    # tmp2 = keras.layers.multiply([inp_gate, candidates])
    # c = keras.layers.add([tmp1, tmp2])
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

    return keras.Model(inputs=[vq, r, h0, c0, u0], outputs=c)


inp = np.zeros((1,16,16,256))
model = Generator()([inp,inp])
print(model)
print(model.predict([inp, inp], verbose=1))
keras.utils.plot_model(model, to_file='generator1.png')
