import keras
import numpy as np
import tensorflow as tf

### NOTE: get working with just ONE ConvLSTM2D
# return model
def GeneratorCell():
    # Input is (r, v)
    # r = 14x14x256
    h0 = keras.layers.Input(shape=(16,16,256), name="h0")
    c0 = keras.layers.Input(shape=(16,16,256), name="c0")
    u0 = keras.layers.Input(shape=(4,4,256), name="u0")
    vq = keras.layers.Input(shape=(16,16,256), name="vq")
    r = keras.layers.Input(shape=(16,16,256), name="r")
    z = keras.layers.Input(shape=(16,16,256), name="z")

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
    print(forget)

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
    delta = keras.layers.Conv2D(
                filters = 256,
                kernel_size=(4,4),
                strides=(4,4),
                padding='valid'
            )(output) # what is delta symbol?, and no ACTIVATION???

    u = keras.layers.add([u0, delta])
    model = keras.Model(inputs=[h0, c0, u0, vq, r, z], outputs=[h, c, u])
    return model

inp = np.zeros((1,16,16,256))
model = GeneratorCell()
print(model.predict([inp, inp, np.zeros((1,4,4,256)), inp, inp, inp], verbose=1))
keras.utils.plot_model(model, to_file='generator.png')
