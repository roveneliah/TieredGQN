#
#  Full end-to-end GQN model w/ encoder and decoder
#
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from training.data_reader import DataReader
from training.encoder import Encoder
from training.generator import Generator
from PIL import Image

class GQN:
    def __init__(self):
        context = keras.layers.Input(shape=(64,64,3), name="context")
        camera = keras.layers.Input(shape=(1,1,7), name="camera")
        q = keras.layers.Input(shape=(16,16,256), name="query")

        h00 = keras.layers.Input(tensor=tf.constant(np.zeros((1,16,16,256)), dtype='float32'), name="h00") # p26 of paper indicates this is initialized as 0
        c00 = keras.layers.Input(tensor=tf.constant(np.zeros((1,16,16,256)), dtype='float32'), name="c00")
        u00 = keras.layers.Input(tensor=tf.constant(np.zeros((1,64,64,256)), dtype='float32'), name="u00")

        r = Encoder()(inputs=[context, camera])
        tuner = Generator()(inputs=[q, r, h00, c00, u00])
        self.model = keras.Model(inputs=[context, camera, q, h00, c00, u00], outputs=tuner)

        self.inputs = [context, camera, q]
        self.decoders = [tuner]


# context = np.zeros((1,64,64,3))
# camera = np.zeros((1,1,1,7))
# q = np.zeros((1,16,16,256))
# gqn = GQN()
# # keras.utils.plot_model(gqn.model, to_file='gqn.png')
# img = gqn.model.predict([context, camera, q], verbose=1)
# img = img[0]
# img = Image.fromarray(img, 'RGB')
# img.show()
