#
#  Full end-to-end GQN model w/ encoder and decoder
#
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
# from training.data_reader import DataReader
# from training.encoder import Encoder
# from training.generator import Generator
from data_reader import DataReader
from encoder import Encoder
from generator import Generator
from PIL import Image

class GQN:
    def __init__(self, samples): # use samples in dynamic definition
        frames = keras.layers.Input(shape=(64,64,3), name="frames")
        cameras = keras.layers.Input(shape=(1,1,7), name="cameras")
        q = keras.layers.Input(shape=(16,16,256), name="query")

        # INITIAL CONVLSTM CELL VALS
        h00 = keras.layers.Input(tensor=tf.constant(np.zeros((1,16,16,256)), dtype='float32'), name="h00") # p26 of paper indicates this is initialized as 0
        c00 = keras.layers.Input(tensor=tf.constant(np.zeros((1,16,16,256)), dtype='float32'), name="c00")
        u00 = keras.layers.Input(tensor=tf.constant(np.zeros((1,64,64,256)), dtype='float32'), name="u00")

        r = Encoder()(inputs=[frames, cameras])
        target = Generator()(inputs=[q, r, h00, c00, u00])
        self.model = keras.Model(inputs=[frames, cameras, q, h00, c00, u00], outputs=target)
        # self.encoder = keras.Model(inputs=[r0, frame, camera, q, h00, c00, u00], outputs=r)



def testGQN():
    frame = np.random.rand(1,64,64,3)
    camera = np.random.rand(1,1,1,7)
    q = np.random.rand(1,16,16,256)
    r = np.random.rand(1,16,16,256)
    gqn = GQN().model
    img = gqn.predict([frame, camera, q, r])[0]
    Image.fromarray(img, 'RGB').show()

def testGenerator():
    q = np.random.rand(1,16,16,256)
    r = np.random.rand(1,16,16,256)
    gqn = GQN().generator
    img = gqn.predict([q,r])[0]
    Image.fromarray(img, 'RGB').show()

# testGQN()
