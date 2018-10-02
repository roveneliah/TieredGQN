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
from encoder_class import EncoderClass
# from PIL import Image

class GQN:
    # If using numpy arrays, set tfrecords to false
    def __init__(self):
        frames = keras.layers.Input(shape=(None,64,64,3), name="frames")
        camera = keras.layers.Input(shape=(None,1,1,7), name="camera")
        q = keras.layers.Input(shape=(16,16,256), name="query")

        # INITIAL CONVLSTM CELL VALS
        h00 = keras.layers.Input(tensor=tf.constant(np.zeros((1,16,16,256)), dtype='float32'), name="h00") # p26 of paper indicates this is initialized as 0
        c00 = keras.layers.Input(tensor=tf.constant(np.zeros((1,16,16,256)), dtype='float32'), name="c00")
        u00 = keras.layers.Input(tensor=tf.constant(np.zeros((1,64,64,256)), dtype='float32'), name="u00")

        # TODO: Create an encoder for each context input
        encoders = [Encoder()(inputs=[frames[:][i], camera[:][i]]) for i in range()]
        r = keras.layers.add(encoders)
        tuner = Generator()(inputs=[q, r, h00, c00, u00])

        self.model = keras.Model(inputs=[frames, camera, q, h00, c00, u00], outputs=tuner)

        self.inputs = [frames, camera, q]
        self.decoders = [tuner]



class GQN_TFRecord:
    # If using numpy arrays, set tfrecords to false
    def __init__(self, frames, camera, query, target):
        ## GCLOUD PARAMS ##
        self.logs_path =  './logs/tensorboard'

        # if using tfrecords, tell input layers to expect these tensors
        frames = keras.layers.Input(tensor=frames, shape=(1,64,64,3), name="frames")
        camera = keras.layers.Input(tensor=camera, shape=(1,1,1,7), name="camera")
        q = keras.layers.Input(tensor=query, shape=(1,16,16,256), name="query")

        # target output to be used by fit method
        self.target = target

        # INITIAL CONVLSTM CELL VALS
        h00 = keras.layers.Input(tensor=tf.constant(np.zeros((1,16,16,256)), dtype='float32'), name="h00") # p26 of paper indicates this is initialized as 0
        c00 = keras.layers.Input(tensor=tf.constant(np.zeros((1,16,16,256)), dtype='float32'), name="c00")
        u00 = keras.layers.Input(tensor=tf.constant(np.zeros((1,64,64,256)), dtype='float32'), name="u00")

        r = EncoderClass(frames, camera)
        tuner = Generator()(inputs=[q, r, h00, c00, u00])

        self.model = keras.Model(inputs=[frames, camera, q, h00, c00, u00], outputs=tuner)

        self.inputs = [frames, camera, q]
        self.decoders = [tuner]

        # CALLBACKS
        tensorboard = keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)
        epoch_callback = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: saveModelToCloud(epoch, 3))
        filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks = [tensorboard, epoch_callback, checkpoint]



    def fit():
        # set fit params
        epochs = 10
        verbose = 1
        steps_per_epoch = 2

        # fit model with tfrecords
        self.model.fit(
            # x is already known from input layers
            y = this.target,
            epochs = epochs,
            verbose = verbose,
            callbacks=self.callbacks,
            steps_per_epoch=steps_per_epoch,
            # validation_data=([test_frames, test_cameras, test_query], test_target),
            # validation_steps=1
        )

        # Save model.h5 on to google storage
        self.model.save('model.h5')
        with file_io.FileIO('model.h5', mode='r') as input_f:
            with file_io.FileIO(job_dir + 'model/model.h5', mode='w+') as output_f:
                output_f.write(input_f.read())

frames = np.zeros((2,64,64,3))
camera = np.zeros((2,1,1,7))
q = np.zeros((1,16,16,256))
gqn = GQN(2)
# keras.utils.plot_model(gqn.model, to_file='gqn.png')
img = gqn.model.predict([frames, camera, q], verbose=1)
img = img[0]
img = Image.fromarray(img, 'RGB')
img.show()
