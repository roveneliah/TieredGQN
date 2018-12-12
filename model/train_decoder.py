import argparse
import keras
from data_reader import DataReader
from generator import Generator
import numpy as np
import tensorflow as tf

# Path for saving logs
logs_path =  './logs/tensorboard'
# print(args)

# load data
# TODO: read from cloud bucket?
root_path = 'gs://gqn-dataset/'
data_reader = DataReader(dataset='rooms_free_camera_no_object_rotations', context_size=5, root=root_path)
batch_size = 20
data = data_reader.read(batch_size=batch_size)
frames = data.query.context.frames
cameras = data.query.context.cameras
query = data.query.query_camera
target = data.target

# https://www.youtube.com/watch?v=VxnHf-FfWKY

# preprocess data as needed
# initialize model
model = Generator()
# compile model
# TODO: compilation params
model.compile(optimizer = "Adam" , loss = "binary_crossentropy", metrics = ["accuracy"])
# print model summary
model.summary()
# add callbacks for tensorboard and history??
tensorboard = keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)
# fit model
h00 = keras.layers.Input(tensor=tf.constant(np.zeros((batch_size,16,16,256)), dtype='float32'), name="h00") # p26 of paper indicates this is initialized as 0
c00 = keras.layers.Input(tensor=tf.constant(np.zeros((batch_size,16,16,256)), dtype='float32'), name="c00")
u00 = keras.layers.Input(tensor=tf.constant(np.zeros((batch_size,64,64,256)), dtype='float32'), name="u00")
model.fit(
    x = [query, np.zeros((batch_size,16,16,256)), h00, c00, u00],
    y = np.zeros((batch_size,64,64,3)),
    epochs = 4,
    verbose = 1,
    steps_per_epoch=10,
    callbacks=[tensorboard]
    # validation_data=(eval_data,eval_labels)
)

# Save model.h5 on to google storage
model.save('encoder.h5')
with file_io.FileIO('encoder.h5', mode='r') as input_f:
    with file_io.FileIO(job_dir + 'model/encoder.h5', mode='w+') as output_f:
        output_f.write(input_f.read())
