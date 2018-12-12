import argparse
import keras
from data_reader import DataReader
from encoder import Encoder
import numpy as np

# Path for saving logs
logs_path =  './logs/tensorboard'
# print(args)

# load data
# TODO: read from cloud bucket?
root_path = 'gs://gqn-dataset/'
data_reader = DataReader(dataset='rooms_free_camera_no_object_rotations', context_size=5, root=root_path)
data = data_reader.read(batch_size=4)
frames = data.query.context.frames
cameras = data.query.context.cameras
query = data.query.query_camera
target = data.target

# https://www.youtube.com/watch?v=VxnHf-FfWKY

# preprocess data as needed
# initialize model
model = Encoder()
# compile model
# TODO: compilation params
model.compile(optimizer = "Adam" , loss = "binary_crossentropy", metrics = ["accuracy"])
# print model summary
model.summary()
# add callbacks for tensorboard and history??
tensorboard = keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)
# fit model
model.fit(
    x = [frames, cameras],
    y = np.zeros((20,16,16,256)),
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
