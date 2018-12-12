import argparse
import keras
from data_reader import DataReader
from gqn import GQN
import numpy as np
# import utils

# Path for saving logs
logs_path =  './logs/tensorboard'
# print(args)

# fake data
q = np.random.rand(200,16,16,256)
r = np.random.rand(200,16,16,256)
target = np.random.rand(200,64,64,3)

model = GQN().generator
model.compile(optimizer = "Adam" , loss = "binary_crossentropy", metrics = ["accuracy"])
tensorboard = keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)
epoch_callback = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: saveModelToCloud(epoch, 3))
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

model.fit(
    x = [r,q],
    y = target,
    epochs = 10,
    verbose = 1,
    # batch_size=None,
    callbacks=[tensorboard, epoch_callback, checkpoint],
    steps_per_epoch=2,
    # validation_data=([test_frames, test_cameras, test_query], test_target),
    # validation_steps=1
)
print("done training")

# Save model.h5 on to google storage
model.save('model.h5')
with file_io.FileIO('model.h5', mode='r') as input_f:
    with file_io.FileIO(job_dir + 'model/model.h5', mode='w+') as output_f:
        output_f.write(input_f.read())
