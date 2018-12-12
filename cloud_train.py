import argparse
import keras
from training.data_reader import DataReader
from training.gqn import GQN
from time import time

def main(job_dir, **args):
    print('in main')
    # Path for saving logs
    logs_path = job_dir + 'logs/{}'.format(time())
    # print(args)

    # load data
    # TODO: read from cloud bucket?
    root_path = 'gs://gqn-dataset/'
    data_reader = DataReader(dataset='rooms_free_camera_no_object_rotations', context_size=5, root=root_path)
    test_reader = DataReader(dataset='rooms_free_camera_no_object_rotations', context_size=5, root=root_path, mode='test')
    data = data_reader.read(batch_size=80)
    test_data = test_reader.read(batch_size=20)

    frames = data.query.context.frames
    cameras = data.query.context.cameras
    query = data.query.query_camera
    target = data.target

    test_frames = test_data.query.context.frames
    test_cameras = test_data.query.context.cameras
    test_query = test_data.query.query_camera
    test_target = test_data.target

    # https://www.youtube.com/watch?v=VxnHf-FfWKY

    # preprocess data as needed
    # initialize model
    model = GQN().model
    # compile model
    # TODO: compilation params
    model.compile(optimizer = "Adam" , loss = "binary_crossentropy", metrics = ["accuracy"])
    # print model summary
    model.summary()
    # add callbacks for tensorboard and history??
    tensorboard = keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)
    epoch_callback = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: saveModelToCloud(epoch, 3))
    # fit model
    print("started fitting")
    model.fit(
        x = [frames, cameras, query],
        y = target,
        epochs = 1,
        verbose = 1,
        batch_size=None,
        callbacks=[tensorboard, epoch_callback],
        steps_per_epoch=10,
        validation_data=([test_frames, test_cameras, test_query], test_target),
        validation_steps=10
    )
    print("done fitting")

    # Save model.h5 on to google storage
    model.save('model.h5')
    saveModelToCloud
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + 'model/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())


# Run the app
if __name__ == "__main__":
    print(0)
    parser = argparse.ArgumentParser()
    print(1)

    # parser.add_argument(
    #     '--train_files',
    #     help = 'GCS path to training data',
    #     required = True
    # )
    #
    # parser.add_argument(
    #     '--eval_files',
    #     help = 'GCS path to evaluate data',
    #     required = True
    # )

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    print(2)
    args = parser.parse_args()
    print(3)
    arguments = args.__dict__

    print(arguments)

    main(**arguments)
