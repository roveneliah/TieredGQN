import argparse
import keras
from training.data_reader import DataReader
from training.gqn import GQN

def main(job_dir, **args):
    print('in main')
    # Path for saving logs
    logs_path = job_dir + 'logs/tensorboard'
    # print(args)

    # load data
    # TODO: read from cloud bucket?
    root_path = 'gs://gqn-dataset/'
    data_reader = DataReader(dataset='rooms_free_camera_no_object_rotations', context_size=5, root=root_path)
    data = data_reader.read(batch_size=20)
    print(data.query)
    print(data.query.context)
    print(data.target)
    frames = data.query.context.frames
    cameras = data.query.context.cameras
    query = data.query.query_camera
    target = data.target

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
    # fit model
    model.fit(
        x = [frames, cameras, query],
        y = target,
        epochs = 4,
        verbose = 1,
        batch_size=100,
        callbacks=[tensorboard]
        # validation_data=(eval_data,eval_labels)
    )

    # Save model.h5 on to google storage
    model.save('model.h5')
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + 'model/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())


# Run the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    print(0)
    args = parser.parse_args()
    print(1)
    arguments = args.__dict__
    print(2)

    print(arguments)
    print(3)
    main(**arguments)
    main(".")
