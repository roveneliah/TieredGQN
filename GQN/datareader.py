import tensorflow as tf
from data_reader import DataReader

root_path = '../../Datasets'
data_reader = DataReader(dataset='jaco', context_size=5, root=root_path)
data = data_reader.read(batch_size=12)

with tf.train.SingularMonitoredSession() as sess:
    print(data)
    # d = sess.run(data)
