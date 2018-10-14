import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
# from data_reader import DataReader

"""
Implementation of tower network from
"Neural scene representation and rendering"
"""
def Encoder():
    # input 1: 64x64x3
    frames = keras.layers.Input(shape=(64,64,3), name="frames")

    # conv1 kernel 2x2, stride 2x2
    # ==> 32x32x256
    # W = (Wo - kernel_w + 2padding)/stride_w + 1
    # H = (Ho - kernel_h + 2padding)/stride_h + 1
    # D = number of filters
    conv1 = keras.layers.Conv2D(
                filters=256,
                kernel_size=(2,2),
                strides=(2, 2),
                padding='valid',
                activation='relu'
            )(frames)

    # conv2 kernel 3x3, stride 1x1
    #  ===> 32x32x128
    # 128 filters, but shouldn't the W and H be 30?  Why isn't padding stated??  Should I just assume padding?
    skip_conv2 = keras.layers.Conv2D(filters = 128, kernel_size = (3,3), strides = (1,1), padding='same')(conv1)
    conv2 = keras.layers.Conv2D(filters = 128, kernel_size = (3,3), strides = (1,1), activation='relu', padding='same')(conv1)

    # Residual connection between conv1 and conv1
    conv2r = keras.layers.add([skip_conv2, conv2])

    # conv3 kernel 2x2, stride 2x2
    #  ====> 16x16x256 (256 filters)
    conv3 = keras.layers.Conv2D(filters = 256, kernel_size = (2,2), strides = (2,2), activation='relu', padding='same')(conv2)

    # input 2: 1x1x7
    cameras = keras.layers.Input(shape=(1,1,7), name="camera")
    # reshape input to 16x16x7
    b1 = keras.layers.Lambda(lambda x: K.repeat_elements(x, 16, 1))(cameras)
    b2 = keras.layers.Lambda(lambda x: K.repeat_elements(x, 16, 2))(b1)

    # concatenate b2 onto conv3
    conv3a = keras.layers.concatenate([conv3, b2])

    # conv4 kernel 3x3, stride 1x1
    # ---> 16x16x128filters
    skip_conv4 = keras.layers.Conv2D(filters = 128, kernel_size = (3,3), strides = (1,1), padding='same')(conv3a)
    conv4 = keras.layers.Conv2D(filters = 128, kernel_size = (3,3), strides = (1,1), activation='relu', padding='same')(conv3a)

    # Residual connection
    conv4r = keras.layers.add([skip_conv4, conv4])

    # conc kernel 1x1, stride 1x1
    # ---> 16x16x128 (same as before)
    conv5 = keras.layers.Conv2D(filters = 256, kernel_size = (3,3), strides = (1,1), activation='relu', padding='same')(conv4r)

    # THE RESULT OF THIS LAYER IS THE STATE REPRESENTATION
    dr = keras.layers.Conv2D(filters = 256, kernel_size = (1,1), strides=(1,1), activation='relu')(conv5)

    # Define the model
    model = keras.Model(inputs=[frames, cameras], outputs=dr)
    return model

# check if model outputs
# inV = np.zeros((1,64,64,3))
# inP = np.zeros((1,1,1,7))
#
# root_path = '../../Datasets'
# data_reader = DataReader(dataset='jaco', context_size=5, root=root_path)
# data = data_reader.read(batch_size=12)
#
# model = Encoder()
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit([data.query.context.frames, data.query.context.cameras])  # starts training
# sess = tf.Session()
# K.set_session(sess)
# with sess.as_default():
#     print(model.predict([data.query.context.frames[0].eval(session=sess), data.query.context.cameras[0].eval(session=sess)], verbose=1))
# keras.utils.plot_model(model, to_file='representation_net.png')
