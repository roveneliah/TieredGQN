import keras
import numpy as np
import tensorflow as tf
"""
Implementation of tower network from
"Neural scene representation and rendering"
"""
# input 1: 64x64x3
input_vis = keras.layers.Input(shape=(64,64,3), name="vis")

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
        )(input_vis)


# conv2 kernel 3x3, stride 1x1
#  ===> 32x32x128
# 128 filters, but shouldn't the W and H be 30?  Why isn't padding stated??  Should I just assume padding?
skip_conv2 = keras.layers.Conv2D(filters = 128, kernel_size = (3,3), strides = (1,1), padding='same')(conv1)
conv2 = keras.layers.Conv2D(filters = 128, kernel_size = (3,3), strides = (1,1), activation='relu', padding='same')(conv1)

# Residual connection between conv1 and conv1
# TODO: is this right? other implementations have sizing issues...
conv2r = keras.layers.add([skip_conv2, conv2])

# conv3 kernel 2x2, stride 2x2
#  ====> 16x16x256 (256 filters)
conv3 = keras.layers.Conv2D(filters = 256, kernel_size = (2,2), strides = (2,2), activation='relu', padding='same')(conv2)

# input 2: 1x1x7
input_pos = keras.layers.Input(shape=(16,16,7), name="pos")
# print(input_pos)
# input_pos = tf.broadcast_to(input_pos, [16,16,7])
# TODO: reshape input to 16x16x7

# concatenate input_pos onto conv3
conv3a = keras.layers.concatenate([conv3, input_pos])

# conv4 kernel 3x3, stride 1x1
# ---> 16x16x128filters
skip_conv4 = keras.layers.Conv2D(filters = 128, kernel_size = (3,3), strides = (1,1), padding='same')(conv3a)
conv4 = keras.layers.Conv2D(filters = 128, kernel_size = (3,3), strides = (1,1), activation='relu', padding='same')(conv3a)

# Residual connection
conv4r = keras.layers.add([skip_conv4, conv4])

# conc kernel 1x1, stride 1x1
# ---> 16x16x128 (same as before)
conv5 = keras.layers.Conv2D(filters = 256, kernel_size = (3,3), strides = (1,1), activation='relu')(conv4r)

# THE RESULT OF THIS LAYER IS THE STATE REPRESENTATION
r = keras.layers.Conv2D(filters = 256, kernel_size = (1,1), strides=(1,1), activation='relu')(conv5)

# Define the model
model = keras.Model(inputs=[input_vis, input_pos], outputs=r)

# check if model outputs
inV = np.zeros((1,64,64,3))
inP = np.zeros((1,16,16,7)) # TODO: should reshape within model itself
print(model.predict([inV, inP], verbose=1))
keras.utils.plot_model(model, to_file='model.png')
