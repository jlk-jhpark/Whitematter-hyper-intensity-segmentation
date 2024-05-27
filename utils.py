from keras.utils import Sequence
from keras.models import Model
from keras import backend as K
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Activation,Dropout
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from keras import backend as K
import numpy as np
import os

##1-1 loss function setting
smooth = 1e-7

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth * 0.01) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


##1-2 model structure
## model 3D Unet

def conv_unet_layer(inputs_block,n):
    conv1 = Conv2D(n, (3, 3), padding='same')(inputs_block)
    conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(n, (3, 3), padding='same')(conv1)
    conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(n, (3, 3), padding='same')(conv1)
    conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    return conv1

def pooling_unet_layer(inputs,n):
    pool1 = Conv2D(n, (3, 3), padding='same', strides = (2 ,2))(inputs)
    pool1 = BatchNormalization(axis=3)(pool1)
    pool1 = Activation('relu')(pool1)
    return pool1


def SE_block(input, ratio=16):

    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = int(init.shape[channel_axis])
    se_shape = (1, 1, int(filters))
    
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def get_SEunet(drop_rate = 0.5):
    n =32
    inputs = Input((256, 256, 1))
    conv1 = conv_unet_layer(inputs,n)
    conv1 = SE_block(conv1)
    pool1 = pooling_unet_layer(conv1,n)
    Drop1 = Dropout(drop_rate)(pool1)
    conv2 = concatenate([Drop1, MaxPooling2D(pool_size=(2, 2))(inputs)], axis=3)
    conv22 = conv2 
    
    conv2 = conv_unet_layer(conv2,2*n)
    conv2 = SE_block(conv2)
    pool2 = pooling_unet_layer(conv2,2*n)
    Drop2 = Dropout(drop_rate)(pool2)
    conv3 = concatenate([Drop2, MaxPooling2D(pool_size=(2, 2))(conv22)], axis=3)
    conv33 = conv3 
    
    conv3 = conv_unet_layer(conv3,4*n)
    conv3 = SE_block(conv3)
    pool3 = pooling_unet_layer(conv3,4*n)
    Drop3 = Dropout(drop_rate)(pool3)
    conv4 = concatenate([Drop3, MaxPooling2D(pool_size=(2, 2))(conv33)], axis=3)
    
    conv4 = conv_unet_layer(conv4,8*n)

    up5 = concatenate([Conv2DTranspose(2*n, (3, 3), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
    conv5 = conv_unet_layer(up5,4*n)

    up6 = concatenate([Conv2DTranspose(n, (3, 3), strides=(2, 2), padding='same')(conv5), conv2], axis=3)
    conv6 = conv_unet_layer(up6,2*n)

    up7 = concatenate([Conv2DTranspose(n, (3, 3), strides=(2, 2), padding='same')(conv6), conv1], axis=3)
    conv7 = conv_unet_layer(up7,n)

    conv8 = Conv2D(1, (1, 1), activation='sigmoid')(conv7)
    model = Model(inputs=[inputs], outputs=[conv8])
    
    return model

def get_unet(drop_rate = 0.5):
    n =32
    inputs = Input((256, 256, 1))
    conv1 = conv_unet_layer(inputs,n)
    pool1 = pooling_unet_layer(conv1,n)
    Drop1 = Dropout(drop_rate)(pool1)
    conv2 = concatenate([Drop1, MaxPooling2D(pool_size=(2, 2))(inputs)], axis=3)
    conv22 = conv2
    
    conv2 = conv_unet_layer(conv2,2*n)
    pool2 = pooling_unet_layer(conv2,2*n)
    Drop2 = Dropout(drop_rate)(pool2)
    conv3 = concatenate([Drop2, MaxPooling2D(pool_size=(2, 2))(conv22)], axis=3)
    conv33 = conv3
    
    conv3 = conv_unet_layer(conv3,4*n)
    pool3 = pooling_unet_layer(conv3,4*n)
    Drop3 = Dropout(drop_rate)(pool3)
    conv4 = concatenate([Drop3, MaxPooling2D(pool_size=(2, 2))(conv33)], axis=3)
    
    conv4 = conv_unet_layer(conv4,8*n)

    up5 = concatenate([Conv2DTranspose(2*n, (3, 3), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
    conv5 = conv_unet_layer(up5,4*n)

    up6 = concatenate([Conv2DTranspose(n, (3, 3), strides=(2, 2), padding='same')(conv5), conv2], axis=3)
    conv6 = conv_unet_layer(up6,2*n)

    up7 = concatenate([Conv2DTranspose(n, (3, 3), strides=(2, 2), padding='same')(conv6), conv1], axis=3)
    conv7 = conv_unet_layer(up7,n)

    conv8 = Conv2D(1, (1, 1), activation='sigmoid')(conv7)
    model = Model(inputs=[inputs], outputs=[conv8])
    
    return model
class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,data_path,data_path_GT, batch_size=32, dim=(256,256), n_channels=1,
                 class_num=1,shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.class_num = class_num
        self.shuffle = shuffle
        self.data_path = data_path
        self.data_path_GT =data_path_GT
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size, *self.dim, self.n_channels))
                
        # Generate data
        for i, ID in enumerate(list_IDs_temp):            
            # Store sample
            img = np.load(os.path.join(self.data_path,ID))
            X[i,:,:,0] = img
            
            # Store class
            labelimg_idx = np.load(os.path.join(self.data_path_GT,ID))
            y[i,:,:,0] = labelimg_idx
            
        return X, y
    
