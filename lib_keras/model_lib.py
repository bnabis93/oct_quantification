from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras import models
from keras import backend

from keras import backend as K
import tensorflow as tf
import numpy as np
# Compatible with tensorflow backend
K.set_image_data_format('channels_first')
kinit = 'he_normal'
#kinit = 'glorot_normal'
np.random.seed(5)



def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
        
    Reference : https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed


def generalized_dice_coeff(y_true, y_pred):
    Ncl = y_pred.shape[-1]
    w = K.zeros(shape=(Ncl,))
    w = K.sum(y_true, axis=(0,1,2))
    w = 1/(w**2+0.000001)
    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = w*K.sum(numerator,(0,1,2))
    numerator = K.sum(numerator)

    denominator = y_true+y_pred
    denominator = w*K.sum(denominator,(0,1,2))
    denominator = K.sum(denominator)

    gen_dice_coef = 2*(numerator+K.epsilon()) / (denominator + K.epsilon())

    return gen_dice_coef

def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice_coeff(y_true, y_pred)



def hybrid_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def hybrid_focal_dice_loss(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        focal_loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        focal_loss = K.sum(focal_loss, axis = 1)
        
        # Sum the losses in mini_batch
        Ncl = y_pred.shape[-1]
        w = K.zeros(shape=(Ncl,))
        #w = K.sum(y_true, axis=(0,1,2))
        w = K.sum(y_true, axis=-1)
        w = 1/(w**2+0.000001)
        # Compute gen dice coef:
        numerator = y_true*y_pred
        #numerator = w*K.sum(numerator,(0,1,2))
        numerator = w*K.sum(numerator,axis=-1)
        numerator = K.sum(numerator)

        denominator = y_true+y_pred
        #denominator = w*K.sum(denominator,(0,1,2))
        denominator = w*K.sum(denominator,axis=-1)
        denominator = K.sum(denominator)

        gen_dice_coef = 2*(numerator+K.epsilon()) / (denominator + K.epsilon())
        gen_dice_loss = (1- gen_dice_coef)
        
        
        return focal_loss + gen_dice_loss

    return hybrid_focal_dice_loss


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    K.set_image_data_format('channels_first')
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

# https://stackoverflow.com/questions/57155970/why-reshape-and-permute-for-segmentation-with-unet

def unet(n_ch,patch_height,patch_width,num_classes):
    print('is ch first ? ',backend.image_data_format())
    K.set_image_data_format('channels_first')
    inputs = layers.Input(shape=(n_ch,patch_height,patch_width))
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Dropout(0.2)(conv1)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    #
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Dropout(0.2)(conv2)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    #
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Dropout(0.2)(conv3)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = layers.UpSampling2D(size=(2, 2))(conv3)
    up1 = layers.concatenate([conv2,up1],axis=1)
    conv4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = layers.Dropout(0.2)(conv4)
    conv4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    #
    up2 = layers.UpSampling2D(size=(2, 2),data_format='channels_first')(conv4)
    up2 = layers.concatenate([conv1,up2], axis=1)
    conv5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = layers.Dropout(0.2)(conv5)
    conv5 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    #
    conv6 = layers.Conv2D(num_classes, (1, 1), activation='relu',padding='same')(conv5)
    conv6 = layers.core.Reshape((num_classes,patch_height*patch_width))(conv6)
    conv6 = layers.core.Permute((2,1))(conv6)
    ############
    conv7 = layers.core.Activation('softmax')(conv6)

    model = models.Model(inputs=inputs, outputs=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer= adam, loss=focal_loss(alpha=.25, gamma=2) ,metrics=['accuracy'])
    print('is ch first ? ',backend.image_data_format())
    model.summary()
    return model

#for batch normalization
def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    x = layers.Conv2D(filters=n_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer= 'he_normal',
                     padding = 'same')(input_tensor)
    
    if batchnorm ==True:
        x = layers.BatchNormalization()(x)
    
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters=n_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer= 'he_normal',
                     padding = 'same')(x)
    if batchnorm ==True:
        x = layers.BatchNormalization()(x)
    
    x = layers.Activation('relu')(x)
    return x





'''
For Attention block

[To Do]
    - Explain 'Function what to do'
    - Explain 'What is the self-attention in cnn?'
    - Explain 'Novelty of this system'

'''


def channel_attention(input_feature, ratio=8):
    print('input feature shape', input_feature._keras_shape)
    channel = input_feature._keras_shape[1]

    shared_layer_one = layers.Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    
    shared_layer_two = layers.Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    print('before reshpae avg pool', avg_pool._keras_shape)
    avg_pool = layers.Reshape((channel,1, 1))(avg_pool)
    print('after reshpae avg pool', avg_pool._keras_shape)

    avg_pool = shared_layer_one(avg_pool)
    print('after shared layer 01 avg pool', avg_pool._keras_shape)

    avg_pool = shared_layer_two(avg_pool)
    print('after shared layer 02 avg pool', avg_pool._keras_shape)


    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    print('before reshpae max pool', max_pool._keras_shape)

    max_pool = layers.Reshape((channel,1, 1))(max_pool)
    print('after reshpae max pool', max_pool._keras_shape)

    max_pool = shared_layer_one(max_pool)
    print('after shared layer 01 max pool', max_pool._keras_shape)

    max_pool = shared_layer_two(max_pool)
    print('before shared layer 02 max pool', max_pool._keras_shape)


    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)
    
    return layers.multiply([input_feature, cbam_feature])

def spatial_attention(input_feature, kernel_size=7):
        avg_pool = layers.Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(input_feature)
        #print('shape of avg pool and input feature : ', np.shape(input_feature), np.shape(avg_pool))
        max_pool = layers.Lambda(lambda x: K.max(x, axis=1, keepdims=True))(input_feature)
        #print('shape of max pool and input feature : ', np.shape(input_feature), np.shape(max_pool))

        concat = layers.Concatenate(axis=1)([avg_pool, max_pool])
        
        cbam_feature = layers.Conv2D(filters=1,
                      kernel_size=kernel_size,
                      strides=1,
                      padding='same',
                      activation='sigmoid',
                      kernel_initializer='he_normal',
                      use_bias=False)(concat)
        return layers.multiply([input_feature, cbam_feature])
    
def cbam_block(cbam_feature, ratio=2):
        # https://github.com/kobiso/CBAM-keras/blob/master/models/attention_module.py
        cbam_feature = channel_attention(cbam_feature, ratio)
        cbam_feature = spatial_attention(cbam_feature)
        return cbam_feature
    
    
def unet_norm(n_ch,patch_height,patch_width,num_classes):
    K.set_image_data_format('channels_first')
    inputs = layers.Input(shape=(n_ch,patch_height,patch_width))
    conv1 = conv2d_block(inputs, n_filters= 32, kernel_size=3, batchnorm=True)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    #
    conv2 = conv2d_block(pool1, n_filters= 64, kernel_size=3, batchnorm=True)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    
    # middel
    conv3 = conv2d_block(pool2, n_filters= 128, kernel_size=3, batchnorm=True)
    up1 = layers.UpSampling2D(size=(2, 2))(conv3)
    up1 = layers.concatenate([conv2,up1],axis=1)
    
    
    conv4 = conv2d_block(up1, n_filters= 64, kernel_size=3, batchnorm=True)
    
    
    up2 = layers.UpSampling2D(size=(2, 2))(conv4)
    up2 = layers.concatenate([conv1,up2], axis=1)
    conv5 = conv2d_block(up2, n_filters= 32, kernel_size=3, batchnorm=True)
    
    conv6 = layers.Conv2D(num_classes, (1, 1), activation='relu',padding='same')(conv5)
    
    conv6 = layers.core.Reshape((num_classes,patch_height*patch_width))(conv6)
    conv6 = layers.core.Permute((2,1))(conv6)
    ############
    conv7 = layers.core.Activation('softmax')(conv6)
    #print(np.shape(conv7))
    model = models.Model(inputs=inputs, outputs=conv7) # output (patch height * patch width, num classes)
    
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    adam = Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #weights = np.array([0.5,2,1,1,3])
    model.compile(optimizer=adam, loss=hybrid_loss(gamma=2., alpha=.25),metrics=['accuracy'])
    #model.compile(optimizer=adam, loss=generalized_dice_loss,metrics=['accuracy'])
    #model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model

def bigger_unet_norm(n_ch,patch_height,patch_width,num_classes):
    K.set_image_data_format('channels_first')
    inputs = layers.Input(shape=(n_ch,patch_height,patch_width))
    conv1 = conv2d_block(inputs, n_filters= 32, kernel_size=3, batchnorm=True)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    #
    conv2 = conv2d_block(pool1, n_filters= 64, kernel_size=3, batchnorm=True)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    conv3 = conv2d_block(pool2, n_filters= 128, kernel_size=3, batchnorm=True)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    #middel
    conv4 = conv2d_block(pool3, n_filters= 256, kernel_size=3, batchnorm=True)
    
    #up
    up1 = layers.UpSampling2D(size=(2, 2))(conv4)
    up1 = layers.concatenate([conv3,up1],axis=1)
    
    conv5 = conv2d_block(up1, n_filters= 128, kernel_size=3, batchnorm=True)
    
    up2 = layers.UpSampling2D(size=(2, 2))(conv5)
    up2 = layers.concatenate([conv2,up2], axis=1)
    
    conv6 = conv2d_block(up2, n_filters= 64, kernel_size=3, batchnorm=True)
    
    up3 = layers.UpSampling2D(size=(2, 2))(conv6)
    up3 = layers.concatenate([conv1,up3], axis=1)
    
    conv7 = conv2d_block(up3, n_filters= 32, kernel_size=3, batchnorm=True)
    
    conv8 = layers.Conv2D(num_classes, (1, 1), activation='relu',padding='same')(conv7)
    
    conv8 = layers.core.Reshape((num_classes,patch_height*patch_width))(conv8)
    conv8 = layers.core.Permute((2,1))(conv8)
    ############
    conv9 = layers.core.Activation('softmax')(conv8)
    #print(np.shape(conv7))
    model = models.Model(inputs=inputs, outputs=conv9) # output (patch height * patch width, num classes)
    
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    adam = Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #weights = np.array([0.5,2,1,1,3])
    model.compile(optimizer=adam, loss=hybrid_loss(gamma=2., alpha=.25),metrics=['accuracy'])
    #model.compile(optimizer=adam, loss=generalized_dice_loss,metrics=['accuracy'])
    #model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model


def naive_attn_unet(n_ch,patch_height,patch_width,num_classes):
    K.set_image_data_format('channels_first')
    inputs = layers.Input(shape=(n_ch,patch_height,patch_width))
    conv1 = conv2d_block(inputs, n_filters= 32, kernel_size=3, batchnorm=True)
    #conv1 = layers.SpatialDropout2D(0.1)(conv1)
    conv1 = spatial_attention(conv1)
    
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    #
    conv2 = conv2d_block(pool1, n_filters= 64, kernel_size=3, batchnorm=True)
    #conv2 = layers.SpatialDropout2D(0.1)(conv2)
    conv2 = spatial_attention(conv2)
    
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    #center
    conv3 = conv2d_block(pool2, n_filters= 128, kernel_size=3, batchnorm=True)
    #conv3 = layers.SpatialDropout2D(0.4)(conv3)
    conv3 = spatial_attention(conv3)
    
    up1 = layers.UpSampling2D(size=(2, 2))(conv3)
    up1 = layers.concatenate([conv2,up1],axis=1)
    conv4 = conv2d_block(up1, n_filters= 64, kernel_size=3, batchnorm=True)
    conv4 = spatial_attention(conv4)
    
    up2 = layers.UpSampling2D(size=(2, 2))(conv4)
    up2 = layers.concatenate([conv1,up2], axis=1)
    conv5 = conv2d_block(up2, n_filters= 32, kernel_size=3, batchnorm=True)
    conv5 = spatial_attention(conv5)
    
    conv6 = layers.Conv2D(num_classes, (1, 1), activation='relu',padding='same')(conv5)
    
    conv6 = layers.core.Reshape((num_classes,patch_height*patch_width))(conv6)
    conv6 = layers.core.Permute((2,1))(conv6)
    ############
    conv7 = layers.core.Activation('softmax')(conv6)

    model = models.Model(inputs=inputs, outputs=conv7)

    adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer=adam, loss=hybrid_loss(gamma=2., alpha=.25),metrics=['accuracy'])
    model.summary()
    return model

def bigger_naive_attn_unet(n_ch,patch_height,patch_width,num_classes):
    K.set_image_data_format('channels_first')
    inputs = layers.Input(shape=(n_ch,patch_height,patch_width))
    conv1 = conv2d_block(inputs, n_filters= 32, kernel_size=3, batchnorm=True)
    conv1 = spatial_attention(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    #
    conv2 = conv2d_block(pool1, n_filters= 64, kernel_size=3, batchnorm=True)
    conv2 = spatial_attention(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    conv3 = conv2d_block(pool2, n_filters= 128, kernel_size=3, batchnorm=True)
    conv3 = spatial_attention(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    #middel
    conv4 = conv2d_block(pool3, n_filters= 256, kernel_size=3, batchnorm=True)
    
    #up
    up1 = layers.UpSampling2D(size=(2, 2))(conv4)
    up1 = layers.concatenate([conv3,up1],axis=1)
    
    conv5 = conv2d_block(up1, n_filters= 128, kernel_size=3, batchnorm=True)
    conv5 = spatial_attention(conv5)
    
    up2 = layers.UpSampling2D(size=(2, 2))(conv5)
    up2 = layers.concatenate([conv2,up2], axis=1)
    
    conv6 = conv2d_block(up2, n_filters= 64, kernel_size=3, batchnorm=True)
    conv6 = spatial_attention(conv6)
    
    up3 = layers.UpSampling2D(size=(2, 2))(conv6)
    up3 = layers.concatenate([conv1,up3], axis=1)
    
    conv7 = conv2d_block(up3, n_filters= 32, kernel_size=3, batchnorm=True)
    conv7 = spatial_attention(conv7)
    
    conv8 = layers.Conv2D(num_classes, (1, 1), activation='relu',padding='same')(conv7)
    
    conv8 = layers.core.Reshape((num_classes,patch_height*patch_width))(conv8)
    conv8 = layers.core.Permute((2,1))(conv8)
    ############
    conv9 = layers.core.Activation('softmax')(conv8)
    #print(np.shape(conv7))
    model = models.Model(inputs=inputs, outputs=conv9) # output (patch height * patch width, num classes)
    
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    adam = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #weights = np.array([0.5,2,1,1,3])
    model.compile(optimizer=adam, loss=hybrid_loss(gamma=3., alpha=.25),metrics=['accuracy'])
    #model.compile(optimizer=adam, loss=generalized_dice_loss,metrics=['accuracy'])
    #model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model


def cbam_attn_unet(n_ch,patch_height,patch_width):
    K.set_image_data_format('channels_first')
    inputs = layers.Input(shape=(n_ch,patch_height,patch_width))
    conv1 = conv2d_block(inputs, n_filters= 32, kernel_size=3, batchnorm=True)
    conv1 = layers.SpatialDropout2D(0.1)(conv1)
    conv1 = cbam_block(conv1)
    
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    #
    conv2 = conv2d_block(pool1, n_filters= 64, kernel_size=3, batchnorm=True)
    conv2 = layers.SpatialDropout2D(0.1)(conv2)
    conv2 = cbam_block(conv2)
    
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    #center
    conv3 = conv2d_block(pool2, n_filters= 128, kernel_size=3, batchnorm=True)
    conv3 = layers.SpatialDropout2D(0.4)(conv3)
    conv3 = cbam_block(conv3)
    
    up1 = layers.UpSampling2D(size=(2, 2))(conv3)
    up1 = layers.concatenate([conv2,up1],axis=1)
    conv4 = conv2d_block(up1, n_filters= 64, kernel_size=3, batchnorm=True)
    conv4 = cbam_block(conv4)
    
    up2 = layers.UpSampling2D(size=(2, 2))(conv4)
    up2 = layers.concatenate([conv1,up2], axis=1)
    conv5 = conv2d_block(up2, n_filters= 32, kernel_size=3, batchnorm=True)
    conv5 = cbam_block(conv5)
    
    conv6 = layers.Conv2D(2, (1, 1), activation='relu',padding='same')(conv5)
    
    conv6 = layers.core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = layers.core.Permute((2,1))(conv6)
    ############
    conv7 = layers.core.Activation('softmax')(conv6)

    model = models.Model(inputs=inputs, outputs=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model

def small_attn_unet(n_ch,patch_height,patch_width):
    K.set_image_data_format('channels_first')
    inputs = layers.Input(shape=(n_ch,patch_height,patch_width))
    conv1 = conv2d_block(inputs, n_filters= 32, kernel_size=3, batchnorm=True)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    conv2 = conv2d_block(pool1, n_filters= 64, kernel_size=3, batchnorm=True)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    #center
    
    conv3 = conv2d_block(pool2, n_filters= 128, kernel_size=3, batchnorm=True)

    
    gating1 = UnetGatingSignal(conv3,True, 'gating01')
    
    attn1 = AttnGatingBlock(conv2, gating1, 128 ,'attn01')
    up1 = concatenate([Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',activation="relu")(conv3), attn1], axis=3)
    
    gating2 = UnetGatingSignal(up1, True, 'gating02')
    attn2 = AttnGatingBlock(conv1, gating2, 64,'attn02' )
    up2 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu")(up1), attn2], axis=3)
    
    
    conv6 = layers.Conv2D(2, (1, 1), activation='relu',padding='same')(up2)
    
    conv6 = layers.core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = layers.core.Permute((2,1))(conv6)
    ############
    conv7 = layers.core.Activation('softmax')(conv6)

    model = models.Model(inputs=inputs, outputs=conv7)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model


'''
from https://github.com/nabsabraham/focal-tversky-unet/blob/master/newmodels.py

multi-input attn model

'''

def expend_as(tensor, rep, name):
    # 20.02.11 element axis 3 -> 1
    my_repeat = layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis = 1), arguments={'repnum' : rep}, name = 'psi_up'+name)(tensor)
    return my_repeat

def AttnGatingBlock(x, g, inter_shape, name):
    ''' take g which is the spatially smaller signal, do a conv to get the same
    number of feature channels as x (bigger spatially)
    do a conv on x to also get same geature channels (theta_x)
    then, upsample g to be same size as x 
    add x and g (concat_xg)
    relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients'''
    
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(g)
    
    print('shape x,g ', shape_x,shape_g)
    
    theta_x = layers.Conv2D(inter_shape, (2,2), strides=(2,2), padding='same', name = 'xl'+name)(x)
    shape_theta_x  = K.int_shape(theta_x)
    print('inter shape :  ', inter_shape)

    
    phi_g = layers.Conv2D(inter_shape, (1,1), padding='same')(g)
    
    # 20.02.11 shape chane x, [1] -> [2] / y, [2] -> [3]
    print('stride x : {} stride y : {}'.format(shape_theta_x[2] // shape_g[2] ,shape_theta_x[3] // shape_g[3] ))
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[2] // shape_g[2], shape_theta_x[3] // shape_g[3]),padding='same', name='g_up'+name)(phi_g)  # 16
    #upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),strides=(2, 2),padding='same', name='g_up'+name)(phi_g)
    print('theta_x shape : ', K.int_shape(theta_x))
    print('upsample_g shape : ', K.int_shape(upsample_g))
    
    concat_xg = layers.merge.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    
    psi = layers.Conv2D(1, (1,1), padding= 'same', name = 'psi'+name)(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    
    # 20.02.11 shape change x -> [2] y -> [3]
    upsample_psi = layers.UpSampling2D(size=(shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3]))(sigmoid_xg)
    
    # 20.02.11 shape[3] -> shape[1]
    upsample_psi = expend_as(upsample_psi, shape_x[1], name)
    
    y = layers.merge.multiply([upsample_psi, x], name = 'q_attn'+name)
    
    # 20.02.11 shape change shape_x[3] -> shape_x[1]
    result = layers.Conv2D(shape_x[1], (1, 1), padding='same',name='q_attn_conv'+name)(y)
    result_bn = layers.BatchNormalization(name='q_attn_bn'+name)(result)
    
    return result_bn

def UnetGatingSignal(input, is_batchnorm, name):
    '''
    this is simply 1x1 convolution, bn, activation
    
    나는 ch-first를 사용한다. 
    shape[3]이 좀 이상하지?
    
    수정 할 필요가 있다.
    20.02.11
    x shape[3] => x shape[1]
    '''
    
    shape = K.int_shape(input)
    # 20.02.11 shape change, shape[3] -> shape[1]
    x = layers.Conv2D(shape[1] * 1, (1, 1), strides=(1, 1), padding="same", name=name + '_conv')(input)
    if is_batchnorm:
        x = layers.BatchNormalization(name=name + '_bn')(x)
    x = layers.Activation('relu', name = name+ '_act')(x)
    return x

def conv2d_block2(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    x = layers.Conv2D(filters=n_filters, kernel_size = (kernel_size, kernel_size), strides = (1,1),kernel_initializer= 'he_normal',
                     padding = 'same')(input_tensor)
    
    if batchnorm ==True:
        x = layers.BatchNormalization()(x)
    
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters=n_filters, kernel_size = (kernel_size, kernel_size), strides = (1,1),kernel_initializer= 'he_normal',
                     padding = 'same')(x)
    if batchnorm ==True:
        x = layers.BatchNormalization()(x)
    
    x = layers.Activation('relu')(x)
    return x

def attn_unet(n_ch,patch_height,patch_width,num_classes):
    K.set_image_data_format('channels_first')
    inputs = layers.Input(shape=(n_ch,patch_height,patch_width))
    print('input shape : ', K.int_shape(inputs))
    conv1 = conv2d_block2(inputs, n_filters= 32, kernel_size=3, batchnorm=True)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    conv2 = conv2d_block2(pool1, n_filters= 32, kernel_size=3, batchnorm=True)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    conv3 = conv2d_block2(pool2, n_filters= 64, kernel_size=3, batchnorm=True)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    
    conv4 = conv2d_block2(pool3, n_filters= 64, kernel_size=3, batchnorm=True)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    #center
    
    center = conv2d_block2(pool4, n_filters= 128, kernel_size=3, batchnorm=True)

    gating1 = UnetGatingSignal(center,True, 'gating01')
    print('\ngating shape : {}, conv4 shape : {}'.format(K.int_shape(gating1), K.int_shape(conv4)))
    
    attn1 = AttnGatingBlock(conv4, gating1, 128 ,'attn01')
    print('\nattn1 shape : {} center shape : {} '.format(K.int_shape(attn1), K.int_shape(center)))
    up1 = layers.concatenate([layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',activation="relu",kernel_initializer = kinit)(center), attn1], axis = 1)
    print('\nattn1 shape : {} up1 shape : {}'.format(K.int_shape(attn1), K.int_shape(up1)))
    
    gating2 = UnetGatingSignal(up1, True, 'gating02')

    attn2 = AttnGatingBlock(conv3, gating2, 64,'attn02' )
    up2 = layers.concatenate([layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',activation="relu",kernel_initializer = kinit)(up1), attn2], axis = 1)
    
    gating3 = UnetGatingSignal(up2, True, 'gating03')
    attn3 = AttnGatingBlock(conv2, gating3, 64,'attn03' )
    up3 = layers.concatenate([layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu",kernel_initializer = kinit)(up2), attn3], axis = 1)
    
    up4 = layers.concatenate([layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu",kernel_initializer = kinit)(up3), conv1], axis = 1)

    conv6 = layers.Conv2D(num_classes, (1, 1), activation='relu',padding='same',kernel_initializer = kinit)(up4)
    
    conv6 = layers.core.Reshape((num_classes,patch_height*patch_width))(conv6)
    conv6 = layers.core.Permute((2,1))(conv6)
    ############
    conv7 = layers.core.Activation('softmax')(conv6)

    model = models.Model(inputs=inputs, outputs=conv7)

    adam = Adam(lr=0.000005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    
    model.compile(optimizer=adam ,loss=hybrid_loss(gamma=.5, alpha=.25),metrics=['accuracy'])
    model.summary()
    return model


def attn_reg_test(nch, patch_height, patch_width, num_classes):
    K.set_image_data_format('channels_first')
    
    img_input = layers.Input(shape=(nch, patch_height, patch_width), name='input_scale1')
    print('input shape : ',K.int_shape(img_input))
    # resize image
    scale_img_2 = layers.AveragePooling2D(pool_size=(2, 2), name='input_scale2')(img_input)
    scale2_patch_height, scale2_patch_width = K.int_shape(scale_img_2)[2],K.int_shape(scale_img_2)[3]
    print('scale 2 input shape : ',K.int_shape(scale_img_2))
    
    scale_img_3 = layers.AveragePooling2D(pool_size=(2, 2), name='input_scale3')(scale_img_2)
    scale3_patch_height, scale3_patch_width =K.int_shape(scale_img_3)[2], K.int_shape(scale_img_3)[3]
    print('scale 3 input shape : ',K.int_shape(scale_img_3))
    
    scale_img_4 = layers.AveragePooling2D(pool_size=(2, 2), name='input_scale4')(scale_img_3)
    scale4_patch_height, scale4_patch_width = K.int_shape(scale_img_4)[2], K.int_shape(scale_img_4)[3]
    print('scale 4 input shape : ',K.int_shape(scale_img_4))
    
    
    
    conv1 = conv2d_block2(img_input, n_filters = 32,kernel_size= 3, batchnorm=True)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    input2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv_scale2')(scale_img_2)
    # axis 1이 ch (batch, ch, h, w)
    input2 = layers.concatenate([input2, pool1], axis=1)
    conv2 = conv2d_block2(input2, n_filters = 64,kernel_size = 3, batchnorm=True)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    input3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv_scale3')(scale_img_3)
    input3 = layers.concatenate([input3, pool2], axis=1)
    conv3 = conv2d_block2(input3,n_filters =  128, kernel_size = 3,batchnorm=True)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    input4 = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv_scale4')(scale_img_4)
    input4 = layers.concatenate([input4, pool3], axis=1)
    conv4 = conv2d_block2(input4, n_filters =  64,kernel_size = 3, batchnorm=True)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
        
    center = conv2d_block2(pool4,n_filters = 512,kernel_size = 3, batchnorm=True)
    
    # attention gate 
    
    g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 128, '_1')
    up1 = layers.concatenate([layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(center), attn1], name='up1', axis = 1)

    g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 64, '_2')
    up2 = layers.concatenate([layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up1), attn2], name='up2',axis = 1)

    g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 32, '_3')
    up3 = layers.concatenate([layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up2), attn3], name='up3', axis = 1)

    up4 = layers.concatenate([layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up3), conv1], name='up4', axis = 1)
    
    conv6 = conv2d_block2(up1, 256, batchnorm=True)
    conv7 = conv2d_block2(up2, 128, batchnorm=True)
    conv8 = conv2d_block2(up3, 64, batchnorm=True)
    conv9 = conv2d_block2(up4, 32, batchnorm=True)

    
    
    conv6 = layers.Conv2D(num_classes, (1, 1), activation='relu',padding='same',kernel_initializer = kinit)(conv6)
    conv6 = layers.core.Reshape((num_classes,scale4_patch_height * scale4_patch_width))(conv6)
    conv6 = layers.core.Permute((2,1))(conv6)
    #out6 = layers.Conv2D(1, (1, 1), activation='softmax', name='pred1')(conv6)
    out6 = layers.core.Activation('softmax',name = 'pred1')(conv6)
    #out6.name = 'pred1'
    
    conv7 = layers.Conv2D(num_classes, (1, 1), activation='relu',padding='same',kernel_initializer = kinit)(conv7)
    conv7 = layers.core.Reshape((num_classes, scale3_patch_height * scale3_patch_width))(conv7)
    conv7 = layers.core.Permute((2,1))(conv7)
    #out7 = layers.Conv2D(1, (1, 1), activation='softmax', name='pred2')(conv7)
    out7 = layers.core.Activation('softmax',name = 'pred2')(conv7)
    #out7.name = 'pred2'
    
    conv8 = layers.Conv2D(num_classes, (1, 1), activation='relu',padding='same',kernel_initializer = kinit)(conv8)
    conv8 = layers.core.Reshape((num_classes, scale2_patch_height * scale2_patch_width))(conv8)
    conv8 = layers.core.Permute((2,1))(conv8)
    #out8 = layers.Conv2D(1, (1, 1), activation='softmax', name='pred3')(conv8)
    out8 = layers.core.Activation('softmax',name = 'pred3')(conv8)
    #out8.name = 'pred3'
    print('conv8 shape : {} , out8 shape : {}'.format(K.int_shape(conv8), K.int_shape(out8)))
    
    conv9 = layers.Conv2D(num_classes, (1, 1), activation='relu',padding='same',kernel_initializer = kinit)(conv9)
    conv9 = layers.core.Reshape((num_classes,patch_height*patch_width))(conv9)
    conv9 = layers.core.Permute((2,1))(conv9)
    #out9 = layers.Conv2D(1, (1, 1), activation='softmax', name='final')(conv9)
    out9 = layers.core.Activation('softmax',name = 'final')(conv9)
    #out9.name = 'final'
    print('conv9 shape : {} , out9 shape : {}'.format(K.int_shape(conv9), K.int_shape(out9)))
    
    #out6 = Conv2D(1, (1, 1), activation='sigmoid', name='pred1')(conv6)
    #out7 = Conv2D(1, (1, 1), activation='sigmoid', name='pred2')(conv7)
    #out8 = Conv2D(1, (1, 1), activation='sigmoid', name='pred3')(conv8)
    #out9 = Conv2D(1, (1, 1), activation='sigmoid', name='final')(conv9)

    model = Model(inputs=[img_input], outputs=[out8,out9])
    
    # [To do ] loss
    multi_loss = {#'pred1': generalized_dice_loss,
            #'pred2': generalized_dice_loss,
            'pred3': generalized_dice_loss,
            'final': hybrid_loss(gamma=3., alpha=.25)}
    
    loss_weights = {#'pred1':1,
                    #'pred2':1,
                    'pred3':1,
                    'final':1}
    
    multi_metrics = {#'pred1' : 'accuracy', 
                    #'pred2' : 'accuracy', 
                    'pred3' : 'accuracy',
                    'final' : 'accuracy'}
    
    adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss= multi_loss, loss_weights=loss_weights,
                  metrics=multi_metrics)
    #model.compile(optimizer=opt, loss=loss, loss_weights=loss_weights,
    #              metrics=[generalized_dice_coeff])
    model.summary()
    return model


def attn_reg(nch, patch_height, patch_width, num_classes):
    K.set_image_data_format('channels_first')
    
    img_input = layers.Input(shape=(nch, patch_height, patch_width), name='input_scale1')
    print('input shape : ',K.int_shape(img_input))
    # resize image
    scale_img_2 = layers.AveragePooling2D(pool_size=(2, 2), name='input_scale2')(img_input)
    scale2_patch_height, scale2_patch_width = K.int_shape(scale_img_2)[2],K.int_shape(scale_img_2)[3]
    print('scale 2 input shape : ',K.int_shape(scale_img_2))
    
    scale_img_3 = layers.AveragePooling2D(pool_size=(2, 2), name='input_scale3')(scale_img_2)
    scale3_patch_height, scale3_patch_width =K.int_shape(scale_img_3)[2], K.int_shape(scale_img_3)[3]
    print('scale 3 input shape : ',K.int_shape(scale_img_3))
    
    scale_img_4 = layers.AveragePooling2D(pool_size=(2, 2), name='input_scale4')(scale_img_3)
    scale4_patch_height, scale4_patch_width = K.int_shape(scale_img_4)[2], K.int_shape(scale_img_4)[3]
    print('scale 4 input shape : ',K.int_shape(scale_img_4))
    
    
    
    conv1 = conv2d_block2(img_input, n_filters = 32,kernel_size= 3, batchnorm=True)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    input2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv_scale2')(scale_img_2)
    # axis 1이 ch (batch, ch, h, w)
    input2 = layers.concatenate([input2, pool1], axis=1)
    conv2 = conv2d_block2(input2, n_filters = 64,kernel_size = 3, batchnorm=True)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    input3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv_scale3')(scale_img_3)
    input3 = layers.concatenate([input3, pool2], axis=1)
    conv3 = conv2d_block2(input3,n_filters =  128, kernel_size = 3,batchnorm=True)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    input4 = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv_scale4')(scale_img_4)
    input4 = layers.concatenate([input4, pool3], axis=1)
    conv4 = conv2d_block2(input4, n_filters =  64,kernel_size = 3, batchnorm=True)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
        
    center = conv2d_block2(pool4,n_filters = 512,kernel_size = 3, batchnorm=True)
    
    # attention gate 
    
    g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 128, '_1')
    up1 = layers.concatenate([layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(center), attn1], name='up1', axis = 1)

    g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 64, '_2')
    up2 = layers.concatenate([layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up1), attn2], name='up2',axis = 1)

    g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 32, '_3')
    up3 = layers.concatenate([layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up2), attn3], name='up3', axis = 1)

    up4 = layers.concatenate([layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation='relu', kernel_initializer=kinit)(up3), conv1], name='up4', axis = 1)
    
    conv6 = conv2d_block2(up1, 256, batchnorm=True)
    conv7 = conv2d_block2(up2, 128, batchnorm=True)
    conv8 = conv2d_block2(up3, 64, batchnorm=True)
    conv9 = conv2d_block2(up4, 32, batchnorm=True)

    conv6 = layers.Conv2D(num_classes, (1, 1), activation='relu',padding='same',kernel_initializer = kinit)(conv6)
    conv6 = layers.core.Reshape((num_classes,scale4_patch_height * scale4_patch_width))(conv6)
    conv6 = layers.core.Permute((2,1))(conv6)
    #out6 = layers.Conv2D(1, (1, 1), activation='softmax', name='pred1')(conv6)
    out6 = layers.core.Activation('softmax',name = 'pred1')(conv6)
    #out6.name = 'pred1'
    
    conv7 = layers.Conv2D(num_classes, (1, 1), activation='relu',padding='same',kernel_initializer = kinit)(conv7)
    conv7 = layers.core.Reshape((num_classes, scale3_patch_height * scale3_patch_width))(conv7)
    conv7 = layers.core.Permute((2,1))(conv7)
    #out7 = layers.Conv2D(1, (1, 1), activation='softmax', name='pred2')(conv7)
    out7 = layers.core.Activation('softmax',name = 'pred2')(conv7)
    #out7.name = 'pred2'
    
    conv8 = layers.Conv2D(num_classes, (1, 1), activation='relu',padding='same',kernel_initializer = kinit)(conv8)
    conv8 = layers.core.Reshape((num_classes, scale2_patch_height * scale2_patch_width))(conv8)
    conv8 = layers.core.Permute((2,1))(conv8)
    #out8 = layers.Conv2D(1, (1, 1), activation='softmax', name='pred3')(conv8)
    out8 = layers.core.Activation('softmax',name = 'pred3')(conv8)
    #out8.name = 'pred3'
    
    conv9 = layers.Conv2D(num_classes, (1, 1), activation='relu',padding='same',kernel_initializer = kinit)(conv9)
    conv9 = layers.core.Reshape((num_classes,patch_height*patch_width))(conv9)
    conv9 = layers.core.Permute((2,1))(conv9)
    #out9 = layers.Conv2D(1, (1, 1), activation='softmax', name='final')(conv9)
    out9 = layers.core.Activation('softmax',name = 'final')(conv9)
    #out9.name = 'final'
    
    
    #out6 = Conv2D(1, (1, 1), activation='sigmoid', name='pred1')(conv6)
    #out7 = Conv2D(1, (1, 1), activation='sigmoid', name='pred2')(conv7)
    #out8 = Conv2D(1, (1, 1), activation='sigmoid', name='pred3')(conv8)
    #out9 = Conv2D(1, (1, 1), activation='sigmoid', name='final')(conv9)

    model = Model(inputs=[img_input], outputs=[out6, out7, out8, out9])
    
    # [To do ] loss
    loss = {'pred1': generalized_dice_loss,
            'pred2': generalized_dice_loss,
            'pred3': generalized_dice_loss,
            'final': hybrid_loss(gamma=3., alpha=.25)}
    
    loss_weights = {'pred1':1,
                    'pred2':1,
                    'pred3':1,
                    'final':1}
    
    multi_metrics = {'pred1' : 'accuracy', 
                    'pred2' : 'accuracy', 
                    'pred3' : 'accuracy',
                    'final' : 'accuracy'}
    
    adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss= loss, loss_weights=loss_weights,
                  metrics=multi_metrics)
    #model.compile(optimizer=opt, loss=loss, loss_weights=loss_weights,
    #              metrics=[generalized_dice_coeff])
    model.summary()
    return model

def class3_attn_unet(n_ch,patch_height,patch_width,num_classes):
    K.set_image_data_format('channels_first')
    inputs = layers.Input(shape=(n_ch,patch_height,patch_width))
    print('input shape : ', K.int_shape(inputs))
    conv1 = conv2d_block2(inputs, n_filters= 32, kernel_size=3, batchnorm=True)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    conv2 = conv2d_block2(pool1, n_filters= 32, kernel_size=3, batchnorm=True)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    conv3 = conv2d_block2(pool2, n_filters= 64, kernel_size=3, batchnorm=True)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    
    conv4 = conv2d_block2(pool3, n_filters= 64, kernel_size=3, batchnorm=True)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    #center
    
    center = conv2d_block2(pool4, n_filters= 128, kernel_size=3, batchnorm=True)

    gating1 = UnetGatingSignal(center,True, 'gating01')
    print('\ngating shape : {}, conv4 shape : {}'.format(K.int_shape(gating1), K.int_shape(conv4)))
    
    attn1 = AttnGatingBlock(conv4, gating1, 128 ,'attn01')
    print('\nattn1 shape : {} center shape : {} '.format(K.int_shape(attn1), K.int_shape(center)))
    up1 = layers.concatenate([layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',activation="relu",kernel_initializer = kinit)(center), attn1], axis = 1)
    print('\nattn1 shape : {} up1 shape : {}'.format(K.int_shape(attn1), K.int_shape(up1)))
    
    gating2 = UnetGatingSignal(up1, True, 'gating02')

    attn2 = AttnGatingBlock(conv3, gating2, 64,'attn02' )
    up2 = layers.concatenate([layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',activation="relu",kernel_initializer = kinit)(up1), attn2], axis = 1)
    
    gating3 = UnetGatingSignal(up2, True, 'gating03')
    attn3 = AttnGatingBlock(conv2, gating3, 64,'attn03' )
    up3 = layers.concatenate([layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu",kernel_initializer = kinit)(up2), attn3], axis = 1)
    
    up4 = layers.concatenate([layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu",kernel_initializer = kinit)(up3), conv1], axis = 1)

    conv6 = layers.Conv2D(num_classes, (1, 1), activation='relu',padding='same',kernel_initializer = kinit)(up4)
    
    conv6 = layers.core.Reshape((num_classes,patch_height*patch_width))(conv6)
    conv6 = layers.core.Permute((2,1))(conv6)
    ############
    conv7 = layers.core.Activation('softmax')(conv6)

    model = models.Model(inputs=inputs, outputs=conv7)

    adam = Adam(lr=0.000005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    
    model.compile(optimizer=adam ,loss=hybrid_loss(gamma= 3.5, alpha=.25) ,metrics=[generalized_dice_coeff])
    model.summary()
    return model

def small_class3_attn_unet(n_ch,patch_height,patch_width,num_classes):
    K.set_image_data_format('channels_first')
    inputs = layers.Input(shape=(n_ch,patch_height,patch_width))
    print('input shape : ', K.int_shape(inputs))
    conv1 = conv2d_block2(inputs, n_filters= 32, kernel_size=3, batchnorm=True)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    
    conv2 = conv2d_block2(pool1, n_filters= 64, kernel_size=3, batchnorm=True)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    
    conv3 = conv2d_block2(pool2, n_filters= 96, kernel_size=3, batchnorm=True)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    #center
    center = conv2d_block2(pool3, n_filters= 128, kernel_size=3, batchnorm=True)

    gating1 = UnetGatingSignal(center,True, 'gating01')
    attn1 = AttnGatingBlock(conv3, gating1, 128 ,'attn01')
    up1 = layers.concatenate([layers.Conv2DTranspose(96, (3, 3), strides=(2, 2), padding='same',activation="relu",kernel_initializer = kinit)(center), attn1], axis = 1)
    
    gating2 = UnetGatingSignal(up1, True, 'gating02')
    attn2 = AttnGatingBlock(conv2, gating2, 64,'attn02' )
    up2 = layers.concatenate([layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same',activation="relu",kernel_initializer = kinit)(up1), attn2], axis = 1)
    
    up3 = layers.concatenate([layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same',activation="relu",kernel_initializer = kinit)(up2), conv1], axis = 1)

    conv6 = layers.Conv2D(num_classes, (1, 1), activation='relu',padding='same',kernel_initializer = kinit)(up3)
    
    conv6 = layers.core.Reshape((num_classes,patch_height*patch_width))(conv6)
    conv6 = layers.core.Permute((2,1))(conv6)
    
    conv7 = layers.core.Activation('softmax')(conv6)

    model = models.Model(inputs=inputs, outputs=conv7)

    adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam ,loss=hybrid_loss(gamma=3., alpha=.25),metrics=[generalized_dice_coeff])
    #model.compile(optimizer=adam ,loss=hybrid_loss(gamma=3., alpha=.25),metrics=['accuracy'])
    model.summary()
    return model