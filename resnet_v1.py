from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K

class ResNet_v1:
    # data: input to the residual module
    # K: number of filters that will be learned by the final convolutional layer (the first two convolutional layers will learn K/4 filters)
    # stride: controls the stride of the convolution (will help us reduce spatial dimensions without using max pooling)
    # chanDim: defines the axis which will perform batch normalization
    # red (i.e. reduce) will control whether we are reducing spatial dimensions (True) or not (False) as not all residual modules will reduce dimensions of our spatial volume
    # reg: applies regularization strength for all convolutional layers in the residual module
    # bnEps: controls the Ɛ responsible for avoiding “division by zero” errors when normalizing inputs
    # bnMom: controls the momentum for the moving average
    def residual_module(data, K, stride, chanDim, red=False, reg=0.0001, bnEps=2e-5, bnMom=0.9):

        shortcut = data

        # ResNet module first bloc : 1x1 convolution
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)

        # ResNet module second bloc : 3x3 convolution
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False, kernel_regularizer=l2(reg))(act2)

        # ResNet module third bloc : 1x1 convolution
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        # if we want to reduce the spatial size, apply a convolutional layer to the shortcut
        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)

        # add together the shortcut and the final CONV
        x = add([conv3, shortcut])

        # the output of the ResNet module is the addition of the shortcut and the third bloc last convolution
        return x

    # width, height and depth are the input shape of the model
    # classes is the number of categories
    # stages are de different steps where we change our filter size
    # filters are the different filters sizes that will we used acress the architecture
    # reg: applies regularization strength for all convolutional layers in the residual module
    # bnEps: controls the Ɛ responsible for avoiding “division by zero” errors when normalizing inputs
    # bnMom: controls the momentum for the moving average
    def build(width, height, depth, classes, stages, filters, reg=0.0001, bnEps=2e-5, bnMom=0.9):