from keras.layers.experimental.preprocessing import Rescaling
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Multiply
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K

class ResNet_v1:

    NET_NAME = 'ResNet_v1'

    @staticmethod
    def residual_module(data, K, stride, chanDim, red=False, reg=0.0001, bnEps=2e-5, bnMom=0.9, se=False):
        """
        :param data: input to the residual module
        :param K: number of filters that will be learned by the final convolutional layer (the first two convolutional layers will learn K/4 filters)
        :param stride: controls the stride of the convolution (will help us reduce spatial dimensions without using max pooling)
        :param chanDim: defines the axis which will perform batch normalization
        :param red: (i.e. reduce) will control whether we are reducing spatial dimensions (True) or not (False) as not all residual modules will reduce dimensions of our spatial volume
        :param reg: applies regularization strength for all convolutional layers in the residual module
        :param bnEps: controls the Ɛ responsible for avoiding “division by zero” errors when normalizing inputs
        :param bnMom: controls the momentum for the moving average
        :return: Output = input + layer(input)
        """

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

        if se:
            conv3 = ResNet_v1.se_module(conv3, K)

        # if we want to reduce the spatial size, apply a convolutional layer to the shortcut
        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)

        # add together the shortcut and the final CONV
        x = add([conv3, shortcut])

        # the output of the ResNet module is the addition of the shortcut and the third bloc last convolution
        return x

    @staticmethod
    def se_module(input, channel, ratio=16):
        se = GlobalAveragePooling2D()(input)
        se = Dense(channel // ratio, activation='relu')(se)
        se = Dense(channel, activation='sigmoid')(se)
        se = Reshape([1, 1, channel])(se)
        return Multiply()([input, se])

    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=0.0001, bnEps=2e-5, bnMom=0.9, se=False):
        """
        :param width: Width of the model input
        :param height: Height of the model input
        :param depth: Depth of the model input
        :param classes: Number of categories
        :param stages: Different steps where our filter size change
        :param filters: Filters sizes that will we used across the architecture
        :param reg: Regularization strength for all convolution layers in the residual module
        :param bnEps: Ɛ responsible for avoiding “division by zero” errors when normalizing inputs
        :param bnMom: Momentum for the moving average
        :return: return a full fledged resnet module
        """

        inputShape = (height, width, depth)
        chanDim = -1

        # if the data is using "channels first" format, update the input shape and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # set the input and apply BN
        inputs = Input(shape=inputShape)
        # Normalize pixel values from 255 to [0,1]
        x = Rescaling(1./255, input_shape=(height, width, depth))(inputs)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)

        # apply CONV => BN => ACT => POOL to reduce spatial size. Original paper presents a 7x7 kernel.
        x = Conv2D(filters[0], (5, 5), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        x = Activation("relu")(x)

        # ZeroPadding2D layer can add rows and columns of zeros at the top, bottom, left and right side of an image tensor.
        x = ZeroPadding2D((1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # loop over the number of stages
        for i in range(0, len(stages)):
            # initialize the stride, then apply a residual module
            # used to reduce the spatial size of the input volume
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet_v1.residual_module(x, filters[i + 1], stride, chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

            # loop over the number of layers in the stage
            for j in range(0, stages[i] - 1):
                # apply a ResNet module
                x = ResNet_v1.residual_module(x, filters[i + 1], (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom, se=se)

        # apply BN => ACT => POOL
        x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                               momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)

        # create the model
        model = Model(inputs, x, name="resnet_v1")

        # return the constructed network architecture
        return model