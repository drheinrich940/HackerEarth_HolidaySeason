from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
import trainingFunctions
from resnet_v1 import ResNet_v1

EPOCHS = 20

stages = (3, 6, 9)
filters = (64, 128, 256, 512)

model = ResNet_v1.build(width=trainingFunctions.IMAG_WIDTH, height=trainingFunctions.IMG_HEIGHT, depth=trainingFunctions.IMG_DEPTH,
                        classes=trainingFunctions.NB_CLASSES, stages=stages,
                        filters=filters)

model.compile(
    optimizer=Adam(),
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
)

history = trainingFunctions.training_augmented(model, EPOCHS)
trainingFunctions.plot_training_results(history, EPOCHS)
