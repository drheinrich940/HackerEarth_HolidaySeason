from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from trainingConstants import *
import trainingFunctions
from resnet_v1 import ResNet_v1

model = ResNet_v1.build(width=IMAG_WIDTH, height=IMG_HEIGHT, depth=IMG_DEPTH,
                        classes=NB_CLASSES, stages=STAGES,
                        filters=FILTERS)

model.compile(
    optimizer=Adam(),
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
)

history = trainingFunctions.training_augmented(model, EPOCHS)
trainingFunctions.plot_training_results(history, EPOCHS, model, save=True)
trainingFunctions.increment_training_cpt()
