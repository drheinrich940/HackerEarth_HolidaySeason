from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from constants.trainingConstants import *
import utils
import trainingFunctions
from resnet_v1 import ResNet_v1

training_loss = None
if LOSS == 'SCCE':
    training_loss = SparseCategoricalCrossentropy()

training_opti = None
if OPTI == 'ADAM':
    training_opti = Adam()


model = ResNet_v1.build(width=IMAG_WIDTH, height=IMG_HEIGHT, depth=IMG_DEPTH,
                        classes=NB_CLASSES, stages=STAGES,
                        filters=FILTERS, se=SE_MODULES)

model.compile(
    optimizer=training_opti,
    loss=training_loss,
    metrics=['accuracy'],
)

history = trainingFunctions.training_augmented(model, EPOCHS, SEED)
utils.plot_training_results(history, EPOCHS, model, save=True)
utils.log_training_results(history, model)
trainingFunctions.increment_training_cpt()
