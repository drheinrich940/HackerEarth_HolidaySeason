import matplotlib.pyplot as plt
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from trainingConstants import *
import numpy as np
import pandas as pd

def training(model, epochs):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Classical loading, no data augmentation
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'dataset\\trainSorted',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMAG_WIDTH),
        batch_size=BATCH_SIZE)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'dataset\\trainSorted',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMAG_WIDTH),
        batch_size=BATCH_SIZE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )


def training_augmented(model, epochs, seed):
    train_datagen = ImageDataGenerator(horizontal_flip=H_FLIP, vertical_flip=V_FLIP, validation_split=0.2)

    train_ds = train_datagen.flow_from_directory(
        'dataset\\trainSorted',
        target_size=(IMG_HEIGHT, IMAG_WIDTH),
        batch_size=BATCH_SIZE,
        subset="training",
        seed=seed,
        class_mode='sparse')

    val_ds = train_datagen.flow_from_directory(
        'dataset\\trainSorted',
        target_size=(IMG_HEIGHT, IMAG_WIDTH),
        batch_size=BATCH_SIZE,
        subset="validation",
        seed=seed,
        class_mode='sparse')

    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )


def plot_training_results(history, epochs, model, save=False):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    fig.subplots_adjust(wspace=.35)

    # As we plot validation acc & loss after their training counterparts, we use the corresponding curve color
    #   to pain the box color of the values of interest of each curve. Second plotted curve's color can be got
    #   using the following call.
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]

    # For the first subplot dedicated to Accuracy
    ax[0].plot(epochs_range, acc, label='Training Accuracy')
    ax[0].plot(epochs_range, val_acc, label='Validation Accuracy')
    ax[0].set_title('Training and Validation Accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('accuracy')
    ax[0].set_yticks(np.arange(0, 1.05, 0.05))
    ax[0].legend(loc='lower right')

    # Annotate helps us draw text at a custom location corresponding to a chosen value. It allows us to select our
    #   points of interests in the curve and add a custom labelled bbox displaying the exact value of the given point.
    #   Here we display the last epoch's accuracy value, and the best accuracy encountered during training.
    ax[0].annotate('Max :%0.2f' % max(val_acc), xy=(1, max(val_acc)), xytext=(2, 8),
                   xycoords=('axes fraction', 'data'),
                   textcoords='offset points',
                   transform=ax[0].transAxes,
                   bbox=dict(boxstyle="round", fc=color, ec=color))

    ax[0].annotate('Last :%0.2f' % val_acc[-1], xy=(1, val_acc[-1]), zorder=1, xytext=(2, -8),
                   xycoords=('axes fraction', 'data'),
                   textcoords='offset points',
                   bbox=dict(boxstyle="round", fc=color, ec=color))

    # For the second subplot dedicated to Loss
    ax[1].plot(epochs_range, loss, label='Training Loss')
    ax[1].plot(epochs_range, val_loss, label='Validation Loss')
    ax[1].set_title('Training and Validation Loss')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('loss')
    ax[1].set_yticks(np.arange(0, max(val_loss) + 0.05, 0.05))
    ax[1].legend(loc='upper right')

    # Annotate helps us again to display last epoch's loss value, and the minimal loss value ever computed during train.
    ax[1].annotate('Min :%0.2f' % min(val_loss), xy=(1, min(val_loss)), xytext=(2, -8),
                   xycoords=('axes fraction', 'data'),
                   textcoords='offset points',
                   transform=ax[0].transAxes,
                   bbox=dict(boxstyle="round", fc=color, ec=color))

    ax[1].annotate('Last :%0.2f' % val_loss[-1], xy=(1, val_loss[-1]), zorder=1, xytext=(2, 8),
                   xycoords=('axes fraction', 'data'),
                   textcoords='offset points',
                   bbox=dict(boxstyle="round", fc=color, ec=color))

    # Plot saving procedure.
    # Handles several variables to build a representative file name, including :
    #   - Model name
    #   - Optional SE
    #   - Number of epochs
    #   - Batch size
    #   - Data augmentation used, including horizontal and vertical flips
    #   - Stages of the model if it has any
    #   - Filters if it has stages
    #   - Training iteration of the model during the project.
    if save:
        data_aug = ''
        se = ''
        if H_FLIP:
            data_aug = data_aug + 'H_FLIP'
        if V_FLIP:
            data_aug = data_aug + 'V_FLIP'
        if SE_MODULES:
            se = se + 'SE_'

        text_file = open('trainingCounter.txt', 'r')
        train_cpt = text_file.readline()
        text_file.close()

        stages = 's_' + str(STAGES).replace(',', '_').replace('(', '').replace(')', '').replace(' ', '')
        filters = 'f_' + str(FILTERS).replace(',', '_').replace('(', '').replace(')', '').replace(' ', '')

        model_name = se + model.name
        path = 'TrainingResults'
        plt.savefig(path+'/'+model_name+'_'+str(EPOCHS)+'_'+str(BATCH_SIZE)+'_'+str(data_aug)+'_'+stages+'_'+filters+'_'+train_cpt+'.png')

    plt.show()

def increment_training_cpt():
    text_file = open('trainingCounter.txt', 'r')
    cpt = text_file.readline()
    text_file.close()
    text_file = open('trainingCounter.txt', 'w')
    n = text_file.write(str(int(cpt) + 1))
    text_file.close()

# For each training, log :
#   - model used : 'model',
#           example : ResNet_v1
#   - improvements used : ['imp1', 'imp2', ...],
#           example : ['SE', 'ASPP']
#   - loss used : 'loss',
#           example : 'SCCE'
#   - optimizer used : 'optim',
#           example : 'Nadamax'
#   - validation accuracy best, last, and history array : [best, last, [history]],
#           example : [0.981, 0.973, [0, 0.1, 0.4, 0.981, 0.973]]
#   - validation loss best, last, and history array : [best, last, [history]],
#           example : [0.1, 0.12, [0.9, 0.8, 0.5, 0.1, 0.12]]
#   - data aug used : ['data_aug1', 'data_aug2', ... ],
#           example : ['V_FLIP', 'H_FLIP']
#   - training iteration id (from file) : n,
#           example : 5
#   ? - train validation split ratio : n,
#           example : 0.2
#   ? - train validation split seed : n,
#           example : 123
'''def log_training_results(history, _model):
    model = 'model'
    improvements = 'improvements'
    loss = 'loss'
    optimizer = 'optimizer'

    data_aug = 'data_aug'

    train_iter_id = 'train_iter_id'
    batch_iter_id = 'batch_iter_id'

    val_acc_best = 'val_acc_best'
    val_acc_last = 'val_acc_last'
    val_acc_hist = 'val_acc_hist'
    val_loss_best = 'val_loss_best'
    val_loss_last = 'val_loss_last'
    val_loss_hist = 'val_loss_hist'

    split_ratio = 'split_ratio'
    split_seed = 'split_seed'

    headers = [model, improvements, loss, optimizer, val_acc_best, val_acc_last, val_acc_hist, val_loss_best,
               val_loss_last, val_loss_hist, data_aug, train_iter_id, batch_iter_id, split_ratio, split_seed]

    text_file = open('trainingCounter.txt', 'r')
    train_cpt = text_file.readline()
    text_file.close()

    _improvements = []
    if SE_MODULES:
        _improvements.append('SE')
    if ASPP:
        _improvements.append('ASPP')
    if SAM:
        _improvements.append('SAM')

    df = pd.DataFrame([], columns=headers)
    df = df.append({model: _model.name,
                    improvements: _improvements,
                    loss:
                    train_iter_id: train_cpt}, ignore_index=True)'''



