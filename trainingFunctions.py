import matplotlib.pyplot as plt
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from trainingConstants import *
import numpy as np

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

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax[0].plot(epochs_range, acc, label='Training Accuracy')
    ax[0].plot(epochs_range, val_acc, label='Validation Accuracy')
    ax[0].set_title('Training and Validation Accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('accuracy')
    ax[0].set_yticks(np.arange(0, 1.05, 0.05))
    ax[0].legend(loc='lower right')

    ax[0].annotate('Max :%0.2f' % max(val_acc), xy=(1, max(val_acc)), xytext=(2, 8),
                   xycoords=('axes fraction', 'data'),
                   textcoords='offset points',
                   transform=ax[0].transAxes,
                   bbox=dict(boxstyle="round", fc=colors[1], ec=colors[1]))

    ax[0].annotate('Last :%0.2f' % val_acc[-1], xy=(1, val_acc[-1]), zorder=1, xytext=(2, -8),
                   xycoords=('axes fraction', 'data'),
                   textcoords='offset points',
                   bbox=dict(boxstyle="round", fc=colors[0], ec=colors[0]))

    ax[1].plot(epochs_range, loss, label='Training Loss')
    ax[1].plot(epochs_range, val_loss, label='Validation Loss')
    ax[1].set_title('Training and Validation Loss')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('loss')
    ax[1].set_yticks(np.arange(0, 1.05, 0.05))
    ax[1].legend(loc='upper right')

    ax[1].annotate('Min :%0.2f' % min(val_loss), xy=(1, min(val_loss)), xytext=(2, -8),
                   xycoords=('axes fraction', 'data'),
                   textcoords='offset points',
                   transform=ax[0].transAxes,
                   bbox=dict(boxstyle="round", fc=colors[1], ec=colors[1]))

    ax[1].annotate('Last :%0.2f' % val_loss[-1], xy=(1, val_loss[-1]), zorder=1, xytext=(2, 8),
                   xycoords=('axes fraction', 'data'),
                   textcoords='offset points',
                   bbox=dict(boxstyle="round", fc=colors[1], ec=colors[1]))

    '''for var in (f1y, f2y):
        print(var)
        ax[1].annotate('%0.2f' % max(var), xy=(1, max(np.array(var))), xytext=(2, 0), 
                     xycoords=('axes fraction', 'data'), textcoords='offset points', transform=ax[1].transAxes)
    '''
    plt.show()

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


