import matplotlib.pyplot as plt
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from trainingConstants import *
from pandas import read_csv, DataFrame

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

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

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


