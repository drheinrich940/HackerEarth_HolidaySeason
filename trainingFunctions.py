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


def training_augmented(model, epochs):
    train_datagen = ImageDataGenerator(horizontal_flip=H_FLIP, vertical_flip=V_FLIP)
    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        'dataset\\augmented\\trainAugmented',
        target_size=(IMG_HEIGHT, IMAG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='sparse')

    val_generator = val_datagen.flow_from_directory(
        'dataset\\augmented\\validationAugmented',
        target_size=(IMG_HEIGHT, IMAG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='sparse')

    return model.fit(
        train_generator,
        validation_data=val_generator,
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
        if H_FLIP:
            data_aug = data_aug + 'H_FLIP'
        if V_FLIP:
            data_aug = data_aug + 'V_FLIP'

        text_file = open('trainingCounter.txt', 'r')
        train_cpt = text_file.readline()
        text_file.close()

        stages = 's_' + str(STAGES).replace(',', '_').replace('(', '').replace(')', '').replace(' ', '')
        filters = 'f_' + str(FILTERS).replace(',', '_').replace('(', '').replace(')', '').replace(' ', '')

        filename = model.name
        path = 'TrainingResults'
        plt.savefig(path+'/'+filename+'_'+str(EPOCHS)+'_'+str(BATCH_SIZE)+'_'+str(data_aug)+'_'+stages+'_'+filters+'_'+train_cpt+'.png')

    plt.show()

def increment_training_cpt():
    text_file = open('trainingCounter.txt', 'r')
    cpt = text_file.readline()
    text_file.close()
    text_file = open('trainingCounter.txt', 'w')
    n = text_file.write(str(int(cpt) + 1))
    text_file.close()

