import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from constants.trainingConstants import *

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
    train_datagen = ImageDataGenerator(horizontal_flip=H_FLIP, vertical_flip=V_FLIP, validation_split=SPLIT_RATIO)

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


def increment_training_cpt():
    text_file = open('trainingCounter.txt', 'r')
    cpt = text_file.readline()
    text_file.close()
    text_file = open('trainingCounter.txt', 'w')
    n = text_file.write(str(int(cpt) + 1))
    text_file.close()
