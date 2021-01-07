import matplotlib.pyplot as plt
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

IMAG_WIDTH = 80
IMG_HEIGHT = 300
IMG_DEPTH = 3
BATCH_SIZE = 8
NB_CLASSES = 6


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
    train_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
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


def plot_training_results(history, epochs):
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
    plt.show()
