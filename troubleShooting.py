from resnet_v1 import ResNet_v1
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
import tensorflow as tf

imgWidth = 80
imgHeight = 300
imgDepth = 3
batchSize = 32
epochs = 10
num_classes = 6

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  'dataset\\trainSorted',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(imgHeight, imgWidth),
  batch_size=batchSize)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  'dataset\\trainSorted',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(imgHeight, imgWidth),
  batch_size=batchSize)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

stages = (3, 6, 9)
filters = (64, 128, 256, 512)

model = ResNet_v1.build(width=imgWidth, height=imgHeight, depth=imgDepth, classes=num_classes, stages=stages, filters=filters)

model.compile(
    optimizer=Adam(),  # Optimizer
    # Loss function to minimize
    loss=SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[SparseCategoricalAccuracy()],
)

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)