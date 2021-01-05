from resnet_v1 import ResNet_v1
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy

stages = (3, 6, 9)
filters = (64, 128, 256, 512)

model = ResNet_v1.build(width=80, height=300, depth=3, classes=3, stages=stages, filters=filters)

model.compile(
    #
    optimizer=Adam(),  # Optimizer
    # Loss function to minimize
    loss=SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[SparseCategoricalAccuracy()],
)

model.summary()