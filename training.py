import os
import pathlib

import tensorflow as tf
from tensorflow import keras
from model import DeeplabV3Plus
from data.loader import create_train_val_dataset
from data.data_augmentation import aug_fn

tf.config.run_functions_eagerly(True)

loss = keras.losses.MeanSquaredError()

model = DeeplabV3Plus(image_size=512)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=loss,
    metrics=[tf.keras.metrics.MeanAbsoluteError()],
)

dataset_path = os.getenv("DATASET_PATH")

BATCH_SIZE = 4
train_data_path = pathlib.Path(dataset_path) / "train"
train_dataset, val_dataset = create_train_val_dataset(train_data_path, aug_fn)
train_dataset = train_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(1)

from datetime import datetime

run_id = f'{model.name}-{datetime.now().strftime("%m-%H%M%S")}'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"models/{run_id}/" + "{epoch:02d}-{val_mean_absolute_error:.2f}",
    save_weights_only=True,
    monitor='val_mean_absolute_error',
    mode='max',
    save_best_only=False)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"./logs/{run_id}", update_freq=100, )

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                          patience=5)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_mean_absolute_error",
    mode="max",
    patience=7,
    restore_best_weights=True)

history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=100,
                    callbacks=[model_checkpoint_callback, tensorboard_callback, reduce_lr_callback])
