import os
import pathlib
import io

import tensorflow as tf
from tensorflow import keras
from model import DeeplabV3Plus
from data.loader import create_train_val_dataset
from data.data_augmentation import aug_fn


def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def custom_loss(y_true, y_pred):
    return keras.losses.mean_squared_error(y_true, y_pred) + SSIMLoss(y_true, y_pred)


if __name__ == "__main__":
    model = DeeplabV3Plus(image_size=512)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=custom_loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    dataset_path = os.getenv("DATASET_PATH")

    BATCH_SIZE = 4
    train_data_path = pathlib.Path(dataset_path) / "train"
    train_dataset, val_dataset = create_train_val_dataset(train_data_path, aug_fn, BATCH_SIZE)

    from datetime import datetime

    run_id = f'{model.name}-{datetime.now().strftime("%m-%H%M%S")}'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"models/{run_id}/" + "{epoch:02d}-{val_mean_absolute_error:.2f}",
        save_weights_only=True,
        monitor='val_mean_absolute_error',
        mode='max',
        save_best_only=False)

    log_dir = f"./logs/{run_id}"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=100,
                                                          profile_batch=5)
    file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm', max_queue=1)


    def log_validation_images(epoch, logs):
        import matplotlib.pyplot as plt
        def plot_to_image(figure):
            """Converts the matplotlib plot specified by 'figure' to a PNG image and
            returns it. The supplied figure is closed and inaccessible after this call."""
            # Save the plot to a PNG in memory.
            buf = io.BytesIO()
            figure.savefig(buf, format='png')
            plt.close(figure)
            buf.seek(0)
            # Convert PNG buffer to TF image
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            # Add the batch dimension
            image = tf.expand_dims(image, 0)
            return image

        def image_grid(input, prediction):
            import numpy as np
            # Create a figure to contain the plot.
            figure = plt.figure(figsize=(20, 10))
            for idx, i in enumerate([input[0][0], input[1][0], prediction]):
                if i.dtype == np.float32:
                    i = np.copy(i)
                    i /= np.max(i)
                    i *= 255.0
                    i = i.astype(np.uint8)

                # Start next subplot.
                plt.subplot(1, 3, idx + 1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(i)
                plt.tight_layout()
            return figure

        # Use the model to predict the values from the validation dataset.
        in_image = val_dataset.take(1)
        test_pred_raw = model.predict(in_image)[0]
        input_image_as_np = in_image.as_numpy_iterator().next()

        figure = image_grid(input_image_as_np, test_pred_raw)
        cm_image = plot_to_image(figure)

        with file_writer_cm.as_default():
            res = tf.summary.image("epoch_validation_images", cm_image, step=epoch)


    # Define the per-epoch callback.
    images_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_validation_images)

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                              patience=10)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_mean_absolute_error",
        mode="max",
        patience=7,
        restore_best_weights=True)

    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=100,
                        callbacks=[model_checkpoint_callback, tensorboard_callback,
                                   reduce_lr_callback,
                                   images_callback])
