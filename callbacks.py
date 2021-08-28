from tensorflow import keras
import tensorflow as tf

class LogImageCallback(keras.callbacks.Callback):
    def __init__(self, log_dir, reference_images):
        self.image_writer = tf.summary.create_file_writer(log_dir + '/images')
        self.num_images = reference_images.shape[0]
        self.references = reference_images

    def on_train_begin(self, logs=None):
        with self.image_writer.as_default():
            tf.summary.image("auto/Original", self.references / tf.reduce_max(self.references), max_outputs=self.num_images, step=0)

    def on_epoch_begin(self, epoch, logs=None):
        reconstructed = self.model.predict(self.references)
        with self.image_writer.as_default():
            tf.summary.image("auto/Reconstructed", reconstructed / tf.reduce_max(self.references), max_outputs=self.num_images, step=epoch)

class LogImageSequenceCallback(keras.callbacks.Callback):
    def __init__(self, log_dir, prior, truth):
        self.image_writer = tf.summary.create_file_writer(log_dir + '/images')
        self.prior_length = prior.shape[0]
        self.truth_length = truth.shape[0]
        self.prior = prior
        self.truth = truth

    def on_train_begin(self, logs=None):
        with self.image_writer.as_default():
            tf.summary.image("seq/Prior", self.prior, max_outputs=self.prior_length, step=0)
            tf.summary.image("seq/Truth", self.truth, max_outputs=self.truth_length, step=0)

    def on_epoch_begin(self, epoch, logs=None):
        encoded_x = tf.expand_dims(self.model.autoencoder.encoder(self.prior), 0)
        pred = self.model.predict(encoded_x)
        if self.truth_length != 1: pred = tf.squeeze(pred)
        decoded = self.model.autoencoder.decoder(pred)
        with self.image_writer.as_default():
            tf.summary.image("seq/Prediction", decoded, max_outputs=self.truth_length, step=epoch)
