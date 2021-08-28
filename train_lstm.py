from models import ParticlePredictorLSTM, ConvolutionalAutoencoder
from data import create_sequence_generator
from callbacks import LogImageSequenceCallback
import tensorflow as tf
import tensorflow.keras as keras

import numpy as np

import datetime
import matplotlib.pyplot as plt

lr = 1e-2
steps_per_epoch = 20
image_size = 64
batch_size = 16
prior_len = 8
truth_len = 1
code_dim = 64
video_dataset = create_sequence_generator(image_size, batch_size, prior_len, truth_len)

autoencoder = keras.models.load_model(
    f"trained_models/autoencoder_dim_{code_dim}/model/",
)
autoencoder.trainable = False
video_dataset = create_sequence_generator(image_size, batch_size, prior_len, truth_len)

log_dir = f'logs/lstm_{code_dim}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')

x, y = next(video_dataset)
prior = tf.convert_to_tensor(x[0])
truth = tf.convert_to_tensor(y[0])
image_callback = LogImageSequenceCallback(log_dir, prior, truth)

model = ParticlePredictorLSTM(autoencoder, code_dim, prior_len, truth_len, filt_dim=code_dim+2)
model.build(input_shape=(None, prior_len, code_dim))
optimizer = keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError(), run_eagerly=False)

model.fit(
    video_dataset,
    epochs=50,
    steps_per_epoch=steps_per_epoch,
    callbacks=[
        image_callback,
        tensorboard_callback,
    ]
)

model.save(f'trained_models/lstm_predictor')
