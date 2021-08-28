import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Layer, Conv1D, Conv2D, MaxPooling2D, Dense, Flatten, Reshape, UpSampling2D, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping

import deeptrack as dt # Used to construct dataset of particle movement
from itertools import islice

import tensorboard
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.close('all')
plt.style.use('seaborn-deep')
mpl.rcParams.update({
    'text.usetex': True,
    'pgf.rcfonts': False,
    'lines.linewidth': 1,
    'figure.dpi': 300,
})

from models import ConvolutionalAutoencoder
from callbacks import LogImageCallback
from data import create_autoencoder_generator
import json

image_size = 64
batch_size = 128
code_dims = [8,16,32,64,128]
depth = 4

for code_dim in code_dims:
    frame_dataset = create_autoencoder_generator(image_size, batch_size)
    autoencoder = ConvolutionalAutoencoder(image_size, code_dim, depth)
    autoencoder.compile(optimizer='adam', loss='mse')

    log_dir = f'logs/auto_{autoencoder.code_dim}_depth_{autoencoder.depth}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq='batch',
    )

    reference_images = tf.convert_to_tensor(np.array([x[0][0] for _, x in zip(range(5), frame_dataset)]))
    image_callback = LogImageCallback(log_dir, reference_images)

    auto_history = autoencoder.fit(
        frame_dataset,
        epochs=20,
        steps_per_epoch=50,
        shuffle=True,
        callbacks=[
            tensorboard_callback,
            image_callback,
            EarlyStopping(monitor='loss',
                          mode='min',
                          patience=10,
                          restore_best_weights=True)
        ]
    )

    autoencoder.save(f'trained_models/autoencoder_dim_{code_dim}/model')
    autoencoder.save_weights(f'trained_models/autoencoder_dim_{code_dim}/weights')
    history_dict = auto_history.history
    data = json.dumps(history_dict)
    with open(f'trained_models/autoencoder_dim_{code_dim}/history.json',"w") as f:
        f.write(data)

fig, axes = plt.subplots(2,5, figsize=(10,5), dpi=200, sharex=True, sharey=True)
for ax1, ax2 in axes.T:
    original = next(frame_dataset)[0][0]
    reconstructed = autoencoder(np.expand_dims(original, 0))
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original')
    ax2.imshow(reconstructed[0], cmap='gray')
    ax2.set_title('Reconstruction')

fig.tight_layout()

code = autoencoder.encoder(np.expand_dims(original, 0))
print(code.shape)
