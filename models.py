import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv1D, Conv2D, MaxPooling2D, Dense, Flatten, Reshape, UpSampling2D, Concatenate, Dropout
import numpy as np

class ConvolutionDecoder(Model):
    def __init__(self, image_size, code_dim, depth):
        super(ConvolutionDecoder, self).__init__()
        self.code_dim = code_dim
        self.image_size = image_size
        self.depth = depth

    def build(self, input_shape):
        self.dense_1 = Dense(4*4*16)
        self.reshape = Reshape((4,4,16))
        self.convs = [
            Conv2D((16*2**(self.depth)) // 2**(i),
                   kernel_size=3, strides=1, padding='same', activation='relu')
            for i in range(self.depth)
        ]
        self.upsamples = [UpSampling2D((2,2)) for _ in range(self.depth)]
        self.final_conv = Conv2D(1, kernel_size=3, strides=1, padding='same')

    def call(self, inputs):
        out = self.dense_1(inputs)
        out = self.reshape(out)

        for i in range(self.depth):
            out = self.convs[i](out)
            out = self.upsamples[i](out)

        out = self.final_conv(out)
        return out

class ConvolutionEncoder(Model):
    def __init__(self, image_size, code_dim, depth):
        super(ConvolutionEncoder, self).__init__()
        self.code_dim = code_dim
        self.image_size = image_size
        self.depth = depth

    def build(self, input_shape):
        self.convs = [
            Conv2D(16 * 2**(i),
                   3, strides=2, activation='relu')
            for i in range(self.depth)
        ]
        self.flat = Flatten()
        self.dense_1 = Dense(self.code_dim)

    def call(self, inputs):
        out = inputs
        for l in self.convs:
            out = l(out)

        out = self.flat(out)
        return self.dense_1(out)

class ConvolutionalAutoencoder(Model):
    def __init__(self, image_size, code_dim, depth):
        super(ConvolutionalAutoencoder, self).__init__()
        self.image_size = image_size
        self.code_dim = code_dim
        self.depth = depth

    def build(self, input_shape):
        self.encoder = ConvolutionEncoder(self.image_size, self.code_dim, self.depth)
        self.decoder = ConvolutionDecoder(self.image_size, self.code_dim, self.depth)

    def get_config(self):
        return {
            "image_size": self.image_size,
            "code_dim": self.code_dim,
            "depth": self.depth,
        }

    def call(self, inputs):
        encoding = self.encoder(inputs)
        encoding = layers.GaussianNoise(0.1)(encoding)
        decoding = self.decoder(encoding)
        return decoding

class Time2Vector(Layer):
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        self.weights_linear = self.add_weight(name='weight_linear',
                                            shape=(int(self.seq_len),),
                                            initializer='uniform',
                                            trainable=True)

        self.bias_linear = self.add_weight(name='bias_linear',
                                        shape=(int(self.seq_len),),
                                        initializer='uniform',
                                        trainable=True)

        self.weights_periodic = self.add_weight(name='weight_periodic',
                                                shape=(int(self.seq_len),),
                                                initializer='uniform',
                                                trainable=True)

        self.bias_periodic = self.add_weight(name='bias_periodic',
                                            shape=(int(self.seq_len),),
                                            initializer='uniform',
                                            trainable=True)

    def call(self, x):
        x = tf.math.reduce_mean(x, axis=-1) # Convert (batch, seq_len, 5) to (batch, seq_len)
        time_linear = self.weights_linear * x + self.bias_linear
        time_linear = tf.expand_dims(time_linear, axis=-1) # (batch, seq_len, 1)

        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1) # (batch, seq_len, 1)
        return tf.concat([time_linear, time_periodic], axis=-1) # (batch, seq_len, 2)

# Attention layer
class SingleAttention(Layer):
    def __init__(self, d_k, d_v):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.query = Dense(self.d_k, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')
        self.key = Dense(self.d_k, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')
        self.value = Dense(self.d_v, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')

    def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
        q = self.query(inputs[0])
        k = self.key(inputs[1])

        attn_weights = tf.matmul(q, k, transpose_b=True)
        attn_weights = tf.map_fn(lambda x: x/np.sqrt(self.d_k), attn_weights)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        v = self.value(inputs[2])
        attn_out = tf.matmul(attn_weights, v)
        return attn_out

# Multihead attention
class MultiAttention(Layer):
    def __init__(self, d_k, d_v, n_heads,filt_dim):
        super(MultiAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.filt_dim=filt_dim
        self.attn_heads = list()

    def build(self, input_shape):
        for n in range(self.n_heads):
            self.attn_heads.append(SingleAttention(self.d_k, self.d_v))
            self.linear = Dense(self.filt_dim, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')

    def call(self, inputs):
        attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis=-1)
        multi_linear = self.linear(concat_attn)
        return multi_linear

# Combining everything into a Transformer encoder
class TransformerEncoder(Layer):
    def __init__(self, d_k, d_v, n_heads, ff_dim, filt_dim, dropout=0.1, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.filt_dim=filt_dim
        self.attn_heads = list()
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads,self.filt_dim)
        self.attn_dropout = layers.Dropout(self.dropout_rate)
        self.attn_normalize = layers.LayerNormalization(input_shape=input_shape, epsilon=1e-6)

        self.ff_conv1D_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
        self.ff_conv1D_2 = Conv1D(filters=self.filt_dim, kernel_size=1) # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7
        self.ff_dropout = layers.Dropout(self.dropout_rate)
        self.ff_normalize = layers.LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
        attn_layer = self.attn_multi(inputs)
        attn_layer = self.attn_dropout(attn_layer)
        attn_layer = self.attn_normalize(inputs[0] + attn_layer)

        ff_layer = self.ff_conv1D_1(attn_layer)
        ff_layer = self.ff_conv1D_2(ff_layer)
        ff_layer = self.ff_dropout(ff_layer)
        ff_layer = self.ff_normalize(inputs[0] + ff_layer)
        return ff_layer

class ParticlePredictor(Model):
    def __init__(self, autoencoder, code_dim, prior_len, truth_len, d_k, d_v, n_heads=1, ff_dim=16, filt_dim=4):
        super(ParticlePredictor, self).__init__()
        self.autoencoder = autoencoder
        self.code_dim = code_dim
        self.d_k = d_k
        self.d_v = d_v
        self.prior_len = prior_len
        self.truth_len = truth_len
        self.filt_dim=filt_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim

        self.time_embedding = Time2Vector(self.prior_len)
        self.attn1 = TransformerEncoder(self.d_k, self.d_v, self.n_heads, self.ff_dim, self.filt_dim)
        self.bnorm = layers.BatchNormalization()
        self.attn2 = TransformerEncoder(self.d_k, self.d_v, self.n_heads, self.ff_dim, self.filt_dim)
        self.attn3 = TransformerEncoder(self.d_k, self.d_v, self.n_heads, self.ff_dim, self.filt_dim)
        # self.conv1d_1 = Conv1D(filters=256, kernel_size=3, strides=2, padding='same', activation='relu')
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(self.code_dim, activation='linear')

    def build(self, input_shape):
        super(ParticlePredictor, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        time_embeddings = self.time_embedding(inputs)
        x = layers.Concatenate(axis=-1)([time_embeddings, inputs])
        x = self.attn1([x,x,x])
        x = self.attn2([x,x,x])
        x = self.attn3([x,x,x])
        x = layers.Flatten()(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    @tf.function
    def train_step(self, images):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = images
        encoded_x = tf.map_fn(self.autoencoder.encoder, x)
        encoded_y = tf.map_fn(self.autoencoder.encoder, y)[:,0,:]

        with tf.GradientTape() as tape:
            y_pred = self(encoded_x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # loss = self.compiled_loss(encoded_y, y_pred, regularization_losses=self.losses)
            loss = self.compiled_loss(encoded_y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(encoded_y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

class ParticlePredictorLSTM(Model):
    def __init__(self, autoencoder, code_dim, prior_len, truth_len, filt_dim=4):
        super(ParticlePredictorLSTM, self).__init__()
        self.autoencoder = autoencoder
        self.code_dim = code_dim
        self.prior_len = prior_len
        self.truth_len = truth_len
        self.filt_dim=filt_dim

        self.time_embedding = Time2Vector(self.prior_len)
        self.lstm1 = layers.LSTM(512, return_sequences=True)
        self.lstm2 = layers.LSTM(512, return_sequences=False)
        self.dense1 = Dense(self.code_dim)
        self.dense2 = Dense(256, activation='relu')
        self.dense3 = Dense(self.code_dim)
        self.dense4 = Dense(256, activation='relu')
        self.dense5 = Dense(self.code_dim)

    def build(self, input_shape):
        super(ParticlePredictorLSTM, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        res = self.dense1(x)
        x = self.dense2(x)
        res += self.dense3(x)
        x = self.dense4(x)
        res += self.dense5(x)
        return res

    @tf.function
    def train_step(self, images):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = images
        encoded_x = tf.map_fn(self.autoencoder.encoder, x)
        encoded_y = tf.map_fn(self.autoencoder.encoder, y)[:,0,:]

        with tf.GradientTape() as tape:
            y_pred = self(encoded_x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # loss = self.compiled_loss(encoded_y, y_pred, regularization_losses=self.losses)
            loss = self.compiled_loss(encoded_y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(encoded_y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
