"""
Models
"""
from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import regularizers
from keras import backend as K
from keras import metrics


def get_regularizer(regularizer_config):
    method, coef = next(iter(regularizer_config.items()))
    assert method in (None, 'l1', 'l2')
    if method is None:
        regularizer = None
    elif method == 'l1':
        regularizer = regularizers.l1(coef)
    elif method == 'l2':
        regularizer = regularizers.l2(coef)
    return regularizer


def get_autoencoder_models(sizes, bottleneck_config):
    input_img = Input(shape=(sizes[0],))
    encoded = input_img
    for size in sizes[1:-1]:
        encoded = Dense(size, activation='relu')(encoded)
    # bottleneck layer
    encoded = Dense(
        sizes[-1],
        activation=bottleneck_config['activation'],
        activity_regularizer=get_regularizer(bottleneck_config['activity_regularizer']),
        kernel_regularizer=get_regularizer(bottleneck_config['kernel_regularizer'])
    )(encoded)

    decoded = encoded
    for size in sizes[::-1][1:-1]:
        decoded = Dense(size, activation='relu')(decoded)
    decoded = Dense(sizes[0], activation='sigmoid')(decoded)

    # models
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)

    encoded_input = Input(shape=(sizes[-1],))
    decoded = encoded_input
    for decoder_layer in autoencoder.layers[len(sizes):]:
        decoded = decoder_layer(decoded)
    decoder = Model(encoded_input, decoded)

    autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return {
        'autoencoder': autoencoder,
        'encoder': encoder,
        'decoder': decoder
    }


def get_vae_models(sizes, bottleneck_config):
    """
    Based on the Keras VAE example script.
    """
    #
    epsilon_std = 1.0
    original_dim, intermediate_dim, latent_dim = sizes

    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim, activation=bottleneck_config['activation'])(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)


    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x, x_decoded_mean):
            rec_loss = original_dim * metrics.mean_squared_error(x, x_decoded_mean)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(rec_loss + bottleneck_config['activity_regularizer']['vae'] * kl_loss)

        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean = inputs[1]
            loss = self.vae_loss(x, x_decoded_mean)
            self.add_loss(loss, inputs=inputs)
            # We won't actually use the output.
            return x

    y = CustomVariationalLayer()([x, x_decoded_mean])
    vae = Model(x, y)
    vae.compile(optimizer='rmsprop', loss=None)

    # build a model to project inputs on the latent space
    if bottleneck_config['stochastic_encoder']:
        encoder = Model(x, z)
    else:
        encoder = Model(x, z_mean)

    # build a digit generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    decoder = Model(decoder_input, _x_decoded_mean)

    return {
        'autoencoder': vae,
        'encoder': encoder,
        'decoder': decoder
    }
