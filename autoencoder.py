"""
Fit and plot autoencoders
"""
import argparse
import subprocess
import os

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.utils import plot_model
from keras.datasets import mnist
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


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


def get_mnist_datasets():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    return {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test
    }


class AutoEncoderPlot(object):
    def __init__(self, bottleneck_config, models, datasets, exp_name, folder):
        self.bottleneck_config = bottleneck_config
        self.models = models
        self.datasets = datasets
        self.exp_name = exp_name
        self.folder = folder


    @staticmethod
    def save_close(filename):
        plt.savefig(filename)
        plt.close()
        mpl.rcParams.update(mpl.rcParamsDefault)


    def plot_reconstructions(self, decoded_imgs):
        n = 10  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(self.datasets['x_test'][i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        filename = '{}/exp_{}_reconstruction_test_data.png'.format(self.folder, self.exp_name)
        self.save_close(filename)


    def plot_dreams(self):
        _, latent_dim = self.models['decoder'].layers[0].input_shape
        n = 10  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(min(latent_dim, 20)):
            latent_basis_vector = np.eye(latent_dim)[i]
            latent_basis_vector.shape = (1, latent_dim)
            decoded_img = self.models['decoder'].predict(latent_basis_vector)
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(decoded_img.reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        filename = '{}/exp_{}_dream.png'.format(self.folder, self.exp_name)
        self.save_close(filename)


    def plot_latent_vectors_2d(self):
        x_test_encoded = self.models['encoder'].predict(
            self.datasets['x_test'], batch_size=256)
        plt.figure(figsize=(6, 6))
        plt.scatter(
            x_test_encoded[:, 0], x_test_encoded[:, 1], c=self.datasets['y_test'])
        plt.colorbar()
        filename = '{}/exp_{}_latent_vectors.png'.format(self.folder, self.exp_name)
        self.save_close(filename)


    def plot_latent_activities(self, encoded_imgs):
        ser = pd.Series(np.sort(np.abs(encoded_imgs.flatten())))
        ser.plot(kind='line')
        plt.xlabel('Latent unit')
        plt.ylabel('Absolute activity')
        plt.title(self.get_title())
        filename = '{}/exp_{}_latent_activities.png'.format(self.folder, self.exp_name)
        self.save_close(filename)


    def get_title(self):
        title = ""
        method, coef = next(iter(self.bottleneck_config['kernel_regularizer'].items()))
        if coef > 0:
            title += "{} - C_kernel={}".format(method, coef)
        method, coef = next(iter(self.bottleneck_config['activity_regularizer'].items()))
        if coef > 0:
            title += "{} - C_activity={}".format(method, coef)
        return title


    def plot_latent_weights(self):
        weights = np.concatenate([
            x.flatten()
            for x in self.models['encoder'].layers[-1].get_weights()
        ])
        ser = pd.Series(np.sort(np.abs(weights)))
        ser.plot(kind='line')
        plt.xlabel('Edge of latent unit')
        plt.ylabel('Absolute weight')
        plt.title(self.get_title())
        filename = '{}/exp_{}_latent_weights.png'.format(self.folder, self.exp_name)
        self.save_close(filename)


    def fit_and_plot(self, epochs):
        self.models['autoencoder'].fit(
            self.datasets['x_train'],
            self.datasets['x_train'],
            epochs=epochs,
            batch_size=256,
            shuffle=True,
            validation_data=(self.datasets['x_test'], self.datasets['x_test'])
        )

        encoded_train_imgs = self.models['encoder'].predict(self.datasets['x_train'])
        encoded_test_imgs = self.models['encoder'].predict(self.datasets['x_test'])
        decoded_test_imgs = self.models['decoder'].predict(encoded_test_imgs)

        self.plot_reconstructions(decoded_test_imgs)
        self.plot_dreams()
        self.plot_latent_vectors_2d()
        filename = '{}/exp_{}_model.png'.format(self.folder, self.exp_name)
        plot_model(
            self.models['autoencoder'], show_shapes=True, to_file=filename)
        self.plot_latent_activities(encoded_train_imgs)
        self.plot_latent_weights()


def write_gifs(n_files, folder):
    start = ['convert', '-loop', '0', '-delay', '100']
    command1 = start + [
        "{}/exp_{}_latent_activities.png".format(folder, idx)
        for idx in range(n_files)
    ]
    command1 += ["{}/latent_activities.gif".format(folder)]
    subprocess.call(command1)

    command2 = start + [
        "{}/exp_{}_latent_weights.png".format(folder, idx)
        for idx in range(n_files)
    ]
    command2 += ["{}/latent_weights.gif".format(folder)]
    subprocess.call(command2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, required=True)
    parser.add_argument('-s', '--sizes', type=int, nargs='+', default=[784, 128, 64, 32])
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-a', '--activation', type=str, default='relu')
    parser.add_argument('-rkm', '--reg-kernel-method', type=str, default=None)
    parser.add_argument('-rkc', '--reg-kernel-coefs', type=float, nargs='+', default=[0])
    parser.add_argument('-ram', '--reg-activity-method', type=str, default=None)
    parser.add_argument('-rac', '--reg-activity-coefs', type=float, nargs='+', default=[0])
    args = parser.parse_args()

    if os.path.exists(args.folder):
        print("folder={}, overwriting".format(args.folder))
    else:
        os.makedirs(args.folder)

    exp_idx = 0
    for activity_coef in args.reg_activity_coefs:
        for kernel_coef in args.reg_kernel_coefs:
            bottleneck_config = {
                'activation': args.activation,
                'activity_regularizer': {args.reg_activity_method: activity_coef},
                'kernel_regularizer': {args.reg_kernel_method: kernel_coef}
            }
            models = get_autoencoder_models(args.sizes, bottleneck_config)
            aep = AutoEncoderPlot(
                bottleneck_config, models, get_mnist_datasets(),
                exp_idx, args.folder)
            aep.fit_and_plot(args.epochs)
            exp_idx += 1
    write_gifs(exp_idx, args.folder)


if __name__ == '__main__':
    main()
