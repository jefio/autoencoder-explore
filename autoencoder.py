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
    def __init__(self, models, datasets, exp_name):
        self.models = models
        self.datasets = datasets
        self.exp_name = exp_name

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
        filename = expanduser('~/plot/exp_{}_reconstruction_test_data.png'.format(self.exp_name))
        plt.savefig(filename)
        plt.clf()
        mpl.rcParams.update(mpl.rcParamsDefault)


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

        filename = expanduser('~/plot/exp_{}_dream.png'.format(self.exp_name))
        plt.savefig(filename)
        plt.clf()
        mpl.rcParams.update(mpl.rcParamsDefault)


    def plot_latent_vectors_2d(self):
        x_test_encoded = self.models['encoder'].predict(
            self.datasets['x_test'], batch_size=256)
        plt.figure(figsize=(6, 6))
        plt.scatter(
            x_test_encoded[:, 0], x_test_encoded[:, 1], c=self.datasets['y_test'])
        plt.colorbar()
        filename = expanduser('~/plot/exp_{}_latent_vectors.png'.format(self.exp_name))
        plt.savefig(filename)
        plt.clf()


    def plot_histogram_latent_activities(self, encoded_imgs):
        vec = encoded_imgs.flatten()
        weights = 100. * np.ones_like(vec) / float(len(vec))
        plt.hist(vec, weights=weights, ec='black')
        plt.xlabel('activity')
        plt.ylabel('percentage')
        plt.grid(True)
        filename = expanduser('~/plot/exp_{}_histogram_latent_components.png'.format(self.exp_name))
        plt.savefig(filename)
        plt.clf()


    def plot_histogram_latent_weights(self):
        vec = np.concatenate([
            x.flatten()
            for x in self.models['encoder'].layers[-1].get_weights()
        ])
        weights = 100. * np.ones_like(vec) / float(len(vec))
        plt.hist(vec, weights=weights, ec='black')
        plt.xlabel('weight')
        plt.ylabel('percentage')
        plt.grid(True)
        filename = expanduser('~/plot/exp_{}_histogram_latent_weights.png'.format(self.exp_name))
        plt.savefig(filename)
        plt.clf()


    def fit_and_plot(self, epochs):
        self.models['autoencoder'].fit(
            self.datasets['x_train'],
            self.datasets['x_train'],
            epochs=epochs,
            batch_size=256,
            shuffle=True,
            validation_data=(self.datasets['x_test'], self.datasets['x_test'])
        )

        encoded_imgs = self.models['encoder'].predict(self.datasets['x_test'])
        decoded_imgs = self.models['decoder'].predict(encoded_imgs)

        self.plot_reconstructions(decoded_imgs)
        self.plot_dreams()
        self.plot_latent_vectors_2d()
        filename = expanduser('~/plot/exp_{}_model.png'.format(self.exp_name))
        plot_model(
            self.models['autoencoder'], show_shapes=True, to_file=filename)
        self.plot_histogram_latent_activities(encoded_imgs)
        self.plot_histogram_latent_weights()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sizes', type=int, nargs='+', default=[784, 128, 64, 32])
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-a', '--activation', type=str, default='relu')
    parser.add_argument('--reg-kernel-method', type=str, default=None)
    parser.add_argument('--reg-kernel-coefs', type=float, nargs='+', default=[0])
    parser.add_argument('--reg-activity-method', type=str, default=None)
    parser.add_argument('--reg-activity-coefs', type=float, nargs='+', default=[0])
    args = parser.parse_args()

    exp_idx = 0
    for activity_coef in args.reg_activity_coefs:
        for kernel_coef in args.reg_kernel_coefs:
            bottleneck_config = {
                'activation': args.activation,
                'regularization': {
                    'activity': {
                        'method': args.reg_activity_method,
                        'coef': activity_coef},
                    'kernel': {
                        'method': args.reg_kernel_method,
                        'coef': kernel_coef}
                }
            }
            models = get_autoencoder_models(args.sizes, bottleneck_config)
            aep = AutoEncoderPlot(
                models, get_mnist_datasets(), string.ascii_uppercase[exp_idx])
            aep.fit_and_plot(args.epochs)
            exp_idx += 1


if __name__ == '__main__':
    main()
