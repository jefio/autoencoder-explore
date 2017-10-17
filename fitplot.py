"""
Fit and plot autoencoders
"""
import argparse
import subprocess
import os

from keras.utils import plot_model
from keras.datasets import mnist
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from models import get_autoencoder_models, get_vae_models


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


    def plot_latent_vectors_2d(self, train=True):
        if train:
            x_mat = self.datasets['x_train']
            y_vec = self.datasets['y_train']
            filename = '{}/exp_{}_latent_vectors_train.png'.format(self.folder, self.exp_name)
        else:
            x_mat = self.datasets['x_test']
            y_vec = self.datasets['y_test']
            filename = '{}/exp_{}_latent_vectors_test.png'.format(self.folder, self.exp_name)

        x_test_encoded = self.models['encoder'].predict(
            x_mat, batch_size=256)
        plt.figure(figsize=(6, 6))
        plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_vec, cmap='tab10', alpha=0.8)
        plt.colorbar()
        plt.title(self.get_title())
        self.save_close(filename)


    def plot_latent_activities(self, encoded_imgs):
        ser = pd.Series(np.sort(np.abs(encoded_imgs.flatten())))
        ser.plot(kind='line')
        plt.xlabel('Latent unit')
        plt.ylabel('Absolute activity')
        plt.title(self.get_title())
        filename = '{}/exp_{}_latent_activities.png'.format(self.folder, self.exp_name)
        self.save_close(filename)

        ser = pd.Series(np.sort(np.mean(np.abs(encoded_imgs), axis=0)))
        ser.plot(kind='line')
        plt.xlabel('Latent unit')
        plt.ylabel('Mean absolute activity')
        plt.title(self.get_title())
        plt.gca().set_ylim([0, 1])
        filename = '{}/exp_{}_mean_latent_activities.png'.format(self.folder, self.exp_name)
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
        self.plot_latent_vectors_2d(train=True)
        self.plot_latent_vectors_2d(train=False)
        filename = '{}/exp_{}_model.png'.format(self.folder, self.exp_name)
        plot_model(
            self.models['autoencoder'], show_shapes=True, to_file=filename)
        self.plot_latent_activities(encoded_train_imgs)
        try:
            self.plot_latent_weights()
        except ValueError:
            print('plot_latent_weights not available with VAE, skipping')


def write_gif(n_files, folder, title):
    start = ['convert', '-loop', '0', '-delay', '100']
    infiles = ["{}/exp_{}_{}.png".format(folder, idx, title)
               for idx in range(n_files)]
    if all(os.path.isfile(filename) for filename in infiles):
        command = start + infiles + ["{}/{}.gif".format(folder, title)]
        subprocess.call(command)


def write_gifs(n_files, folder):
    titles = ['latent_activities', 'mean_latent_activities', 'latent_weights',
              'latent_vectors_test', 'latent_vectors_train']
    for title in titles:
        write_gif(n_files, folder, title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[784, 256, 2])
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-v', '--vae', type=int, default=0, choices=[0, 1])
    parser.add_argument('-a', '--activation', type=str, default=None)
    parser.add_argument('-rkm', '--reg-kernel-method', type=str, default=None,
                        choices=[None, 'l1', 'l2'])
    parser.add_argument('-rkc', '--reg-kernel-coefs', type=float, nargs='+', default=[0])
    parser.add_argument('-ram', '--reg-activity-method', type=str, default=None,
                        choices=[None, 'l1', 'l2', 'vae'])
    parser.add_argument('-rac', '--reg-activity-coefs', type=float, nargs='+', default=[0])
    # specific to VAE
    parser.add_argument('--stochastic-encoder', type=int, default=0, choices=[0, 1])
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
                'kernel_regularizer': {args.reg_kernel_method: kernel_coef},
                'stochastic_encoder': args.stochastic_encoder
            }
            if args.vae:
                models = get_vae_models(args.sizes, bottleneck_config)
            else:
                models = get_autoencoder_models(args.sizes, bottleneck_config)
            aep = AutoEncoderPlot(
                bottleneck_config, models, get_mnist_datasets(),
                exp_idx, args.folder)
            aep.fit_and_plot(args.epochs)
            exp_idx += 1
    write_gifs(exp_idx, args.folder)


if __name__ == '__main__':
    main()
