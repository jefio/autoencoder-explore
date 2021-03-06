# autoencoder-explore

Fit autoencoders on the MNIST dataset and plot properties of the learned models.

## Dependencies

Python 3.5, Keras 2.0.6, Numpy, Pandas, and Matplotlib.

## Examples

Fit a simple autoencoder
```
python fitplot.py --folder ~/plot_ae --reg-activity-method l2 --reg-activity-coefs 1e-5
```

Fit a variational autoencoder
```
python fitplot.py --folder ~/plot_vae --vae 1 --reg-activity-method vae --reg-activity-coefs 0.1
```
