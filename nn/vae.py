from typing import List, Callable, Any, Tuple

import jax
import chex
import optax

import matplotlib.pyplot as plt
import jax.numpy as jnp
import flax.linen as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Encoder(nn.Module):
    """The encoder of a variational autoencoder."""
    out_dim: int
    hidden_dim: List[int]
    activation: Callable[..., Any] = nn.relu
    mu_activation: Callable[..., Any] = nn.relu
    logvar_activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x) -> Tuple[chex.Array, chex.Array]:
        """Run the encoder."""
        for i, hidden_dim in enumerate(self.hidden_dim):
            x = nn.Dense(hidden_dim)(x)
            x = self.activation(x)
        mu = nn.Dense(self.out_dim)(x)  # The mean of the normal distribution.
        mu = self.mu_activation(mu)
        logvar = nn.Dense(self.out_dim)(x)  # The log variance of the normal distribution.
        logvar = self.logvar_activation(logvar)
        return mu, logvar


class Decoder(nn.Module):
    """The decoder of a variational autoencoder."""
    out_dim: int
    hidden_dim: List[int]
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, z) -> chex.Array:
        """Run the encoder."""
        for i, hidden_dim in enumerate(self.hidden_dim):
            z = nn.Dense(hidden_dim)(z)
            z = self.activation(z)
        x = nn.Dense(self.out_dim)(z)
        return x


class VAE(nn.Module):
    """A variational autoencoder."""
    enc_out_dim: int
    dec_out_dim: int
    enc_hidden_dim: List[int]
    dec_hidden_dim: List[int]
    enc_activation: Callable[..., Any] = nn.relu
    dec_activation: Callable[..., Any] = nn.relu
    mu_activation: Callable[..., Any] = nn.relu
    logvar_activation: Callable[..., Any] = nn.relu

    def setup(self):
        self.encoder = Encoder(self.enc_out_dim, self.enc_hidden_dim, self.enc_activation, self.mu_activation,
                               self.logvar_activation)
        self.decoder = Decoder(self.dec_out_dim, self.dec_hidden_dim, self.dec_activation)

    def __call__(self, x, key) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Run the variational autoencoder."""
        mu, logvar = self.encoder(x)
        z = gaussian_sample(mu, logvar, key)
        logits = self.decoder(z)
        return logits, mu, logvar

    def sample(self, rng, num_samples, latent_dim) -> chex.Array:
        """Sample images from the decoder."""
        img_rng, z_rng = jax.random.split(rng)
        z = jax.random.normal(z_rng, (num_samples, latent_dim))
        logits = self.decoder(z)

        # Convert the logits to probabilities.
        sampled_images = jax.random.bernoulli(img_rng, jnp.logaddexp(0., logits))
        return sampled_images


def kl_gaussian(mu: chex.Array, logvar: chex.Array) -> chex.Array:
    """Calculate the KL divergence between a normal distribution and a standard normal distribution.

    Note:
        KL(N(mu, sigma^2) || N(0, 1)) = -1/2 * sum(1 + log(sigma^2) - mu^2 - sigma^2).
        See https://mbernste.github.io/posts/vae/ for a derivation of this formula.

    Args:
        mu (Array): The mean of the normal distribution.
        logvar (Array): The log variance of the normal distribution.

    Returns:
        Array: The KL divergence between the normal distribution and the standard normal distribution.
    """
    return -0.5 * jnp.sum(1 + logvar - mu ** 2 - jnp.exp(logvar))


def gaussian_sample(mu: chex.Array, logvar: chex.Array, rng: chex.PRNGKey) -> chex.Array:
    """Sample from a normal distribution.

    Args:
        mu (Array): The mean of the normal distribution.
        logvar (Array): The log variance of the normal distribution.
        rng (Array): The random number generator.

    Returns:
        Array: A sample from the normal distribution.
    """
    std = jnp.exp(0.5 * logvar)  # This is equivalent to taking the square root of the variance.
    eps = jax.random.normal(rng, mu.shape)  # Sample from a standard normal distribution.
    return mu + std * eps  # Reparameterization trick.


def bernoulli_logpdf(logits: chex.Array, data: chex.Array) -> chex.Array:
    """Calculate the log probability of the data under a Bernoulli distribution.

    Note:
        This is equivalent to the negative binary cross entropy loss. The reason we can use this in this case is because
        the data is binary (i.e. 0 or 1). If the data was continuous, we would have to use the MSE for the
        reconstruction loss.

    Note:
        This is implemented with a logaddexp trick to avoid numerical instability. This is equivalent to the binary
        cross entropy loss when combined with the sigmoid function. https://stackoverflow.com/a/29863846/9058050

    Args:
        logits (Array): The logits of the Bernoulli distribution.
        data (Array): The data.

    Returns:
        Array: The log probability of the data under the Bernoulli distribution.
    """
    return - jnp.sum(jnp.logaddexp(0., jnp.where(data, -1., 1.) * logits))


def elbo(key: chex.PRNGKey, vae: VAE, params: chex.Array, x: chex.Array) -> chex.Array:
    """ Compute the evidence lower bound (ELBO) for a variational autoencoder.

    Note:
        The elbo is essentially -logprob - kl_divergence. The logprob is the reconstruction loss. The kl_divergence is
        the regularization term. The regularization term is necessary to prevent the encoder from learning a trivial
        solution.

    Args:
        x (Array): The data.
        logits (Array): The logits of the Bernoulli distribution.
        mu (Array): The mean of the normal distribution.
        logvar (Array): The log variance of the normal distribution.

    Returns:
        Array: The evidence lower bound.
    """
    logits, mu, logvar = vae.apply(params, x, key)
    bce_loss = bernoulli_logpdf(logits, x)
    kl_loss = kl_gaussian(mu, logvar)
    return bce_loss - kl_loss


def image_grid(nrow, ncol, imagevecs, imshape, title=None):
    """Reshape a stack of image vectors into an image grid for plotting.
    Note:
        Code taken from: https://github.com/google/jax/blob/main/examples/mnist_vae.py
    """
    images = iter(imagevecs.reshape((-1,) + imshape))
    images = jnp.vstack([jnp.hstack([next(images).T for _ in range(ncol)][::-1]) for _ in range(nrow)]).T
    # Plot image grid.
    plt.figure(figsize=(2 * nrow, 2 * ncol))
    plt.axis('off')
    plt.imshow(images, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.show()


def run_epoch(train_keys: chex.Array,
              train_images: chex.Array,
              num_batches: int,
              vae: VAE,
              optimizer: callable,
              state: Tuple[chex.Array, chex.Array]) -> Tuple[chex.Array, chex.Array]:
    """Run an epoch on the training data.

    Note:
        This uses jax.lax.fori_loop to compile the training loop.

    Args:
        train_keys (Array): The random number generators for each batch.
        train_images (Array): The training images.
        vae (VAE): The variational autoencoder.
        optimizer (optax.GradientTransformation): The optimizer.
        state (tuple): The state of the optimizer.
    """

    def body_fun(i, state):
        params, opt_state = state
        batch = train_images[i]
        key = train_keys[i]

        def loss_f(params):
            return - elbo(key, vae, params, batch) / batch.shape[0]

        grads = jax.grad(loss_f)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return jax.lax.fori_loop(0, num_batches, body_fun, state)


def evaluate(rng, vae, state, test_data):
    """Evaluate the model on the test data."""
    params, _ = state
    test_elbo = elbo(rng, vae, params, test_data) / test_data.shape[0]
    return test_elbo


def batch_data(data, batch_size):
    """Utility to batch data."""
    num_samples = data.shape[0]
    num_batches = num_samples // batch_size
    data = data[:num_batches * batch_size]  # Drop the last batch if it is incomplete.
    data = data.reshape((num_batches, batch_size, -1))
    return data


def train_vae():
    """Train a variational autoencoder on the MNIST dataset.

    Note:
        Code adapted from https://github.com/google/jax/blob/main/examples/mnist_vae.py
    """
    step_size = 0.001
    num_epochs = 100
    batch_size = 32
    train_split = 0.8  # fraction of data to be used for training
    latent_dim = 10  # Output dim for the encoder.
    img_dim = 28 * 28  # Output dim for the decoder.
    encoder_dims = [512, 512]
    decoder_dims = [512, 512]
    nrow, ncol = 3, 3  # sampled image grid size
    plot_every = 5

    key = jax.random.PRNGKey(1)  # fixed prng key for evaluation
    key, subkey = jax.random.split(key)

    # Download MNIST and split in train and test images.
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    data = jnp.asarray(datasets.MNIST('../data', train=True, download=True, transform=transform).data)
    shuffled_data = jax.random.permutation(subkey, data)
    train_images = shuffled_data[:int(train_split * len(data))]
    test_images = shuffled_data[int(train_split * len(data)):]

    # Make images black and white.
    train_images = train_images[..., None] > 0.5
    test_images = test_images[..., None] > 0.5

    # Show images in a grid.
    image_grid(nrow, ncol, train_images[:nrow * ncol], imshape=(28, 28), title='Training images')

    # Divide the train and test images in batches.
    train_images = batch_data(train_images, batch_size)
    test_images = batch_data(test_images, batch_size)

    # Setup the VAE.
    vae = VAE(enc_out_dim=latent_dim, dec_out_dim=img_dim, enc_hidden_dim=encoder_dims, dec_hidden_dim=decoder_dims,
              enc_activation=nn.relu, dec_activation=nn.relu, mu_activation=nn.softplus, logvar_activation=nn.softplus)

    # Initialize the model.
    key, init_key, test_key = jax.random.split(key, 3)
    data = jnp.zeros((batch_size, img_dim))
    vae_params = vae.init(init_key, data, init_key)

    # Initialize the optimizer.
    optimizer = optax.adam(step_size)
    opt_state = optimizer.init(vae_params)

    state = (vae_params, opt_state)
    keys = jax.random.split(key, num_epochs)

    # Train the model.
    for epoch, key in enumerate(keys):
        print(f'Epoch {epoch}')
        train_key, test_key, image_key = jax.random.split(key, 3)
        batch_keys = jax.random.split(train_key, train_images.shape[0])
        state = run_epoch(batch_keys, train_images, train_images.shape[0], vae, optimizer, state)
        test_elbo = evaluate(test_key, vae, state, test_images)
        print(f'Iteration {epoch} - Loss {test_elbo}')

        if epoch % plot_every == 0:
            sampled_images = vae.apply(state[0], image_key, nrow * ncol, latent_dim, method=vae.sample)
            image_grid(nrow, ncol, sampled_images, imshape=(28, 28), title=f'Epoch {epoch}')

    # Sample from the trained model.
    sampled_images = vae.apply(state[0], test_key, nrow * ncol, latent_dim, method=vae.sample)
    image_grid(nrow, ncol, sampled_images, imshape=(28, 28), title=f'Trained model')


if __name__ == '__main__':
    train_vae()
