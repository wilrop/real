import argparse
import os
import pickle
import random
import time
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm

from nn.attention import MultiHeadScaledDotProductAttention


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", action="store_true", default=False, help="Track the experiments using wandb")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory where to save the logs")
    parser.add_argument("--wandb-project-name", type=str, default="real", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument("--seed", type=int, default=42, help="The seed for random number generation.")
    parser.add_argument("--batch_size", type=int, default=128, help="The batch size.")
    parser.add_argument("--num_epochs", type=int, default=30, help="The number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="The learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="The weight decay.")
    parser.add_argument("--embed_dim", type=int, default=256, help="The embedding dimension.")
    parser.add_argument("--hidden_dim", type=int, default=512, help="The hidden dimension.")
    parser.add_argument("--num_heads", type=int, default=8, help="The number of heads.")
    parser.add_argument("--num_layers", type=int, default=6, help="The number of layers.")
    parser.add_argument("--patch_size", type=int, default=4, help="The patch size.")
    parser.add_argument("--num_patches", type=int, default=64, help="The number of patches.")
    parser.add_argument("--num_classes", type=int, default=10, help="The number of classes.")
    parser.add_argument("--p_dropout", type=float, default=0.2, help="The dropout probability.")
    parser.add_argument("--eval_steps", type=int, default=2, help="The number of steps between evaluations.")
    args = parser.parse_args()
    return args


def numpy_collate(batch):
    """Collate function for numpy arrays.

    Args:
        batch (Any): A list of tuples (x, y) where x is a numpy array and y is a label.

    Returns:
        array_like: A stack of bach elements.
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def image_to_numpy(img):
    """Converts an image to a numpy array."""
    data_means = np.array([0.49139968, 0.48215841, 0.44653091])
    data_std = np.array([0.24703223, 0.24348513, 0.26158784])
    img = np.array(img, dtype=np.float32)
    img = (img / 255. - data_means) / data_std
    return img


def get_cifar_dataloaders(dataset_path="data", batch_size=128, seed=42):
    """Get the CIFAR10 train and test dataloaders."""
    # Transformations applied on each image => bring them into a numpy array

    test_transform = image_to_numpy

    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                                          transforms.Lambda(image_to_numpy)
                                          ])

    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    train_dataset = CIFAR10(root=dataset_path, train=True, download=True, transform=train_transform)
    val_dataset = CIFAR10(root=dataset_path, train=True, download=True, transform=test_transform)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000],
                                                 generator=torch.Generator().manual_seed(seed))
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000],
                                               generator=torch.Generator().manual_seed(seed))

    # Loading the test set
    test_set = CIFAR10(root=dataset_path, train=False, transform=test_transform, download=True)

    train_loader = data.DataLoader(train_set,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   collate_fn=numpy_collate,
                                   num_workers=4,
                                   persistent_workers=True)
    val_loader = data.DataLoader(val_set,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 collate_fn=numpy_collate,
                                 num_workers=4,
                                 persistent_workers=True)
    test_loader = data.DataLoader(test_set,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  drop_last=False,
                                  collate_fn=numpy_collate,
                                  num_workers=4,
                                  persistent_workers=True)

    return train_loader, val_loader, test_loader


def img_to_patch(x, patch_size, flatten_channels=True):
    """Convert an image array into an array of patches.

    Args:
        x (ndarray): The input image as an array. This has shape [batch, height, width, channel].
        patch_size (int): The width and height size of the patches that will be cut from the image.
        flatten_channels (bool, optional): Whether to flatten the channels in the image. (Default value = True)

    Returns:
        ndarray: The image as a sequence of patches.
    """
    batch, height, width, channel = x.shape
    assert height % patch_size == 0 and width % patch_size == 0
    x = x.reshape(batch, height // patch_size, patch_size, width // patch_size, patch_size, channel)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # Set the patches in the right order.
    x = x.reshape(batch, -1, *x.shape[3:])
    if flatten_channels:
        x = x.reshape(batch, x.shape[1], -1)
    return x


class MLP(hk.Module):
    """
    The MLP module used in the attention block.
    """

    def __init__(self, dim1, dim2, p_dropout, name=None):
        super().__init__(name=name)
        self.lin1 = hk.Linear(dim1)
        self.lin2 = hk.Linear(dim2)
        self.p_dropout = p_dropout

    def __call__(self, x, is_train=True):
        x = self.lin1(x)
        x = jax.nn.gelu(x)
        if is_train:
            hk.dropout(hk.next_rng_key(), self.p_dropout, x)
        x = self.lin2(x)
        return x


class AttentionBlock(hk.Module):
    """
    An attention block using pre-layer normalisation.
    """

    def __init__(self, d_model, num_heads, hidden_dim, p_dropout, name=None):
        super().__init__(name=name)
        self.mha = MultiHeadScaledDotProductAttention(d_model, num_heads)
        self.ffn = MLP(hidden_dim, d_model, p_dropout)
        self.layer_norm1 = hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True, create_offset=True)
        self.layer_norm2 = hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True, create_offset=True)
        self.p_dropout = p_dropout

    def __call__(self, x, is_train=True):
        x = self.layer_norm1(x)
        x = self.mha(x, x, x)
        if is_train:
            x = x + hk.dropout(hk.next_rng_key(), self.p_dropout, x)
        x = self.layer_norm2(x)
        x = self.ffn(x, is_train=is_train)

        if is_train:
            x = x + hk.dropout(hk.next_rng_key(), self.p_dropout, x)
        return x


class VisionTransformer(hk.Module):
    """
    A vision transformer for image classification [1].

    References:
        .. [1] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N.
            (2020). An image is worth 16x16 words: Transformers for image recognition at scale.
            arXiv preprint arXiv:2010.11929.
    """

    def __init__(self, embed_dim, num_heads, hidden_dim, num_layers, num_classes, patch_size, num_patches, p_dropout,
                 name=None):
        super().__init__(name=name)
        self.p_dropout = p_dropout
        self.patch_size = patch_size
        self.to_patch_embedding = hk.Linear(embed_dim)
        self.transformer = [AttentionBlock(embed_dim, num_heads, hidden_dim, p_dropout) for _ in range(num_layers)]
        self.mlp_head = hk.Sequential([hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True, create_offset=True),
                                       hk.Linear(num_classes)])
        # Parameters/Embeddings
        self.cls_token = hk.get_parameter("cls_token", (1, 1, embed_dim), init=hk.initializers.RandomNormal(stddev=1.0))
        self.pos_embedding = hk.get_parameter("pos_embedding", (1, 1 + num_patches, embed_dim),
                                              init=hk.initializers.RandomNormal(stddev=1.0))

    def __call__(self, x, is_train=True):
        # Preprocess the input data.
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.to_patch_embedding(x)

        # Add CLS token and positional encoding.
        cls_token = self.cls_token.repeat(B, axis=0)  # Repeat a class token across the entire batch.
        x = jnp.concatenate([cls_token, x], axis=1)  # Concatenate the class token to the front.
        x = x + self.pos_embedding[:, :T + 1]  # Set the positional encodings for every patch.

        # Apply the transformer.
        if is_train:
            x = x + hk.dropout(hk.next_rng_key(), self.p_dropout, x)

        for block in self.transformer:
            x = block(x, is_train=is_train)

        # Perform the classification.
        cls = x[:, 0]
        out = self.mlp_head(cls)
        return out


class TrainingState(NamedTuple):
    """
    The training state is a named tuple containing the model parameters and the optimizer state.
    """
    params: hk.Params  # The model parameters.
    opt_state: optax.OptState  # The optimizer state.
    rng: jnp.DeviceArray  # The RNG key.


@hk.transform
def vit_model(x, is_train=True, **model_hparams):
    """The vision transformer model."""
    vit = VisionTransformer(**model_hparams)
    return vit(x, is_train=is_train)


@jax.jit
def train_step(state, batch):
    rng, new_rng = jax.random.split(state.rng)  # Split the rng key.

    # Compute loss and gradients.
    loss_fn = lambda params: calculate_loss(params, batch, rng=state.rng, is_train=True)
    (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    # Perform an optimization step.
    updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)

    # Compose the new state.
    new_state = TrainingState(params, opt_state, new_rng)
    return new_state, loss, acc


@jax.jit
def eval_step(state, batch):
    """Evaluate the model on a batch of data."""
    loss, acc = calculate_loss(state.params, batch, is_train=False)
    return loss, acc


def calculate_loss(params, batch, rng=None, is_train=True):
    """Calculate the loss for a batch of data."""
    imgs, labels = batch
    logits = vit_model.apply(params=params, rng=rng, x=imgs, is_train=is_train, **model_hparams)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    acc = (logits.argmax(axis=-1) == labels).mean()
    return loss, acc


def save_model(state, log_dir, step):
    """Save the model parameters to a file.

    Args:
        state (TrainingState): The training state.
        log_dir (str): The directory to save the model to.
        step (int): The current training step.
    """
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(log_dir, f"model_{step}.pkl")
    with open(model_path, "wb") as fp:
        pickle.dump(state, fp)


def load_model(log_dir):
    """Load the model parameters from a file.

    Args:
        log_dir (str): The directory to load the model from.

    Returns:
        TrainingState: The training state.
    """
    files = [f for f in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, f))]

    def sort_files(filename):
        try:
            return int(filename.split("_")[-1])
        except:
            return 0

    checkpoint_file = sorted(files, key=sort_files)[-1]

    with open(os.path.join(log_dir, checkpoint_file), "rb") as fp:
        state = pickle.load(fp)

    return state


def eval_model(data_loader, split="val", epoch_idx=0):
    """Evaluate the model on a specific split."""
    total_loss = jax.numpy.zeros(1)
    total_acc = jax.numpy.zeros(1)

    if split == "val":
        desc = "Validation"
    else:
        desc = "Test"

    for batch in tqdm(data_loader, desc=desc, leave=False):  # Loop over the data.
        loss, acc = eval_step(state, batch)
        total_loss += loss
        total_acc += acc

    avg_loss = total_loss.item() / len(data_loader)
    avg_acc = total_acc.item() / len(data_loader)

    writer.add_scalar(f"{split}/loss", avg_loss, epoch_idx)
    writer.add_scalar(f"{split}/acc", avg_acc, epoch_idx)
    return avg_loss, avg_acc


def classification_gallery(images, true_labels, predictions, num_rows, num_columns, title="Test images"):
    """Plot a gallery of images with their predictions."""
    data_means = np.array([0.49139968, 0.48215841, 0.44653091])
    data_std = np.array([0.24703223, 0.24348513, 0.26158784])
    resize = transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC)

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 6))
    fig.suptitle(title)

    for image, true_label, prediction, ax in zip(images, true_labels, predictions, axes.flat):
        image = (image * data_std + data_means)
        image = (image * 255).astype(np.uint8)
        ax.imshow(resize(Image.fromarray(image)))
        ax.set_title(f"Label: {true_label} - Prediction: {prediction}")
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    args = parse_args()

    # Setup logging.
    run_name = f"VIT_{args.seed}_{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Setup seeding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # Load the dataset.
    train_loader, val_loader, test_loader = get_cifar_dataloaders(seed=args.seed, batch_size=args.batch_size)

    # Initialise the model.
    rng, init_rng = jax.random.split(key)
    model_hparams = {"embed_dim": args.embed_dim,
                     "num_heads": args.num_heads,
                     "hidden_dim": args.hidden_dim,
                     "num_layers": args.num_layers,
                     "num_classes": args.num_classes,
                     "patch_size": args.patch_size,
                     "num_patches": args.num_patches,
                     "p_dropout": args.p_dropout}
    init_params = vit_model.init(rng=init_rng,
                                 x=next(iter(train_loader))[0],
                                 is_train=True,
                                 **model_hparams)

    # Initialise the optimizer.
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=args.learning_rate,
        boundaries_and_scales=
        {int(args.num_epochs * len(train_loader) * 0.6): 0.1,
         int(args.num_epochs * len(train_loader) * 0.85): 0.1}
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients at norm 1
        optax.adamw(lr_schedule, weight_decay=args.weight_decay)
    )

    init_opt_state = optimizer.init(init_params)

    # Initialise the training state.
    state = TrainingState(init_params, init_opt_state, rng)

    # Training loop.
    best_eval = 0.0
    for epoch_idx in tqdm(range(1, args.num_epochs + 1), desc="Epoch"):
        total_loss = jax.numpy.zeros(1)
        total_acc = jax.numpy.zeros(1)

        for batch in tqdm(train_loader, desc="Training", leave=False):  # Loop over the training data.
            state, loss, acc = train_step(state, batch)
            total_loss += loss
            total_acc += acc

        writer.add_scalar("train/loss", total_loss.item() / len(train_loader), epoch_idx)
        writer.add_scalar("train/acc", total_acc.item() / len(train_loader), epoch_idx)

        if epoch_idx % args.eval_steps == 0:  # Perform an evaluation.
            avg_eval_loss, avg_eval_acc = eval_model(val_loader, split="val", epoch_idx=epoch_idx)
            if avg_eval_acc > best_eval:
                best_eval = avg_eval_acc
                save_model(state, args.log_dir, epoch_idx)

    # Load the best model and evaluate on the test set.
    state = load_model(args.log_dir)
    avg_test_loss, avg_test_acc = eval_model(test_loader, split="test", epoch_idx=0)
    print(f"Test accuracy: {avg_test_acc}")

    # Plot the results of the first 9 images in the test set.
    images, labels = next(iter(test_loader))[:9]
    logits = vit_model.apply(params=state.params, rng=None, x=images, is_train=False, **model_hparams)
    predictions = logits.argmax(axis=-1)
    classification_gallery(images, labels, predictions, 3, 3, title="Test images")

    writer.close()
    if args.track:
        wandb.finish()
