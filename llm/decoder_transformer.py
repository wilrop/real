import chex
import jax
import tiktoken
import optax
import modal

import jax.numpy as jnp
import flax.linen as nn

from flax import traverse_util, struct
from flax.linen import FrozenDict
from flax.training.train_state import TrainState
from typing import Optional, Generator
from functools import partial

ACT_DTYPE = jnp.bfloat16  # activations, most layer outputs
PARAM_DTYPE = jnp.bfloat16  # stored weights
COMPUTE_DTYPE = jnp.float32  # numerically sensitive math (norms/softmax/matmul logits)
LOGIT_DTYPE = jnp.float32  # final logits for stable losses

JAX_VER = "0.6.2"  # keep one version everywhere

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .env(
        {
            # JAX niceties
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85",
            "NCCL_P2P_DISABLE": "1",  # avoids occasional NCCL hangs in small single-GPU jobs
        }
    )
    # Use run_commands so we can pass pip flags cleanly.
    .run_commands(
        "python3 -m pip install --upgrade pip",
        # Install the CUDA-enabled JAX wheel from the official index of prebuilt wheels.
        # (CUDA 12 variant matches Modal's current GPU driver stack.)
        "python3 -m pip install -U 'jax[cuda12]' "
        "-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        # Core ecosystem
        "python3 -m pip install -U flax optax chex tiktoken",
        # Useful utils
        "python3 -m pip install -U numpy msgpack rich",
    )
)
app = modal.App("transformer", image=image)


class TrainStateWithRNG(TrainState):
    rng: chex.PRNGKey
    sched: optax.Schedule = struct.field(pytree_node=False)


def precompute_rope(seq_len: int, head_dim: int, base: int = 10000) -> tuple[chex.Array, chex.Array]:
    """Precompute the RoPE rotations.

    Args:
        seq_len (int): Length of the sequence.
        head_dim (int): Head dimension.
        base (int): Base frequency.

    Returns:
        tuple[chex.Array, chex.Array]: RoPE rotations.
    """
    assert head_dim % 2 == 0, "Head dimension must be even."
    half_dim = head_dim // 2

    positions = jnp.arange(seq_len, dtype=jnp.int32)[..., None]  # (seq_len, 1)
    freqs = 2 * jnp.arange(half_dim, dtype=jnp.float32) / jnp.float32(head_dim)  # (half_dim,)
    angles = positions * jnp.power(jnp.float32(base), -freqs)[None, :]  # (seq_len, half_dim)
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x: chex.Array, cos: chex.Array, sin: chex.Array, positions: chex.Array) -> chex.Array:
    """Applies the RoPE rotation to the given positions.

    Args:
        x (chex.Array): The Q or K matrix to apply the RoPE to.
        cos (chex.Array): The precomputed cosine of the RoPE rotation.
        sin (chex.Array): The precomputed sine of the RoPE rotation.
        positions (chex.Array): The positions to apply the RoPE to.

    Returns:
        chex.Array: The rotated x.
    """
    head_dim = x.shape[-1]
    assert head_dim % 2 == 0, "RoPE requires an even head_dim."
    half_dim = head_dim // 2

    # Upcast to COMPUTE_DTYPE for the trigs and rotation.
    x_c = x.astype(COMPUTE_DTYPE)
    pos_cos = cos[positions][None, None, :, :].astype(COMPUTE_DTYPE)  # (1, 1, seq_len, half_dim)
    pos_sin = sin[positions][None, None, :, :].astype(COMPUTE_DTYPE)

    x_pairs = x_c.reshape(*x_c.shape[:-1], half_dim, 2)  # (batch, num_heads, seq_len, half_dim, 2)
    x_even, x_odd = x_pairs[..., 0], x_pairs[..., 1]
    rot_even = x_even * pos_cos - x_odd * pos_sin
    rot_odd = x_even * pos_sin + x_odd * pos_cos

    rot_x = jnp.stack((rot_even, rot_odd), axis=-1).reshape(
        (*x_c.shape[:-1], head_dim))  # (batch, num_heads, seq_len, head_dim)
    return rot_x.astype(ACT_DTYPE)


def causal_mask(m: int) -> chex.Array:
    """Create a causal mask for length m.

    Args:
        m (int): Length of the causal mask.

    Returns:
        chex.Array: The causal mask.
    """
    mask = jnp.tril(jnp.ones((m, m), dtype=jnp.bool_))[None, None, :, :]  # (batch, n_heads, seq_q, seq_k)
    return mask


def mha(
        q: chex.Array,
        k: chex.Array,
        v: chex.Array,
        mask: Optional[chex.Array] = None
) -> chex.Array:
    """Multi-head scaled dot-product attention with an optional mask.

    Args:
        q (chex.Array): The Q matrix (batch, num_heads, seq_q, dim)
        k (chex.Array): The K matrix (batch, num_heads, seq_k, dim)
        v (chex.Array): The V matrix (batch, num_heads, seq_k, dim)
        mask (chex.Array): The optional mask (batch, 1, seq_q, seq_k)

    Returns:
        chex.Array: The attention values.
    """
    # Cast to compute dtypes
    q = q.astype(COMPUTE_DTYPE)
    k = k.astype(COMPUTE_DTYPE)
    v = v.astype(COMPUTE_DTYPE)

    scale = jnp.array(1.0 / jnp.sqrt(k.shape[-1]), dtype=COMPUTE_DTYPE)
    scores = (q @ k.swapaxes(-1, -2)) * scale  # (batch, num_heads, seq_q, seq_k)

    if mask is not None:
        neg_inf = jnp.finfo(scores.dtype).min  # Negative inf for this dtype
        scores = jnp.where(mask, scores, neg_inf)

    weights = jax.nn.softmax(scores, axis=-1)  # (batch, num_heads, seq_q, seq_k)
    return jnp.matmul(weights, v).astype(ACT_DTYPE)  # (batch, num_heads, seq_q, dim)


class TransformerMLP(nn.Module):
    """
    The MLP part of a transformer block with a SwiGLU activation function.
    """
    d_model: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        up = nn.Dense(
            self.hidden_dim,
            use_bias=False,
            dtype=ACT_DTYPE,
            param_dtype=PARAM_DTYPE,
        )(x).astype(COMPUTE_DTYPE)
        gate = nn.Dense(
            self.hidden_dim,
            use_bias=False,
            dtype=ACT_DTYPE,
            param_dtype=PARAM_DTYPE,
        )(x).astype(COMPUTE_DTYPE)
        x = up * jax.nn.silu(gate)
        out = nn.Dense(
            self.d_model,
            use_bias=False,
            dtype=ACT_DTYPE,
            param_dtype=PARAM_DTYPE,
        )(x)
        return out


class TransformerBlock(nn.Module):
    """Transformer block with GQA."""

    d_model: int = 512
    n_heads: int = 4
    n_kv_heads: Optional[int] = None
    mlp_hidden_dim: int = 128
    dropout_rate: float = 0.1
    norm_eps: float = 1e-5

    @nn.compact
    def __call__(
            self,
            x: chex.Array,  # (batch, seq_len, model_dim)
            cos: chex.Array,
            sin: chex.Array,
            positions: chex.Array,
            mask: Optional[chex.Array] = None,
            deterministic: bool = False,
    ) -> chex.Array:
        assert x.shape[-1] % self.n_heads == 0, "Must be divisible by n_heads."
        assert self.n_kv_heads is None or self.n_heads % self.n_kv_heads == 0, "Must be divisible by n_heads."

        n_kv_heads = self.n_kv_heads or self.n_heads
        head_dim = x.shape[-1] // self.n_heads
        kv_dim = n_kv_heads * head_dim
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Attention part
        residual = x
        x_norm = nn.RMSNorm(epsilon=self.norm_eps, dtype=COMPUTE_DTYPE, param_dtype=PARAM_DTYPE)(x).astype(ACT_DTYPE)

        q = nn.Dense(
            self.d_model,
            use_bias=False,
            dtype=ACT_DTYPE,
            param_dtype=PARAM_DTYPE,
        )(x_norm)
        q = q.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        k = nn.Dense(
            kv_dim,
            use_bias=False,
            dtype=ACT_DTYPE,
            param_dtype=PARAM_DTYPE,
        )(x_norm)
        k = k.reshape(batch_size, seq_len, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        v = nn.Dense(
            kv_dim,
            use_bias=False,
            dtype=ACT_DTYPE,
            param_dtype=PARAM_DTYPE,
        )(x_norm)
        v = v.reshape(batch_size, seq_len, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

        q = apply_rope(q, cos, sin, positions)
        k = apply_rope(k, cos, sin, positions)

        if n_kv_heads < self.n_heads:
            repeats = self.n_heads // n_kv_heads
            k = jnp.repeat(k, repeats=repeats, axis=1)
            v = jnp.repeat(v, repeats=repeats, axis=1)

        att_x = mha(q, k, v, mask)
        att_x = att_x.transpose((0, 2, 1, 3)).reshape(x.shape)
        att_x = nn.Dense(
            self.d_model,
            use_bias=False,
            dtype=ACT_DTYPE,
            param_dtype=PARAM_DTYPE,
        )(att_x)
        x = residual + nn.Dropout(rate=self.dropout_rate)(att_x, deterministic=deterministic)

        # MLP part
        residual = x
        x_norm = nn.RMSNorm(epsilon=self.norm_eps, dtype=COMPUTE_DTYPE, param_dtype=PARAM_DTYPE)(x).astype(ACT_DTYPE)
        proj_x = TransformerMLP(
            d_model=self.d_model,
            hidden_dim=self.mlp_hidden_dim,
        )(x_norm)

        return residual + nn.Dropout(rate=self.dropout_rate)(proj_x, deterministic=deterministic)


class Decoder(nn.Module):
    """Full decoder-only Transformer. Uses weight tying for the final projection."""
    vocab_size: int = 1024
    n_attn_blocks: int = 2
    d_model: int = 512
    n_heads: int = 4
    n_kv_heads: Optional[int] = None
    mlp_hidden_dim: int = 128
    dropout_rate: float = 0.1
    norm_eps: float = 1e-5

    @nn.compact
    def __call__(
            self,
            token_ids: chex.Array,
            cos: chex.Array,
            sin: chex.Array,
            positions: chex.Array,
            mask: Optional[chex.Array] = None,
            deterministic: bool = False,
    ) -> chex.Array:
        W = self.param(
            "token_embed",
            nn.initializers.normal(stddev=1.0),  # same default as nn.Embed
            (self.vocab_size, self.d_model),
            PARAM_DTYPE,
        )
        x = jnp.take(W.astype(ACT_DTYPE), token_ids, axis=0)
        for _ in range(self.n_attn_blocks):
            x = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_kv_heads=self.n_kv_heads,
                mlp_hidden_dim=self.mlp_hidden_dim,
                dropout_rate=self.dropout_rate,
                norm_eps=self.norm_eps,
            )(x, cos, sin, positions, mask, deterministic)
        x = nn.RMSNorm(epsilon=self.norm_eps, dtype=COMPUTE_DTYPE, param_dtype=PARAM_DTYPE)(x)
        logits = jnp.einsum("bld,vd->blv", x.astype(COMPUTE_DTYPE), W.astype(COMPUTE_DTYPE))
        return logits.astype(LOGIT_DTYPE)


def tokenize(text: str) -> tuple[chex.Array, int]:
    """Tokenizes a text file into a sequence of tokens.

    Args:
        file_path (str): The path to the text file to tokenize.

    Returns:
        tuple[chex.Array, int]: The tokens and the total vocab size.
    """
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text) + [enc.eot_token]
    x = jnp.array(tokens, dtype=jnp.int32)
    return x, enc.n_vocab


def make_block_batches(tokens: chex.Array, B: int, L: int) -> Generator[dict[str, chex.Array], None, None]:
    """Yield dicts with tokens (B,L), labels (B,L), positions (L,), mask (1,1,L,L).

    We drop the last batch if it is not full to avoid any padding logic.

    Args:
        tokens (chex.Array): Tokens to batch.
        B (int): Batch size.
        L (int): Length of each token.

    Returns:
        Generator[dict[str, chex.Array], None, None]: A generator that returns batches of data as a dict.
    """
    n_blocks = int(tokens.shape[0]) // L
    blocks = tokens[:n_blocks * L].reshape(n_blocks, L)
    positions = jnp.arange(L, dtype=jnp.int32)
    attn_mask = causal_mask(L)

    for i in range(0, n_blocks, B):
        if i + B > n_blocks:  # Drop if we cannot fill the batch.
            break

        x = blocks[i:i + B]
        y = x.at[:, :-1].set(x[:, 1:])  # Next token prediction - shift by 1.
        y = y.at[:, -1].set(-1)  # Ignore last prediction in loss.
        yield {"tokens": x, "labels": y, "positions": positions, "mask": attn_mask}


def xent_loss(
        logits: chex.Array,
        labels: chex.Array,
        ignore_index: int = -1
) -> chex.Array:
    """Compute the cross entropy loss for a batch of logits.

    Args:
        logits (chex.Array): Logits to compute the cross entropy loss.
        labels (chex.Array): Labels to compute the cross entropy loss.
        ignore_index (int, optional): Index to ignore when computing the cross entropy loss.

    Returns:
        chex.Array: Cross-entropy loss.
    """
    valid = labels != ignore_index
    labels_clipped = jnp.where(valid, labels, 0)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels_clipped)
    loss = loss * valid
    denom = jnp.maximum(valid.sum(), 1)
    return loss.sum() / denom


def make_optimizer(
        total_steps: int,
        warmup_steps: int,
        base_lr: float,
        weight_decay: float
) -> tuple[optax.GradientTransformationExtraArgs, optax.Schedule]:
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=0.0,
    )

    # Mask: no weight decay on norms and biases
    def decay_mask(params):
        flat = traverse_util.flatten_dict(params, sep="/")  # { "Module/Submodule/param": array, ... }
        mask_flat = {}
        for name, v in flat.items():
            # no weight decay for norms and biases
            if name.endswith("/bias") or "RMSNorm" in name or "LayerNorm" in name:
                mask_flat[name] = False
            else:
                mask_flat[name] = (v.ndim > 1)  # decay only matrices (not vectors/scalars)
        return traverse_util.unflatten_dict(mask_flat, sep="/")

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=sched,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=weight_decay,
            mask=decay_mask
        ),
    )

    return tx, sched


@partial(jax.jit, static_argnames=('deterministic',))
def train_one_step(
        state: TrainStateWithRNG,
        batch: dict,
        cos: chex.Array,
        sin: chex.Array,
        deterministic: bool,
) -> tuple[TrainStateWithRNG, dict]:
    new_rng, dropout_rng = jax.random.split(state.rng)

    def loss_fn(
            params: FrozenDict,
    ) -> tuple[chex.Array, chex.Array]:
        logits = state.apply_fn(
            {"params": params},
            batch["tokens"],
            cos,
            sin,
            batch["positions"],
            batch["mask"],
            rngs={"dropout": dropout_rng},
            deterministic=deterministic
        )
        loss = xent_loss(logits, batch["labels"])
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    metrics = {
        "loss": loss,
        "ppl": jnp.exp(jnp.minimum(loss, 50.0)),
        "lr": state.sched(state.step),
        "grad_norm": optax.global_norm(grads),
    }
    new_state = state.replace(step=state.step + 1, params=new_params, opt_state=new_opt_state, rng=new_rng)
    return new_state, metrics


def train_one_epoch(
        state: TrainStateWithRNG,
        tokens: chex.Array,
        B: int,
        L: int,
        cos: chex.Array,
        sin: chex.Array,
        log_every: int = 100,
) -> tuple[TrainStateWithRNG, dict]:
    it = make_block_batches(tokens, B, L)
    totals = {"loss": 0.0, "ppl": 0.0}

    step = 1
    for step, batch in enumerate(it, start=1):
        state, metrics = train_one_step(state, batch, cos, sin, deterministic=False)

        totals["loss"] += float(metrics["loss"])
        totals["ppl"] += float(metrics["ppl"])

        if step % log_every == 0:
            print(f"step {int(state.step):>7} | loss {float(metrics['loss']):.4f} | ppl {float(metrics['ppl']):.2f} | "
                  f"lr {float(metrics['lr']):.6f} | grad {float(metrics['grad_norm']):.2f}")

    return state, {k: v / step for k, v in totals.items()}


@app.function(
    gpu="A10",
    timeout=600,
)
def run(
        text: str,
        B: int = 8,
        L: int = 128,
        d_model: int = 512,
        n_heads: int = 4,
        n_kv_heads: int = 2,
        n_attn_blocks: int = 4,
        dropout_rate: float = 0.1,
):
    """Run the full pipeline.

    Args:
        text (str): The text to train on.
        B (int): Batch size.
        L (int): Length of the sequence.
        d_model (int): Dimensionality of the model.
        n_heads (int): Number of attention heads.
        n_kv_heads (int): Number of K/V heads (for GQA).
        n_attn_blocks (int): Number of transformer blocks.
        dropout_rate (float): Dropout rate.
    """
    # --- data ---
    tokens, vocab_size = tokenize(text)  # from our earlier helper

    # --- shapes ---
    head_dim = d_model // n_heads
    cos, sin = precompute_rope(L, head_dim)

    # --- model init ---
    model = Decoder(
        vocab_size=vocab_size,
        n_attn_blocks=n_attn_blocks,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        mlp_hidden_dim=4 * d_model // 2,  # e.g., 2*d_model for SwiGLU
        dropout_rate=dropout_rate,
    )

    rng = jax.random.PRNGKey(0)
    rng, init_rng, state_rng = jax.random.split(rng, 3)
    # Dummy batch for init
    dummy = next(make_block_batches(tokens, B, L))
    variables = model.init(init_rng, dummy["tokens"], cos, sin, dummy["positions"], dummy["mask"], False)
    params = variables["params"]

    # --- optimizer/state ---
    total_steps = 5000
    warmup_steps = 200
    base_lr = 3e-4
    weight_decay = 0.1
    tx, sched = make_optimizer(total_steps, warmup_steps, base_lr, weight_decay)

    state = TrainStateWithRNG(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=tx,
        opt_state=tx.init(params),
        rng=state_rng,
        sched=sched,
    )

    # --- train a short epoch ---
    state, metrics = train_one_epoch(state, tokens, B, L, cos, sin, log_every=50)
    print("epoch avg:", metrics)


@app.local_entrypoint()
def main():
    with open('shakespeare.txt', encoding="utf-8") as f:
        text = f.read()
    print(run.remote(text=text))


if __name__ == "__main__":
    main()
