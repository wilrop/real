import jax
import chex

import jax.numpy as jnp


def stable_softmax(x: chex.Array, axis: int = -1) -> chex.Array:
    """Numerically stable softmax function.

    It takes the following precautions:
        - Subtract max to avoid overflows
        - Operates in float32 for precision and returns the result in the original dtype.
        - if the full row is -inf, it returns zeros

    Args:
        x (chex.Array): Input array.
        axis (int, optional): Axis to stable softmax.

    Returns:
        chex.Array: A softmax over the input.
    """
    x32 = x.astype(jnp.float32)
    m = jnp.max(x32, axis=axis, keepdims=True)
    m_safe = jnp.where(jnp.isfinite(m), m, jnp.array(0.0, dtype=x32.dtype))
    shifted = x32 - m_safe
    exps = jnp.exp(shifted)
    denom = jnp.sum(exps, axis=axis, keepdims=True)
    out = jnp.where(denom > 0, exps / denom, jnp.zeros_like(exps))
    return out.astype(x.dtype)


def apply_top_k(x: chex.Array, top_k: int) -> chex.Array:
    """Apply top-k selection to x. Elements outside are masked to -inf.

    Args:
        x (chex.Array): Input array.
        top_k (int): Number of top-k selections to apply.

    Returns:
        chex.Array: Output array.
    """
    if top_k is None or top_k <= 0:  # Don't do top_k.
        return x

    if top_k >= x.shape[-1]:  # Select everything.
        return x

    if not jnp.issubdtype(x.dtype, jnp.inexact):
        x = x.astype(jnp.float32)

    _, idx = jax.lax.top_k(x, top_k)

    keep = jnp.zeros_like(x, dtype=bool)

    leading = x.shape[:-1]  # Get the leading dims.
    grids = jnp.meshgrid(*[jnp.arange(n, dtype=jnp.int32) for n in leading], indexing="ij")
    grids = [g[..., None] for g in grids]
    keep = keep.at[(*grids, idx)].set(True)
    out = jnp.where(keep, x, jnp.full_like(x, -jnp.inf))
    return out


def apply_top_p(probs: chex.Array, top_p: float) -> chex.Array:
    """Apply top-p selection to probs.

    Args:
        probs (chex.Array): Input array of probabilities.
        top_p (float): Top-p selection probability.

    Returns:
        chex.Array: Output array.
    """
    if top_p is None or top_p >= 1.0:  # Select everything.
        return probs

    if top_p <= 0.0:
        return jnp.zeros_like(probs)

    sort_idx = jnp.argsort(-probs, axis=-1)  # (..., V)
    sorted_probs = jnp.take_along_axis(probs, sort_idx, axis=-1)  # (..., V)

    cumsum = jnp.cumsum(sorted_probs, axis=-1)  # (..., V)
    # First index where cumsum >= top_p, then +1 to include it
    keep_counts = (cumsum < top_p).sum(axis=-1) + 1  # (...,)

    positions = jnp.arange(probs.shape[-1], dtype=jnp.int32)  # (V,)
    ranks = jnp.empty_like(sort_idx)
    leading = probs.shape[:-1]
    grids = jnp.meshgrid(*[jnp.arange(n, dtype=jnp.int32) for n in leading], indexing="ij")
    grids = [g[..., None] for g in grids]
    ranks = ranks.at[(*grids, sort_idx)].set(positions)  # Inverse map back from sorted to original
    keep = ranks < keep_counts[..., None]
    out = jnp.where(keep, probs, jnp.zeros_like(probs))
    return out


def sample_next_token(
        logits: chex.Array,  # Shape (V,) or (B,V)
        key: chex.PRNGKey,
        temperature: float = 1.0,
        top_k: int = 0,  # 0 => disabled
        top_p: float = 1.0,  # 1.0 => disabled
        eos_token_id: int | None = None,
        allow_eos: bool = True,
):
    if not allow_eos and eos_token_id is None:
        raise ValueError("`eos_token_id` must be set if `allow_eos` is False.")

    if top_k < 0:
        raise ValueError("`top_k` must be non-negative.")

    if top_p < 0:
        raise ValueError("`top_p` must be non-negative.")

    if not allow_eos:
        logits = logits.at[..., eos_token_id].set(-jnp.inf)

    if temperature == 0.0 or top_k == 1 or top_p == 0.0:  # Greedy possibilities.
        return jnp.argmax(logits, axis=-1)

    logits = logits / temperature  # Apply temperature consistently.
    logits = apply_top_k(logits, top_k=top_k)  # Apply top-k selection.
    probs = stable_softmax(logits)
    kept = apply_top_p(probs, top_p=top_p)  # Apply top-p (nucleus) selection.
    s = jnp.sum(kept, axis=-1)
    probs = jnp.where(s > 0, kept / s, kept)
    samples = jax.random.categorical(key, jnp.log(probs + 1e-45), axis=-1)
    return samples


def _call_sampler(logits, key, **kwargs):
    """Helper to call sample_next_token while handling key splitting consistently."""
    # Many JAX APIs expect you to split the key before each use
    key, subkey = jax.random.split(key)
    out = sample_next_token(logits, key=subkey, **kwargs)  # or rng=subkey
    if isinstance(out, tuple) and len(out) == 2:
        tokens, maybe_new_key = out
        # Prefer the sampler's returned key if it provides one
        return tokens, maybe_new_key
    else:
        # If your sampler returns only tokens, keep advancing our own key
        return out, key


# DISCLAIMER: Test are vibecoded.
# -----------------------
# Tests
# -----------------------
def test_greedy_temperature_zero():
    key = jax.random.PRNGKey(0)
    logits = jnp.array([0.1, 2.0, -0.5])
    t, key = _call_sampler(logits, key, temperature=0.0)
    # Expect pure argmax = 1
    assert int(t) == 1


def test_top_k_masks_rest():
    key = jax.random.PRNGKey(0)
    counts = {0: 0, 1: 0, 2: 0}
    logits = jnp.array([1.0, 0.9, -2.0])
    for _ in range(1000):
        tok, key = _call_sampler(logits, key, top_k=2)
        tok = int(tok)
        counts[tok] = counts.get(tok, 0) + 1
    assert counts[2] == 0  # class 2 should be pruned entirely


def test_top_p_prefix_mass():
    key = jax.random.PRNGKey(1)
    # Provide logits that correspond to probs [0.60, 0.25, 0.10, 0.05]
    probs = jnp.array([0.60, 0.25, 0.10, 0.05], dtype=jnp.float32)
    logits = jnp.log(probs)
    # With p=0.7, only {0,1} should remain after nucleus truncation.
    hits = set()
    for _ in range(200):
        tok, key = _call_sampler(logits, key, top_p=0.7)
        hits.add(int(tok))
    assert hits.issubset({0, 1})


def test_eos_disallowed():
    key = jax.random.PRNGKey(2)
    # token 2 is EOS; with allow_eos=False it must never be sampled
    logits = jnp.array([0.0, 0.0, 5.0])
    for _ in range(100):
        tok, key = _call_sampler(logits, key, eos_token_id=2, allow_eos=False)
        assert int(tok) in (0, 1)


def test_batch_shape_and_greedy():
    key = jax.random.PRNGKey(3)
    # (B=2, V=2)
    batch_logits = jnp.stack([jnp.array([0.0, 3.0]), jnp.array([4.0, 1.0])], axis=0)
    t, key = _call_sampler(batch_logits, key, temperature=0.0)
    t = jnp.asarray(t)
    assert t.shape == (2,)
    assert jnp.all(t == jnp.array([1, 0]))


# -------------
# Runner
# -------------
if __name__ == "__main__":
    test_greedy_temperature_zero()
    test_top_k_masks_rest()
    test_top_p_prefix_mass()
    test_eos_disallowed()
    test_batch_shape_and_greedy()
    print("All JAX tests passed.")
