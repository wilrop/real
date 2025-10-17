import jax
import chex

import jax.numpy as jnp

from typing import Optional


def causal_attention(
        q: chex.Array,
        k: chex.Array,
        v: chex.Array,
        cache: Optional[dict[str, chex.Array]] = None,
        mask: Optional[chex.Array] = None,
) -> tuple[chex.Array, dict[str, chex.Array]]:
    """Applies causal attention over a batch of queries and keys.

    In train mode, the Q, K and V matrices have dimension [B, H, T, D].
    In decode mode, the Q has dimension [B, H, 1, D] and K and V come from the cache.

    Args:
        q (chex.Array): Query vector.
        k (chex.Array): Key vector.
        v (chex.Array): Value vector.
        cache (Optional[dict[str, chex.Array]], optional): Cache to use. Defaults to None.
        mask (Optional[chex.Array], optional): Padding mask with shape [B, T]. Defaults to None.
        causal (bool, optional): Whether to use causal attention. Defaults to True.

    Returns:
        tuple[chex.Array, dict[str, chex.Array]]: Attention output and updated cache dict.

    """
    B, H, Tq, D = q.shape

    # Extract keys and values from the cache.
    if (Tq == 1) and (cache is not None):
        k_cache = cache['k']
        v_cache = cache['v']
        k = jnp.concatenate([k_cache, k], axis=-2)  # [B,H,Tc+1,D]
        v = jnp.concatenate([v_cache, v], axis=-2)
    Tk = k.shape[-2]

    # Compute attention scores.
    scale = 1 / jnp.sqrt(D)
    k_mat = jnp.swapaxes(k, -1, -2)
    scores = jnp.matmul(q, k_mat) * scale  # [B, H, Tq, Tk]

    # Rectangular causal: allow attending only to t' <= t
    q_offset = Tk - Tq
    q_pos = jnp.arange(Tq) + q_offset  # [Tq]
    k_pos = jnp.arange(Tk)  # [Tk]
    causal = (q_pos[:, None] >= k_pos[None, :])  # [Tq, Tk]
    causal = causal[None, None, :, :]  # [1,1,Tq,Tk]

    if mask is not None:  # mask: [B, Tk], True = valid
        key_mask = mask[:, None, None, :]  # [B,1,1,Tk]
        attn_mask = jnp.logical_and(causal, key_mask)  # [B,1,Tq,Tk]
    else:
        attn_mask = causal

    # Mask the scores.
    neg_inf = jnp.finfo(scores.dtype).min
    add = jnp.where(attn_mask, 0, neg_inf)
    scores += add

    # Compute final output and update cache
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(weights, v)
    if (Tq == 1) and (cache is not None):
        new_cache = {'k': k, 'v': v}
    else:
        new_cache = cache
    return out, new_cache


############### VIBE CODED ##################

# ---------- helpers ----------
def rand_qkv(B=2, H=3, T=5, D=4, key=jax.random.PRNGKey(0)):
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (B, H, T, D))
    k = jax.random.normal(k2, (B, H, T, D))
    v = jax.random.normal(k3, (B, H, T, D))
    return q, k, v


def run_train(q, k, v, mask=None):
    out, new_cache = causal_attention(q, k, v, cache=None, mask=mask)
    return out, new_cache


def run_decode_step(q_step, k_step, v_step, cache, mask=None):
    out, new_cache = causal_attention(q_step, k_step, v_step, cache=cache, mask=mask)
    return out, new_cache


# ---------- 1) Shape & cache passthrough in train ----------
def test_train_shapes_and_cache_passthrough():
    B, H, T, D = 2, 3, 5, 4
    q, k, v = rand_qkv(B, H, T, D)
    out, new_cache = run_train(q, k, v, mask=None)
    chex.assert_shape(out, (B, H, T, D))
    assert new_cache is None or new_cache == {}  # your function returns cache as-is
    print("✓ train shapes & cache passthrough")


# ---------- 2) Mask equivalence: masking == slicing ----------
def test_padding_mask_equivalence():
    # Keep contract: Tq = Tk = Tv in train
    B, H, T, D = 2, 2, 6, 8
    q, k, v = rand_qkv(B, H, T, D, key=jax.random.PRNGKey(1))

    Tk_valid = 4
    # mask keeps the first Tk_valid keys
    mask = jnp.zeros((B, T), dtype=bool).at[:, :Tk_valid].set(True)

    # Masked full sequence
    out_mask, _ = run_train(q, k, v, mask=mask)  # [B,H,T,D]

    # Sliced prefix for q,k,v (same lengths)
    out_slice, _ = run_train(q[:, :, :Tk_valid, :],
                             k[:, :, :Tk_valid, :],
                             v[:, :, :Tk_valid, :],
                             mask=None)  # [B,H,Tk_valid,D]

    # Compare only the first Tk_valid positions
    chex.assert_trees_all_close(out_mask[:, :, :Tk_valid, :], out_slice, atol=1e-5, rtol=1e-5)
    print("✓ padding mask ≈ slicing equivalence (train contract respected)")


# ---------- 3) Decode equivalence: last train step == decode with cache ----------
def test_decode_equals_last_train_step():
    B, H, T, D = 1, 2, 7, 16
    q, k, v = rand_qkv(B, H, T, D, key=jax.random.PRNGKey(2))

    # Full train output
    out_train, _ = run_train(q, k, v, mask=None)
    last_train = out_train[:, :, -1:, :]  # [B,H,1,D]

    # Build cache with first T-1
    cache = {
        "k": k[:, :, :T - 1, :],
        "v": v[:, :, :T - 1, :],
    }

    # Decode step with the final token
    q_step = q[:, :, -1:, :]
    k_step = k[:, :, -1:, :]
    v_step = v[:, :, -1:, :]
    out_decode, new_cache = run_decode_step(q_step, k_step, v_step, cache=cache, mask=None)

    chex.assert_shape(out_decode, (B, H, 1, D))
    chex.assert_trees_all_close(out_decode, last_train, atol=1e-5, rtol=1e-5)
    # Cache grew to T
    chex.assert_shape(new_cache["k"], (B, H, T, D))
    chex.assert_shape(new_cache["v"], (B, H, T, D))
    print("✓ decode equals last train step & cache grows")


# ---------- 4) Causality sanity: future keys don’t change earlier outputs ----------
def test_future_changes_do_not_affect_past_outputs():
    B, H, T, D = 1, 1, 6, 8
    key = jax.random.PRNGKey(3)
    q, k, v = rand_qkv(B, H, T, D, key=key)

    # Baseline
    out0, _ = run_train(q, k, v, mask=None)

    # Perturb *future* keys/values at positions >= t0+1 and check earlier outputs unchanged
    t0 = 2  # we will check positions <= t0
    k2 = k.at[:, :, t0 + 1:, :].add(10.0)  # big perturbation in the future
    v2 = v.at[:, :, t0 + 1:, :].add(10.0)

    out1, _ = run_train(q, k2, v2, mask=None)

    chex.assert_trees_all_close(out0[:, :, :t0 + 1, :], out1[:, :, :t0 + 1, :], atol=1e-5, rtol=1e-5)
    print("✓ past outputs invariant to future K/V changes (causality)")


# ---------- 5) Decode mask length handling ----------
def test_decode_with_padding_mask_length_Tk():
    B, H, T, D = 2, 2, 5, 8
    q, k, v = rand_qkv(B, H, T, D, key=jax.random.PRNGKey(4))

    # Build cache with first T-1
    cache = {"k": k[:, :, :T - 1, :], "v": v[:, :, :T - 1, :]}

    # One-step decode
    q_step = q[:, :, -1:, :]
    k_step = k[:, :, -1:, :]
    v_step = v[:, :, -1:, :]

    # After concat Tk == T; create a mask of length T that pads the last two keys
    mask = jnp.ones((B, T), dtype=bool).at[:, -2:].set(False)

    out_dec, new_cache = run_decode_step(q_step, k_step, v_step, cache=cache, mask=mask)
    chex.assert_shape(out_dec, (B, H, 1, D))
    chex.assert_shape(new_cache["k"], (B, H, T, D))
    print("✓ decode works with padding mask of length Tk")


# ---------- 6) Batch/heads broadcasting sanity ----------
def test_batch_head_broadcasting():
    B, H, T, D = 3, 4, 5, 6
    q, k, v = rand_qkv(B, H, T, D, key=jax.random.PRNGKey(5))
    out, _ = run_train(q, k, v, mask=jnp.ones((B, T), dtype=bool))
    chex.assert_shape(out, (B, H, T, D))
    print("✓ batch/head broadcasting with mask")


# ---------- Run all ----------
def run_all_sanity_checks():
    test_train_shapes_and_cache_passthrough()
    test_padding_mask_equivalence()
    test_decode_equals_last_train_step()
    test_future_changes_do_not_affect_past_outputs()
    test_decode_with_padding_mask_length_Tk()
    test_batch_head_broadcasting()
    print("All sanity checks passed.")


if __name__ == "__main__":
    run_all_sanity_checks()
