import jax
import chex

import jax.numpy as jnp

from typing import Optional


def make_attn_mask(
        key_padding_mask: Optional[chex.Array],  # shape (B, k_len), 1=keep, 0=pad; None => all keep
        q_len: int,
        k_len: int,
        causal: bool = True,
        past_kv_len: int = 0,
        window: int | None = None,  # None => no local windowing
        first_key_abs: int = 0,
        dtype=jnp.float32,
) -> jnp.ndarray:  # shape (B, 1, q_len, k_len), additive mask
    """Generic function for generating attention masks.

    This function returns an additive attention mask, i.e., the mask is added to the attention logits.
    - 0 where attention is allowed
    - -inf where attention is not allowed.

    Args:
        key_padding_mask (Optional[chex.Array]): Padding mask for keys.
        q_len (int): Length of query vector.
        k_len (int): Length of key vector.
        causal (bool): Whether causal attention mask.
        past_kv_len (int): Past key vector length.
        window (Optional[int]): Window length.
        first_key_abs (int): Absolute index of first key.
        dtype (jnp.dtype): Data type.

    Returns:
        chex.Array: Causal attention mask.
    """
    B = key_padding_mask.shape[0] if key_padding_mask is not None else 1

    q_idx = jnp.arange(q_len)  # (q_len,)
    k_idx = jnp.arange(k_len)  # (k_len,)
    i_abs = past_kv_len + q_idx
    j_abs = first_key_abs + k_idx

    if causal:
        causal_allow = (j_abs[None, ...] <= i_abs[:, None])  # (q_len, k_len)
    else:
        causal_allow = jnp.ones((q_len, k_len), dtype=bool)

    allow = jnp.broadcast_to(causal_allow, (B, 1, q_len, k_len))

    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.astype(bool)[:, None, None, :]
        allow = jnp.logical_and(key_padding_mask, allow)

    if window is not None:
        if window <= 0:
            raise ValueError("Window must be positive.")
        left = jnp.maximum(first_key_abs, i_abs - (window - 1))
        # Allow all indices within [left_i, abs_i]
        window_allow = (j_abs[None, ...] >= left[:, None]) & (j_abs[None, ...] <= i_abs[:, None])
        window_allow = jnp.broadcast_to(window_allow, (B, 1, q_len, k_len))
        allow = jnp.logical_and(window_allow, allow)

    mask = jnp.where(allow, jnp.array(0, dtype=dtype), jnp.array(-jnp.inf, dtype=dtype))
    return mask


if __name__ == "__main__":
    def _to_bool(mask):  # convert additive mask to boolean "allowed"
        return jnp.isfinite(mask).squeeze(1)  # (B,q,k)


    # 1) Pure causal, no padding, no cache
    B, q, k = 1, 4, 4
    m = make_attn_mask(None, q, k, causal=True)
    allow = _to_bool(m)[0]  # (q,k)
    # row i should allow columns 0..i
    assert jnp.array_equal(allow, jnp.tril(jnp.ones((q, k), dtype=bool)))

    # 2) Padding only
    pad = jnp.array([[1, 1, 0, 0]], dtype=jnp.int32)  # last 2 keys are pad
    m = make_attn_mask(pad, q_len=3, k_len=4, causal=False)
    allow = _to_bool(m)[0]
    # all queries, only first 2 keys allowed
    assert jnp.array_equal(allow, jnp.array([[1, 1, 0, 0],
                                             [1, 1, 0, 0],
                                             [1, 1, 0, 0]], dtype=bool))

    # 3) Causal + padding
    pad = jnp.array([[1, 0, 1, 1]], dtype=jnp.int32)
    m = make_attn_mask(pad, q_len=4, k_len=4, causal=True)
    allow = _to_bool(m)[0]
    # future disallowed; among past/future, column 1 is always blocked by padding
    assert (allow[0, 1] == False) and (allow[2, 3] == False) and (allow[1, 2] == False)

    # 4) KV cache: past_kv_len=2, q=3, k(current)=3
    # Absolute key indices: [0,1] (cache) + [2,3,4] (current). Query i sees keys ≤ 2+i.
    m = make_attn_mask(None, q_len=3, k_len=3, causal=True, past_kv_len=2, first_key_abs=2)
    allow = _to_bool(m)[0]  # (3,3)
    # row 0: allow abs ≤2 -> within current block, only key 0 (abs=2)
    # row 1: allow abs ≤3 -> current keys 0..1
    # row 2: allow abs ≤4 -> current keys 0..2
    expected = jnp.array([[1, 0, 0],
                          [1, 1, 0],
                          [1, 1, 1]], dtype=bool)
    assert jnp.array_equal(allow, expected)

    # 5) Sliding window w=2 with no cache (q=k=5)
    m = make_attn_mask(None, q_len=5, k_len=5, causal=True, window=2)
    allow = _to_bool(m)[0]
    # each row i keeps {max(0,i-1), i}
    for i in range(5):
        assert allow[i].sum() == min(i + 1, 2)
        assert allow[i, i] == True
