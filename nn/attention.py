import haiku as hk
import jax
import jax.numpy as jnp


def scaled_dot_product(q, k, v, mask=None):
    """Perform scaled dot product attention.

    Args:
        q (ndarray): The queries. This has shape [batch, ..., heads, length, embed_dim].
        k (ndarray): The keys. This has shape [batch, ..., heads, length, embed_dim].
        v (ndarray): The values. This has shape [batch, ..., heads, length, embed_dim].
        mask (ndarray, optional): Attention mask. Must be broadcastable to [batch, ..., heads, length, length].

    Returns:
        ndarray: The scaled dot product attention.
    """
    d_k = q.shape[-1]
    # We want to do a matrix multiplication between the embeddings in q and k.
    # Therefore, we have to swap the last two axes and let the broadcasting do the rest for us.
    attention_logits = jnp.matmul(q, k.swapaxes(-2, -1))

    # Scale the attention logits down to account for the change in variance.
    attention_logits = attention_logits / jnp.sqrt(d_k)

    # Mask out logits with a small number. We don't set this to zero to leave gradient flows.
    if mask is not None:
        attention_logits = jnp.where(mask == 0, -9e15, attention_logits)

    # Do a softmax over the last axis.
    attention_weights = jax.nn.softmax(attention_logits, axis=-1)
    return jnp.matmul(attention_weights, v)


class MultiHeadScaledDotProductAttention(hk.Module):
    """
    A multihead scaled dot product attention module.
    """

    def __init__(self, d_model: int, num_heads: int, name=None):
        super().__init__(name=name)

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.lin_projs = [hk.Linear(self.d_model) for _ in range(4)]

    def __call__(self, q, k, v, mask=None):
        """Compute the attention for a given batch of queries, keys and values.

        Args:
            q (ndarray): The queries. This has shape [batch, ..., length, d_model].
            k (ndarray): The keys. This has shape [batch, ..., length, d_model].
            v (ndarray): The values. This has shape [batch, ..., length, d_model].
            mask (ndarray, optional): Attention mask. Must be broadcastable to [batch, ..., length, length].

        Returns:
            ndarray: The attended output. This has shape [batch, ..., length, embed_dim]
        """
        batch_size, seq_length, d_model = q.shape  # (batch, length, d_k)

        q, k, v = [lin_p(t).reshape(batch_size, -1, self.num_heads, self.d_k).swapaxes(1, 2) for lin_p, t in
                   zip(self.lin_projs, (q, k, v))]  # (batch, heads, length, d_k)

        if mask is not None:
            mask = jnp.expand_dims(mask, 1)  # Expand the mask so it gets broadcasted correctly.

        values = scaled_dot_product(q, k, v, mask=mask)  # Perform the scaled dot product.
        values = values.transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_length, d_model)  # Concat the heads.
        return self.lin_projs[-1](values)
