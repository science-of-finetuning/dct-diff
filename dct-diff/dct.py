import torch.nn as nn
import torch as th
from typing import Callable
from loguru import logger


class DCT(nn.Module):
    """
    Deep Causal Transcoder



    Args:
        hidden_size (int): The dimension of the input and output of the DCT.
        num_vectors (int): The number of vectors to use for the DCT.
        activation_fn (Callable | None): The activation function to use for the DCT. Defaults to `exp(x) - 1`.
        steering_factor (float): R in the paper. The factor to use for the steering vectors.

    Attributes:
        steering_vectors (nn.Linear): v_i in the paper
        activation_fn (Callable): The activation function to use.
        downstream_effects (nn.Linear): u_i in the paper
    """

    def __init__(
        self,
        hidden_size: int,
        num_vectors: int,
        steering_factor: float,
        activation_fn: Callable | None = None,
    ):
        # todo: alphas > 0 ?

        super().__init__()
        self.hidden_size = hidden_size
        self.num_vectors = num_vectors
        self.steering_factor = steering_factor
        self.steering_vectors = nn.Linear(hidden_size, num_vectors)
        if activation_fn is None:
            activation_fn = th.expm1
        self.activation_fn = activation_fn
        self.scales = self.init_scales()
        self.downstream_effects = nn.Linear(num_vectors, hidden_size)

    def forward(
        self, x: th.Tensor, return_activations: bool = False
    ) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
        activations = self.activation_fn(self.steering_vectors(x))
        out = self.downstream_effects(activations)
        if return_activations:
            return out, activations
        return out

    def similarity_penalty(self) -> th.Tensor:
        scale_products = th.outer(self.scales, self.scales)
        downstream_effects = (
            self.downstream_effects.weight.T
        )  # shape: (num_vectors, hidden_size)
        steering_vectors = (
            self.steering_vectors.weight
        )  # shape: (num_vectors, hidden_size)
        scale_products[th.eye(self.num_vectors, dtype=th.bool)] = 0
        assert scale_products.shape == (
            self.num_vectors,
            self.num_vectors,
        ), f"scale_products.shape: {scale_products.shape}, num_vectors: {self.num_vectors}"
        effect_similarities = th.einsum(
            "nd, Nd -> nN", downstream_effects, downstream_effects
        )
        assert effect_similarities.shape == (
            self.num_vectors,
            self.num_vectors,
        ), f"decoder_similarities.shape: {effect_similarities.shape}, num_vectors: {self.num_vectors}"
        steering_similarities = th.einsum(
            "nd, Nd -> nN", steering_vectors, steering_vectors
        )
        assert steering_similarities.shape == (
            self.num_vectors,
            self.num_vectors,
        ), f"steering_similarities.shape: {steering_similarities.shape}, num_vectors: {self.num_vectors}"
        effect_penalty = scale_products * effect_similarities
        steering_penalty = self.activation_fn(
            self.steering_factor * steering_similarities
        )
        return (effect_penalty * steering_penalty).sum()


    def init_scales(self) -> th.Tensor:
        logger.warning("Proper scale initialization not implemented yet")
        return th.ones(self.num_vectors)
