from abc import ABC, abstractmethod
from typing import Callable

from loguru import logger
import torch.nn as nn
import torch as th
from nnterp import StandardizedTransformer
from tqdm import trange

from .utils import replace_with_grad, SteeringDataLoader, dummy_input


class DCT(nn.Module):
    """
    Deep Causal Transcoder

    Args:
        hidden_size (int): The dimension of the input and output of the DCT.
        num_vectors (int): The number of vectors to use for the DCT.
        steering_factor (float): R in the paper. The factor to use for the steering vectors.
        activation_fn (Callable | None): The activation function to use for the DCT. Defaults to `exp(x) - 1`.

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
        self.steering_vectors = nn.Parameter(th.zeros(num_vectors, hidden_size))

        if activation_fn is None:
            activation_fn = th.expm1
        self.activation_fn = activation_fn
        self.scales = self.init_scales()
        downstream_effects = th.randn(num_vectors, hidden_size)
        downstream_effects = downstream_effects / th.norm(
            downstream_effects, dim=1, keepdim=True
        )
        self.downstream_effects = nn.Parameter(downstream_effects)

    def similarity_penalty(self, use_abs_similarity: bool = False) -> th.Tensor:
        scale_products = th.outer(self.scales, self.scales)
        scale_products[th.eye(self.num_vectors, dtype=th.bool)] = 0
        assert scale_products.shape == (
            self.num_vectors,
            self.num_vectors,
        ), f"scale_products.shape: {scale_products.shape}, num_vectors: {self.num_vectors}"
        effect_similarities = th.einsum(
            "nd, Nd -> nN", self.downstream_effects, self.downstream_effects
        )
        assert effect_similarities.shape == (
            self.num_vectors,
            self.num_vectors,
        ), f"decoder_similarities.shape: {effect_similarities.shape}, num_vectors: {self.num_vectors}"
        steering_similarities = th.einsum(
            "nd, Nd -> nN", self.steering_vectors, self.steering_vectors
        )
        assert steering_similarities.shape == (
            self.num_vectors,
            self.num_vectors,
        ), f"steering_similarities.shape: {steering_similarities.shape}, num_vectors: {self.num_vectors}"
        effect_penalty = scale_products * effect_similarities
        steering_penalty = self.steering_factor * steering_similarities
        if use_abs_similarity:
            effect_penalty = th.abs(effect_penalty)
            steering_penalty = th.abs(steering_penalty)
        steering_penalty = self.activation_fn(steering_penalty)
        return (effect_penalty * steering_penalty).sum()

    def init_scales(self) -> th.Tensor:
        logger.warning("Proper scale initialization not implemented yet")
        return th.ones(self.num_vectors)

    def init_steering_vectors(self, steered_diffs: th.Tensor):
        """
        Initialize the steering vectors to be the grad of (u_i, delta_i)

        Args:
            steering_effects (th.Tensor): The steering effects (n_steering_vectors, hidden_size)
        """
        projected_effects_sum = th.einsum(
            "nd, nd ->", steered_diffs, self.downstream_effects
        )
        projected_effects_sum.backward()
        replace_with_grad(self.steering_vectors, normalize=True)

    def causal_importance(
        self, steered_diffs: th.Tensor, use_abs_importance: bool = False
    ) -> th.Tensor:
        projected_diffs = th.einsum(
            "nd, nd -> n", steered_diffs, self.downstream_effects
        )
        importance = (self.scales * projected_diffs).sum()
        if use_abs_importance:
            importance = th.abs(importance)
        return importance

    def objective(self, steered_diffs: th.Tensor) -> th.Tensor:
        return self.causal_importance(steered_diffs) + self.similarity_penalty()

    def soft_normalization(self, log_grad_norms: th.Tensor):
        raise NotImplementedError("Not implemented")  # todo

    def step(self, beta: float | None = None):
        log_grad_norms = th.log(th.norm(self.steering_vectors.grad, dim=-1))
        replace_with_grad(self.steering_vectors, beta=beta, normalize=True)
        replace_with_grad(self.downstream_effects, beta=beta, normalize=True)
        self.soft_normalization(log_grad_norms)


class SteeringEffectExtractor(ABC):
    def __init__(self, steer_layer: int, target_layer: int, batch_size: int):
        self.steer_layer = steer_layer
        self.target_layer = target_layer
        self.batch_size = batch_size

    @abstractmethod
    def get_steering_effect(
        self, prompts: str | list[str], steering_vectors: th.Tensor
    ) -> th.Tensor:
        """
        Given a set of prompts and steering vectors, return a tensor of shape (n_steering_vectors, hidden_size)
        where each element is the difference between the target layer activations with and without steering.

        Args:
            prompts (str | list[str]): The prompts to steer.
            steering_vectors (th.Tensor): The steering vectors to use (n_steering_vectors, hidden_size)

        Returns:
            th.Tensor: The steering effect, averaged over tokens. (n_steering_vectors, hidden_size)
        """
        pass

    def __call__(
        self, prompts: str | list[str], steering_vectors: th.Tensor
    ) -> th.Tensor:
        return self.get_steering_effect(prompts, steering_vectors)


class SteerSingleModel(SteeringEffectExtractor):

    def __init__(
        self,
        model: StandardizedTransformer,
        steer_layer: int,
        target_layer: int,
        batch_size: int,
    ):
        super().__init__(steer_layer, target_layer, batch_size)
        self.model = model
        self.model.requires_grad_(False)

    def get_steering_effect(
        self, prompts: str | list[str], steering_vectors: th.Tensor
    ) -> th.Tensor:
        """
        Given a set of prompts and steering vectors, return a tensor of shape (n_steering_vectors, hidden_size)
        where each element is the difference between the target layer activations with and without steering `model`.

        Args:
            prompts (str | list[str]): The prompts to steer.
            steering_vectors (th.Tensor): The steering vectors to use (n_steering_vectors, hidden_size)

        Returns:
            th.Tensor: The steering effect, averaged over tokens. (n_steering_vectors, hidden_size)
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        num_prompts = len(prompts)
        n_steering_vectors = steering_vectors.shape[0]
        with self.model.trace(prompts) as tracer:
            source_acts = self.model.layers_output[self.steer_layer].cpu().save()
            source_attention_mask = self.model.attention_mask.cpu().save()
            target_acts = self.model.layers_output[self.target_layer].save()
            tracer.stop()
        target_acts = (
            target_acts.unsqueeze(0)
            .expand(n_steering_vectors, -1, -1, -1)
            .reshape(n_steering_vectors * num_prompts, -1, -1)
        )
        dataloader = SteeringDataLoader(
            source_acts, source_attention_mask, steering_vectors, self.batch_size
        )
        diffs = []
        i = 0
        for steered_activations, attention_mask in dataloader:
            with self.model.trace(dummy_input(attention_mask)) as tracer:
                self.model.skip_layers(0, self.steer_layer, steered_activations)
                steered_target_acts = self.model.layers_output[self.target_layer].save()
                tracer.stop()
            diffs.append(
                (steered_target_acts - target_acts[i : i + len(steered_activations)])
            )
            i += len(steered_activations)
            del steered_target_acts
        diffs = th.cat(diffs, dim=0).reshape(steering_vectors.shape[0], -1, -1, -1)
        return diffs[:, source_attention_mask].mean(dim=1)


class SoftOrthGradIteration:

    def __init__(
        self,
        num_iterations: int,
        num_steering_vectors: int,
        hidden_size: int,
        prompts: str | list[str],
        steering_delta_comp: SteeringEffectExtractor,
        beta: float | None = None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]
        self.prompts = prompts
        self.num_iterations = num_iterations
        self.num_steering_vectors = num_steering_vectors
        self.steering_delta_comp = steering_delta_comp
        self.dct = DCT(hidden_size, num_steering_vectors, self.init_steering_scales())
        self.dct.init_steering_vectors(
            self.steering_delta_comp(prompts, self.dct.steering_vectors)
        )
        self.beta = beta

    def init_steering_scales(self) -> float:
        logger.warning("Proper scale initialization not implemented yet")
        return 1.0

    def fit(self, num_epochs: int):
        for _ in trange(num_epochs):
            steering_deltas = self.steering_delta_comp(
                self.prompts, self.dct.steering_vectors
            )
            self.dct.objective(steering_deltas).backward()
