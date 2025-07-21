from abc import ABC, abstractmethod

import torch as th
from nnterp import StandardizedTransformer

from .utils import SteeringDataLoader, dummy_input


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
