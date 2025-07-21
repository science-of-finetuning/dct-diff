from abc import ABC, abstractmethod

import torch as th
import torch.nn.functional as F
from nnterp import StandardizedTransformer
from loguru import logger
from .utils import SteeringDataLoader, dummy_input
from scipy.optimize import root_scalar

class SteeringEffectExtractor(ABC):
    def __init__(self, steer_layer: int, target_layer: int, batch_size: int):
        self.steer_layer = steer_layer
        self.target_layer = target_layer
        self.batch_size = batch_size
        self.steering_factor = None

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

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        pass

    def __call__(
        self, prompts: str | list[str], steering_vectors: th.Tensor
    ) -> th.Tensor:
        return self.get_steering_effect(prompts, steering_vectors)

    @abstractmethod
    def get_calibrated_steering_factor(self, prompts: str | list[str]) -> float:
        """
        Get the calibrated steering factor for the given prompts.
        """
        pass

    def init_steering_factor(self, prompts: str | list[str]):
        """
        Initialize the steering factor for the given prompts.
        """
        self.steering_factor = self.get_calibrated_steering_factor(prompts)
        return self.steering_factor


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

    @property
    def hidden_size(self) -> int:
        return self.model.hidden_size

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
            seq_len = self.model.input_size[1].save()
            source_attention_mask = self.model.attention_mask.bool().cpu().save()
            source_acts = self.model.layers_output[self.steer_layer].cpu().save()
            target_acts = self.model.layers_output[self.target_layer].save()
            tracer.stop()
        target_acts = target_acts.unsqueeze(0).expand(n_steering_vectors, -1, -1, -1)
        target_acts = target_acts.reshape(n_steering_vectors * num_prompts, seq_len, -1)
        assert target_acts.shape == (
            n_steering_vectors * num_prompts,
            seq_len,
            self.model.hidden_size,
        ), f"source_acts.shape: {source_acts.shape}, n_steering_vectors: {n_steering_vectors}, num_prompts: {num_prompts}, seq_len: {seq_len}"
        dataloader = SteeringDataLoader(
            source_acts, source_attention_mask, steering_vectors, self.batch_size
        )
        diffs = []
        i = 0
        for steered_activations, attention_mask in dataloader:
            with self.model.trace(dummy_input(attention_mask)) as tracer:
                # todo: add test showing that dummy_input indeed doesn't matter
                self.model.skip_layers(0, self.steer_layer, steered_activations)
                steered_target_acts = self.model.layers_output[self.target_layer].save()
                tracer.stop()
            diffs.append(
                (steered_target_acts - target_acts[i : i + len(steered_activations)])
            )
            i += len(steered_activations)
            del steered_target_acts
        diffs = th.cat(diffs, dim=0)
        assert diffs.shape == (
            n_steering_vectors * num_prompts,
            seq_len,
            self.model.hidden_size,
        ), f"diffs.shape: {diffs.shape}, n_steering_vectors: {n_steering_vectors}, num_prompts: {num_prompts}, seq_len: {seq_len}"
        diffs = diffs.reshape(
            n_steering_vectors, num_prompts, seq_len, self.model.hidden_size
        )
        print(
            f"diffs.shape post reshape: {diffs.shape}, attention_mask.shape: {source_attention_mask.shape}"
        )
        diffs = diffs[:, source_attention_mask]
        print(f"diffs.shape post mask: {diffs.shape}")
        diffs = diffs.mean(dim=1)
        print(f"diffs.shape post mean: {diffs.shape}")
        assert diffs.shape == (
            n_steering_vectors,
            self.model.hidden_size,
        ), f"diffs.shape: {diffs.shape}, n_steering_vectors: {n_steering_vectors}, hidden_size: {self.model.hidden_size}"
        return diffs


    def get_calibrated_steering_factor(self, prompts: str | list[str], target_ratio: float = 0.5, d_rnd_proj: int = 30) -> float:
        """
        Get the calibrated steering factor for the given prompts.

        Args:
            prompts (str | list[str]): The prompts to steer.
            target_ratio (float): The target ratio of the steering factor (λ). Defaults to 0.5
            d_rnd_proj (int): The number of random projections to use for the calibration (d_proj). Defaults to 30

        """
        if isinstance(prompts, list):
            logger.warn("Using the first prompt only to calibrate the steering factor, (see DCT paper)")
        causal_map = F.normalize(th.randn(self.model.hidden_size, d_rnd_proj), dim=0)  # W, V_cal in legacy code



        def jacobian_ratio(r):
            raise NotImplementedError("Not implemented, todo")

            # denom = (r * U_cal_norms).pow(2)
            # delta_acts_avg = StreamingAverage()
            # with torch.no_grad():
            #     for b in range(0, X.shape[0], batch_size):
            #         x = X[b : b + batch_size, :, :].to(delta_acts_single.device)
            #         y = Y[b : b + batch_size, :, :].to(delta_acts_single.device)
            #         delta_acts_batch = delta_acts(r * V_cal, x, y)
            #         delta_acts_avg.update(delta_acts_batch)
            # num = (delta_acts_avg.get_mean() - r * U_cal).pow(2).sum(dim=0)
            # return math.sqrt((num / denom).mean())

        # solve for jacobian_ratio = target_ratio
        soln = root_scalar(
            lambda r: jacobian_ratio(r) - target_ratio, bracket=[0.001, 100.0]
        )
        self.R = soln.root
