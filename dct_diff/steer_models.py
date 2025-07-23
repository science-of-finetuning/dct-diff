from abc import ABC, abstractmethod

from tqdm.auto import tqdm
import numpy as np
import torch as th
import torch.nn.functional as F
from einops import einsum, rearrange
from nnterp import StandardizedTransformer
from loguru import logger
from .utils import SteeringDataLoader, dummy_input, cleanup_generations
from scipy.optimize import root_scalar
from typing import Literal


class SteeringEffectExtractor(ABC):
    def __init__(self, steer_layer: int, target_layer: int, batch_size: int):
        self.steer_layer = steer_layer
        self.target_layer = target_layer
        self.batch_size = batch_size
        self._steering_factor = None

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

    @property
    def steering_factor(self) -> float:
        if self._steering_factor is None:
            self._steering_factor = self.get_calibrated_steering_factor()
        return self._steering_factor

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
        steering_factor: float | None = None,
    ):
        super().__init__(steer_layer, target_layer, batch_size)
        self.model = model
        self.model.requires_grad_(False)
        self._steering_factor = steering_factor

    @property
    def hidden_size(self) -> int:
        return self.model.hidden_size

    def prepare_steering_data(
        self,
        prompts: str | list[str],
        steering_vectors: th.Tensor,
        steering_factor: float | None = None,
        cache_inputs: bool = False,
        return_type: Literal["steered_acts", "steering_vectors"] = "steered_acts",
    ) -> th.Tensor:
        n_steering_vectors = steering_vectors.shape[0]
        if isinstance(prompts, str):
            prompts = [prompts]
        num_prompts = len(prompts)
        input_ids = None
        with self.model.trace(prompts) as tracer:
            seq_len = self.model.input_size[1].save()
            if cache_inputs:
                input_ids = self.model.input_ids.save()
            source_attention_mask = self.model.attention_mask.bool().save()
            source_acts = self.model.layers_output[self.steer_layer]
            source_acts_device = source_acts.device.save()
            source_acts = source_acts.cpu().save()
            target_acts = self.model.layers_output[self.target_layer].save()
            tracer.stop()
        target_acts = target_acts.unsqueeze(0).expand(n_steering_vectors, -1, -1, -1)
        target_acts = target_acts.reshape(n_steering_vectors * num_prompts, seq_len, -1)
        assert target_acts.shape == (
            n_steering_vectors * num_prompts,
            seq_len,
            self.model.hidden_size,
        ), f"source_acts.shape: {source_acts.shape}, n_steering_vectors: {n_steering_vectors}, num_prompts: {num_prompts}, seq_len: {seq_len}"
        steered_act_dataloader = SteeringDataLoader(
            source_acts,
            source_attention_mask,
            steering_vectors,
            self.steering_factor if steering_factor is None else steering_factor,
            batch_size=self.batch_size,
            input_ids=input_ids,
            return_type=return_type,
        )
        return (
            source_acts,
            target_acts,
            source_attention_mask,
            source_acts_device,
            num_prompts,
            seq_len,
            steered_act_dataloader,
        )

    def get_steering_effect(
        self,
        prompts: str | list[str],
        steering_vectors: th.Tensor,
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
        n_steering_vectors = steering_vectors.shape[0]
        (
            source_acts,
            target_acts,
            source_attention_mask,
            source_acts_device,
            num_prompts,
            seq_len,
            steered_act_dataloader,
        ) = self.prepare_steering_data(prompts, steering_vectors)

        diffs = []
        i = 0
        for steered_activations, attention_mask in steered_act_dataloader:
            with self.model.trace(dummy_input(attention_mask)) as tracer:
                # todo: add test showing that dummy_input indeed doesn't matter
                self.model.skip_layers(
                    0, self.steer_layer, steered_activations.to(source_acts_device)
                )
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
        diffs = diffs[:, source_attention_mask]
        diffs = diffs.mean(dim=1)
        assert diffs.shape == (
            n_steering_vectors,
            self.model.hidden_size,
        ), f"diffs.shape: {diffs.shape}, n_steering_vectors: {n_steering_vectors}, hidden_size: {self.model.hidden_size}"
        return diffs

    @th.no_grad()
    def steered_generations(
        self,
        prompts: str | list[str],
        steering_vectors: th.Tensor,
        steering_factors: th.Tensor | list[float] | float | None = None,
        steer_type: Literal["input_only", "all"] = "input_only",
        **kwargs,
    ) -> list[list[str]]:
        """
        Get the steered generations for the given prompts and steering vectors.

        Args:
            prompts (str | list[str]): The prompts to steer.
            steering_vectors (th.Tensor): The steering vectors to use (n_steering_vectors, hidden_size)
            steering_factor (th.Tensor | list[float] | float | None): The steering factor to use. Defaults to None, which means using the calibrated steering factor.

        Returns:
            list[list[str]] | list[list[list[str]]]: The steered generations for each steering vector on each prompt. (n_steering_vectors, num_prompts) or (n_diff_steering_vectors, num_factors, num_prompts)
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        num_prompts = len(prompts)
        if isinstance(steering_factors, (list, th.Tensor)):
            if isinstance(steering_factors, list):
                steering_factors = th.tensor(steering_factors)
            n_diff_steering_vectors = steering_vectors.shape[0]
            steering_factor = 1.0
            steering_vectors = einsum(
                steering_factors,
                steering_vectors,
                "num_factors, num_vectors hidden_size -> num_vectors num_factors hidden_size",
            )
            steering_vectors = rearrange(
                steering_vectors,
                "num_vectors num_factors hidden_size -> (num_vectors num_factors) hidden_size",
            )
        elif isinstance(steering_factors, float) or steering_factors is None:
            steering_factor = steering_factors
            n_diff_steering_vectors = None
        else:
            raise ValueError(
                f"Invalid steering_factors (expected tensor, list or float or None, got {type(steering_factors)})"
            )

        n_steering_vectors = steering_vectors.shape[0]
        with self.model.trace(prompts) as tracer:
            source_attention_mask = self.model.attention_mask.bool().save()
            input_ids = self.model.input_ids.save()
            tracer.stop()

        steered_act_dataloader = SteeringDataLoader(
            None,
            source_attention_mask,
            steering_vectors,
            self.steering_factor if steering_factor is None else steering_factor,
            batch_size=self.batch_size,
            input_ids=input_ids,
            return_type="steering_vectors",
        )

        generations = []
        for batch_steering, input_ids, attention_mask in tqdm(
            steered_act_dataloader, desc="Generating text with steering vectors"
        ):
            # print(f"batch_steering.shape: {batch_steering.shape}, input_ids.shape: {input_ids.shape}, attention_mask.shape: {attention_mask.shape}")
            batch_steering = batch_steering.to(self.model.device).unsqueeze(1)

            def steer():
                acts = self.model.layers_output[self.steer_layer]
                steered_acts = acts + batch_steering
                assert (
                    steered_acts.shape == acts.shape
                ), f"steered_acts.shape: {steered_acts.shape} != acts.shape: {acts.shape}"
                self.model.layers_output[self.steer_layer] = steered_acts

            with self.model.generate(
                dict(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            ) as tracer:
                if steer_type == "all":
                    with tracer.all():
                        steer()
                elif steer_type == "input_only":
                    steer()
                else:
                    raise ValueError(f"Invalid steer_type: {steer_type}")
                out = self.model.generator.output.save()

            generations.extend(cleanup_generations(self.model.tokenizer.batch_decode(out), self.model.tokenizer))
        generations = np.array(generations)
        generations = generations.reshape(n_steering_vectors, num_prompts)
        if n_diff_steering_vectors is not None:
            generations = generations.reshape(
                n_diff_steering_vectors, len(steering_factors), num_prompts
            )
        return generations.tolist()

    def get_calibrated_steering_factor(
        self, prompts: str | list[str], target_ratio: float = 0.5, d_rnd_proj: int = 30
    ) -> float:
        """
        Get the calibrated steering factor for the given prompts.

        Args:
            prompts (str | list[str]): The prompts to steer.
            target_ratio (float): The target ratio of the steering factor (λ). Defaults to 0.5
            d_rnd_proj (int): The number of random projections to use for the calibration (d_proj). Defaults to 30

        """
        if isinstance(prompts, list):
            logger.warn(
                "Using the first prompt only to calibrate the steering factor, (see DCT paper)"
            )
        causal_map = F.normalize(
            th.randn(self.model.hidden_size, d_rnd_proj), dim=0
        )  # W, V_cal in legacy code

        raise NotImplementedError("Not implemented, todo")

        def jacobian_ratio(r):
            pass

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
