import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@th.no_grad()
def replace_with_grad(
    params: nn.Parameter, normalize: bool = False, beta: float | None = None
):
    assert params.grad is not None, "Parameter has no gradient"
    if beta is not None:
        params.data = beta * params.grad + (1 - beta) * params.data
    else:
        params.data = params.grad.clone()
    if normalize:
        params.data = params.data / th.norm(params.data, dim=-1, keepdim=True)
    params.grad = None


class SteeringDataLoader:
    def __init__(
        self,
        source_activations: th.Tensor,
        source_attention_mask: th.Tensor,
        steering_vectors: th.Tensor,
        steering_factor: float,
        batch_size: int,
    ):
        """
        Given a set of source activations (n_prompt, seq_len, hidden_size) and a set of steering vectors (n_steering_vector, hidden_size),
        create a dataloader that goes through all the combinations of source activations + steering vectors.

        Args:
            source_activations (th.Tensor): (n_prompt, seq_len, hidden_size)
            source_attention_mask (th.Tensor): (n_prompt, seq_len)
            steering_vectors (th.Tensor): (n_steering_vector, hidden_size)
            batch_size (int): The batch size to use for the dataloader.
        """
        self.source_activations = source_activations
        self.source_attention_mask = source_attention_mask
        self.steering_vectors = steering_vectors
        self.steering_factor = steering_factor
        assert (
            source_activations.shape[2] == steering_vectors.shape[1]
        ), f"Hidden size of source activations and steering vectors must match, but got {source_activations.shape[2]} and {steering_vectors.shape[1]}"
        self.batch_size = batch_size
        self._dataloader = None

    def __iter__(self):
        if self._dataloader is None:
            self.init_dataloader()
        assert self._dataloader is not None
        return iter(self._dataloader)

    def __len__(self):
        if self._dataloader is None:
            self.init_dataloader()
        assert self._dataloader is not None
        return len(self._dataloader)

    def init_dataloader(self):
        """Initialize dataloader that returns (steered_activations, attention_mask) tuples."""
        num_prompts, seq_len, hidden_size = self.source_activations.shape
        n_steering_vectors = self.steering_vectors.shape[0]

        data = (
            self.source_activations.unsqueeze(2)
            + self.steering_vectors.unsqueeze(0).unsqueeze(0) * self.steering_factor
        )
        assert data.shape == (
            num_prompts,
            seq_len,
            n_steering_vectors,
            hidden_size,
        ), f"Data shape must be (num_prompts, seq_len, n_steering_vectors, hidden_size), but got {data.shape}"
        data = data.permute(2, 0, 1, 3)
        data = data.reshape(n_steering_vectors * num_prompts, seq_len, hidden_size)

        attention_mask = self.source_attention_mask.unsqueeze(0).expand(
            n_steering_vectors, -1, -1
        )
        attention_mask = attention_mask.reshape(
            n_steering_vectors * num_prompts, seq_len
        )
        assert (
            attention_mask[:num_prompts] == self.source_attention_mask
        ).all(), "Attention mask mismatch"
        assert (
            data.shape[0] == attention_mask.shape[0]
        ), f"Batch dimension mismatch: data {data.shape[0]} vs attention_mask {attention_mask.shape[0]}"
        assert (
            data.shape[1] == attention_mask.shape[1]
        ), f"Sequence length mismatch: data {data.shape[1]} vs attention_mask {attention_mask.shape[1]}"

        dataset = TensorDataset(data, attention_mask)
        self._dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )


def dummy_input(attention_mask: th.Tensor) -> dict:
    return dict(
        input_ids=th.zeros(
            attention_mask.shape[0], attention_mask.shape[1], dtype=th.long
        ),
        attention_mask=attention_mask,
    )
