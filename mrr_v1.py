from collections.abc import Callable

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["MeanReciprocalRank"]

class MeanReciprocalRank(Metric):
    r"""Calculates Mean Reciprocal Rank (MRR) at k for Recommendation Systems."""

    required_output_keys = ("y_pred", "y")
    _state_dict_all_req_keys = ("_sum_mrr_per_k", "_num_examples")

# ---- same from HRR (start) -----
    def __init__(
        self,
        top_k: list[int] | int = 1,
        ignore_zero_hits: bool = True,
        output_transform: Callable = lambda x: x,
        device: str | torch.device = torch.device("cpu"),
        skip_unrolling: bool = False,
    ):

        if not isinstance(top_k, (int, list)):
            raise ValueError("top_k must be either int or a list[int]")

        top_k = [top_k] if isinstance(top_k, int) else top_k

        if len(top_k) == 0:
            raise ValueError("top_k must have at least one positive value")

        if any(k <= 0 for k in top_k):
            raise ValueError("top_k must be list of positive integers only.")

        self.top_k = sorted(top_k)
        self.ignore_zero_hits = ignore_zero_hits

        super().__init__(output_transform, device=device, skip_unrolling=skip_unrolling)

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_mrr_per_k = torch.zeros(len(self.top_k), device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self,output: tuple[torch.Tensor, torch.Tensor]) -> None:
        if len(output) != 2:
            raise ValueError(f"output should be in format `(y_pred,y)` but got tuple of {len(output)} tensors.")

        y_pred, y = output

        if y_pred.shape != y.shape:
            raise ValueError(f"y_pred and y must be in the same shape, got {y_pred.shape} != {y.shape}.")

        if self.ignore_zero_hits:
            # Remove users with no relevant items, like [0,0,0,0]
            valid_mask = torch.any(y > 0, dim=-1)
            y_pred = y_pred[valid_mask]
            y = y[valid_mask]

        if y.shape[0] == 0:
            return

        max_k = self.top_k[-1]
        # indices of top-ranked preds
        _, indices = torch.topk(y_pred, k=max_k, dim=-1)

# ---- same from HRR (end) -----

        # Rearrange labels according to predicted ranking; example: [0,1,0,1]
        # ranked_relevance = [0,1,1,0]
        ranked_relevance = torch.gather(y, dim=-1, index=indices)

        for i, k in enumerate(self.top_k):

            # Only consider top-k recommendations
            relevance_k = ranked_relevance[:, :k]

            # Determine whether user has
            # at least one relevant item in top-k.
            has_hit = torch.any( relevance_k > 0, dim=-1)

            # Find position of first relevant item. Example: [0,0,1]
            # argmax -> 2; rank -> 3
            # edge case: argmax([0,0,0]) returns 0, but corrected using has_hit.
            first_pos = (torch.argmax((relevance_k > 0).int(),dim=-1) + 1)

            # Reciprocal rank; users with no hit receive 0
            rr = torch.where(has_hit, 1.0 / first_pos.float(), torch.zeros_like(first_pos, dtype=torch.float32))

            # Add batch contribution
            self._sum_mrr_per_k[i] += rr.sum().to(self._device)

        self._num_examples += y.shape[0]

    @sync_all_reduce("_sum_mrr_per_k", "_num_examples")
    def compute(self) -> list[float]:
        if self._num_examples == 0:
            raise NotComputableError("MeanReciprocalRank must have at least one example.")

        # avg reciprocal rank over all users
        rates = (self._sum_mrr_per_k/ self._num_examples).tolist()

        return rates
