"""Forward-hook based residual-stream activation capture for OpenVLA."""

from __future__ import annotations

import numpy as np
import torch


class ActivationCollector:
    """Registers forward hooks on selected decoder layers and pools activations.

    Per inference step (one image+instruction), OpenVLA runs a prefill forward pass over
    the full prompt followed by single-token decode passes. We keep only the **prefill**
    activation (the first hook call per layer per step) and mean-pool it over the token
    dimension, yielding one vector per layer per timestep.
    """

    def __init__(
        self,
        layers: torch.nn.ModuleList, # a list of layers which our hooks will be registered on
        layer_indices: list[int] | None = None, # the indices of the layers on which to register hooks
        pool: str = "mean",
        dtype: torch.dtype = torch.float16,
    ):
        if layer_indices is None:
            layer_indices = list(range(len(layers)))
        if pool not in ("mean", "last"):
            raise ValueError(f"Unknown pool '{pool}', expected 'mean' or 'last'.")

        self.layers = layers
        self.layer_indices = layer_indices
        self.pool = pool
        self.dtype = dtype
        self._buffers: dict[int, torch.Tensor] = {} # stores pooled activations where key is the int layer index and the value is the pooled activation tensor for that layer
        self._handles: list[torch.utils.hooks.RemovableHandle] = [] # a list of handles for the registered forward hooks, used to remove the hooks later

        for idx in layer_indices:
            # attach hooks to specified layers and append to the list of handles
            handle = layers[idx].register_forward_hook(self._make_hook(idx)) 
            self._handles.append(handle)

    def _make_hook(self, idx: int):
        # return a hook function that captures the prefill activation for the specified int layer index

        def hook(_module, _inputs, output):
            # Decoder layers return a tuple; hidden states are element 0: [B, S, H].
            hidden = output[0] if isinstance(output, tuple) else output
            if idx in self._buffers:
                return  # already captured the prefill pass for this step
            if self.pool == "mean":
                vec = hidden.float().mean(dim=1)  # [B, H]
            else:  # "last"
                vec = hidden.float()[:, -1, :]  # [B, H]
            self._buffers[idx] = vec.to(self.dtype).cpu()

        return hook

    def reset(self) -> None:
        """Clear buffers before the next inference step."""
        self._buffers = {}

    def gather_single(self) -> np.ndarray:
        """Return activations for a single-sample (batch=1) step as [L, H]."""
        mats = self._stacked()  # [L, B, H]
        if mats.shape[1] != 1:
            raise RuntimeError(
                f"gather_single expects batch size 1, got {mats.shape[1]}."
            )
        return mats[:, 0, :].numpy()

    def gather_batch(self) -> np.ndarray:
        """Return activations for a batched step as [B, L, H]."""
        mats = self._stacked()  # [L, B, H]
        return mats.permute(1, 0, 2).numpy()

    def _stacked(self) -> torch.Tensor:
        # stack the pooled activations tensors over all specified layers across the 0 dimension
        
        missing = [i for i in self.layer_indices if i not in self._buffers]
        if missing:
            raise RuntimeError(
                f"No activations captured for layers {missing}. Did the forward pass run?"
            )
        return torch.stack([self._buffers[i] for i in self.layer_indices], dim=0)

    def remove(self) -> None:
        """Detach all hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "ActivationCollector":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()
