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


class ActivationAblator:
    """Registers a forward hook that edits the residual stream by projecting out
    memorized SAE feature directions before the rest of the model sees them.

    For every hooked forward pass:
        h_ablated = h - (h @ V.T) @ V
    where V [K, d] is the matrix of unit-norm decoder directions for the features
    selected by the current ablation mask.

    Two modes
    ---------
    deterministic  : ablate all features where prob_memorized > threshold on every pass.
    probabilistic  : sample a fresh Bernoulli mask each pass with
                     P(ablate_j) = prob_memorized[j] * ablate_frac.
                     Use for LoRA training so the model sees a variety of ablation
                     conditions across mini-batches.

    Parameters
    ----------
    layer                : the nn.Module to hook.
    decoder_dirs         : [F, d] decoder directions from SAE final.pt W_dec.
                           Normalised to unit norm internally.
    prob_memorized       : [F]  = 1 - P(general) from the generality classifier.
    mode                 : "deterministic" | "probabilistic"
    threshold            : deterministic mode: ablate if prob_memorized > threshold.
    ablate_frac          : probabilistic mode: scale all probs by this factor before
                           sampling (0.0 → never ablate; 1.0 → use raw probs).
    ablate_decode_passes : if False, only ablate the first (prefill) call per step and
                           skip subsequent decode calls.  True is correct for LoRA
                           training (single teacher-forced forward) and for eval
                           rollouts where you want continuous ablation across all steps.
    """

    def __init__(
        self,
        layer: torch.nn.Module,
        decoder_dirs: torch.Tensor,    # [F, d]
        prob_memorized: torch.Tensor,  # [F]
        mode: str = "probabilistic",
        threshold: float = 0.5,
        ablate_frac: float = 1.0,
        ablate_decode_passes: bool = True,
    ):
        if mode not in ("deterministic", "probabilistic"):
            raise ValueError(f"mode must be 'deterministic' or 'probabilistic', got {mode!r}")
        if not (0.0 <= ablate_frac <= 1.0):
            raise ValueError(f"ablate_frac must be in [0, 1], got {ablate_frac}")

        self.mode = mode
        self.threshold = threshold
        self.ablate_frac = ablate_frac
        self.ablate_decode_passes = ablate_decode_passes

        # Normalise to unit norm (SAE training constrains this, but defensive).
        norms = decoder_dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self._V_full = (decoder_dirs / norms).float().cpu()  # [F, d], CPU master copy

        self._prob_mem = prob_memorized.float().clamp(0.0, 1.0).cpu()  # [F]

        if mode == "deterministic":
            det_mask = self._prob_mem > threshold
            if det_mask.sum() == 0:
                import warnings
                warnings.warn(
                    f"ActivationAblator: no features exceed threshold={threshold} in "
                    f"deterministic mode; nothing will be ablated."
                )
            self._det_V: torch.Tensor | None = self._V_full[det_mask]  # [K, d]
        else:
            self._det_V = None

        self._call_count = 0  # hook calls within a single logical forward step
        self._handle = layer.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self._call_count += 1
        if not self.ablate_decode_passes and self._call_count > 1:
            return  # skip decode passes

        hidden = output[0] if isinstance(output, tuple) else output
        # hidden: [B, S, d]
        device = hidden.device
        dtype  = hidden.dtype

        V = self._get_V(device)
        if V is None or V.shape[0] == 0:
            return  # nothing to ablate this pass

        # Project out each ablated direction: h -= (h @ V.T) @ V
        h       = hidden.float()          # [B, S, d]
        coeffs  = h @ V.T                 # [B, S, K]
        h_out   = (h - coeffs @ V).to(dtype)  # [B, S, d]

        if isinstance(output, tuple):
            return (h_out,) + output[1:]
        return h_out

    def _get_V(self, device: torch.device) -> torch.Tensor | None:
        if self.mode == "deterministic":
            return self._det_V.to(device)

        # Probabilistic: sample a fresh Bernoulli mask this pass.
        probs = self._prob_mem * self.ablate_frac
        mask  = torch.bernoulli(probs).bool()
        if not mask.any():
            return None
        return self._V_full[mask].to(device)

    def reset_step(self) -> None:
        """Reset the call counter between logical inference steps (not mini-batches)."""
        self._call_count = 0

    def remove(self) -> None:
        """Detach the hook."""
        self._handle.remove()

    def __enter__(self) -> "ActivationAblator":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()
