"""Sharded on-disk storage for collected activations + metadata."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np

_DTYPES = {"float16": np.float16, "float32": np.float32, "bfloat16": np.float32}


class ShardedActivationWriter:
    """Buffers per-timestep activations and flushes them to compressed ``.npz`` shards.

    Each shard stores:
      - ``acts``     ``[N, L, H]`` pooled residual-stream activations
      - ``episode``  ``[N]`` global episode index
      - ``timestep`` ``[N]`` step index within the episode
      - ``task_id``  ``[N]`` task identifier
      - ``success``  ``[N]`` 1/0 if the episode succeeded, -1 if unknown at write time
    """

    def __init__(
        self,
        out_dir: str,
        layer_indices: list[int],
        hidden_dim: int,
        model_name: str,
        shard_size: int = 8192,
        dtype: str = "float16",
        pool: str = "mean",
        extra_meta: dict | None = None,
    ):
        if dtype not in _DTYPES:
            raise ValueError(f"Unsupported dtype '{dtype}'. Choose from {list(_DTYPES)}.")
        os.makedirs(out_dir, exist_ok=True)

        self.out_dir = out_dir
        self.layer_indices = layer_indices
        self.hidden_dim = hidden_dim
        self.model_name = model_name
        self.shard_size = shard_size
        self.dtype = dtype
        self.np_dtype = _DTYPES[dtype]
        self.pool = pool
        self.extra_meta = extra_meta or {}

        self._acts: list[np.ndarray] = []
        self._episode: list[int] = []
        self._timestep: list[int] = []
        self._task_id: list[int] = []
        self._success: list[int] = []

        self._shard_idx = 0
        self.total_samples = 0
        self.task_names: dict[int, str] = {}

    def register_task(self, task_id: int, name: str) -> None:
        self.task_names[int(task_id)] = name

    def add(
        self,
        acts: np.ndarray,
        episode: int,
        timestep: int,
        task_id: int = 0,
        success: int = -1,
    ) -> None:
        """Add one timestep. ``acts`` must be shape ``[L, H]``."""
        if acts.shape != (len(self.layer_indices), self.hidden_dim):
            raise ValueError(
                f"Expected acts shape {(len(self.layer_indices), self.hidden_dim)}, "
                f"got {acts.shape}."
            )
        self._acts.append(acts.astype(self.np_dtype, copy=False))
        self._episode.append(int(episode))
        self._timestep.append(int(timestep))
        self._task_id.append(int(task_id))
        self._success.append(int(success))

        if len(self._acts) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._acts:
            return
        path = os.path.join(self.out_dir, f"shard_{self._shard_idx:05d}.npz")
        np.savez_compressed(
            path,
            acts=np.stack(self._acts, axis=0),
            episode=np.asarray(self._episode, dtype=np.int32),
            timestep=np.asarray(self._timestep, dtype=np.int32),
            task_id=np.asarray(self._task_id, dtype=np.int32),
            success=np.asarray(self._success, dtype=np.int32),
        )
        self.total_samples += len(self._acts)
        self._shard_idx += 1
        self._acts.clear()
        self._episode.clear()
        self._timestep.clear()
        self._task_id.clear()
        self._success.clear()

    def close(self) -> None:
        """Flush remaining samples and write the manifest."""
        self.flush()
        manifest = {
            "model_name": self.model_name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "layer_indices": self.layer_indices,
            "hidden_dim": self.hidden_dim,
            "dtype": self.dtype,
            "pool": self.pool,
            "shard_size": self.shard_size,
            "num_shards": self._shard_idx,
            "total_samples": self.total_samples,
            "task_names": {str(k): v for k, v in self.task_names.items()},
            **self.extra_meta,
        }
        with open(os.path.join(self.out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    def __enter__(self) -> "ShardedActivationWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
