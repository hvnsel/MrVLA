"""PyTorch 2.6+ compatibility for loading LIBERO's bundled init-state files.

PyTorch 2.6 changed the default of ``torch.load(weights_only=...)`` from False to True.
LIBERO's ``benchmark.get_task_init_states`` calls a bare ``torch.load`` on its shipped
``*.pruned_init`` files, which are pickled NUMPY arrays, so under torch >= 2.6 every rollout
dies with::

    _pickle.UnpicklingError: Weights only load failed ...
    Unsupported global: GLOBAL numpy.core.multiarray._reconstruct

The blunt fix is to force ``weights_only=False`` globally, but that re-enables arbitrary code
execution for EVERY checkpoint the process loads, including model weights pulled from a hub.
We take the narrow route instead: allowlist exactly the numpy symbols an ndarray pickle needs.
``torch.serialization.add_safe_globals`` is process-wide, so this fixes LIBERO's internal call
without patching third-party source and without monkeypatching ``torch.load``.

The required set was determined empirically, not guessed:
  * ``_reconstruct``  -- the array reconstructor. NOTE it lives in ``numpy._core.multiarray``
    on numpy >= 2.0 and ``numpy.core.multiarray`` on 1.x, so both spellings are tried.
  * ``numpy.ndarray`` and ``numpy.dtype``
  * the concrete dtype classes (``numpy.dtypes.Float64DType`` and friends, numpy >= 1.25);
    allowlisting ``numpy.dtype`` alone is NOT sufficient -- the unpickler rejects the concrete
    class next.
  * numpy scalar types, as a fallback for older numpy where dtypes pickle through them.

Everything is best-effort and version-guarded: a symbol that does not exist in the installed
numpy is skipped rather than raising, so this stays safe across the 1.x/2.x split.
"""

from __future__ import annotations

import importlib

_APPLIED = False


def allow_numpy_pickles() -> list[str]:
    """Allowlist the numpy globals needed to unpickle plain ndarrays under weights_only=True.

    Idempotent -- safe to call from every worker and every entry point. Returns the list of
    fully-qualified names registered (empty on torch < 2.6, where nothing is enforced and
    there is nothing to do).
    """
    global _APPLIED
    if _APPLIED:
        return []
    try:
        import torch
    except ImportError:
        return []
    add = getattr(getattr(torch, "serialization", None), "add_safe_globals", None)
    if add is None:
        _APPLIED = True                      # torch predates weights_only enforcement
        return []

    import numpy as np

    cands: list = []
    # The array reconstructor moved package in numpy 2.0; accept whichever exists.
    for modname in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        fn = getattr(mod, "_reconstruct", None)
        if fn is not None:
            cands.append(fn)

    for obj in (getattr(np, "ndarray", None), getattr(np, "dtype", None)):
        if obj is not None:
            cands.append(obj)

    # Concrete dtype classes: numpy.dtype alone is not enough, the unpickler asks for these.
    dtypes_mod = getattr(np, "dtypes", None)
    if dtypes_mod is not None:
        for name in dir(dtypes_mod):
            if name.endswith("DType"):
                obj = getattr(dtypes_mod, name, None)
                if isinstance(obj, type):
                    cands.append(obj)

    # Older numpy pickles dtypes via the scalar types instead.
    for name in ("float64", "float32", "float16", "int64", "int32", "int16", "int8",
                 "uint64", "uint32", "uint16", "uint8", "bool_"):
        obj = getattr(np, name, None)
        if isinstance(obj, type):
            cands.append(obj)

    seen, uniq = set(), []
    for c in cands:
        key = id(c)
        if key not in seen:
            seen.add(key)
            uniq.append(c)

    add(uniq)
    _APPLIED = True
    return [f"{getattr(c, '__module__', '?')}.{getattr(c, '__name__', repr(c))}"
            for c in uniq]
