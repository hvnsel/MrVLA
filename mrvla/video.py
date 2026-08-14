"""Write an episode's frames as a single replayable file, with no new dependencies.

Steering evidence is qualitative: the claim "this feature makes the robot hover above the
object instead of grasping it" is read off a replay, not a success rate. So we need episode
video, but we do NOT want to add a package to a working cluster environment mid-experiment.

Backends are tried in order and the first that imports wins:
  imageio  -> .mp4   (smallest, best quality)
  cv2      -> .mp4   (usually present via robosuite/LIBERO)
  PIL      -> .gif   (always available -- PIL is already a hard dependency of this repo)

ONE FILE PER EPISODE, never a directory of PNGs. On a parallel cluster filesystem tens of
thousands of small files is materially worse than the same bytes in a few large ones, and the
frames are useless individually anyway.

Frames are uint8 [H, W, 3] RGB, in playback order.
"""

from __future__ import annotations

import numpy as np


def _prep(frames, stride: int = 1, max_side: int | None = None):
    """Subsample and downscale. Robot motion is smooth, so keeping every 2nd-3rd frame at
    128px loses nothing that matters for judging behaviour and cuts file size several-fold."""
    from PIL import Image

    out = []
    for i, f in enumerate(frames):
        if i % stride:
            continue
        a = np.asarray(f)
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        if max_side and max(a.shape[:2]) > max_side:
            im = Image.fromarray(a)
            s = max_side / max(im.size)
            im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))),
                           Image.LANCZOS)
            a = np.asarray(im)
        out.append(a)
    return out


def _mp4_imageio(frames, path, fps):
    import imageio.v2 as imageio
    # macro_block_size=1 stops the writer silently resizing odd dimensions
    imageio.mimwrite(path, frames, fps=fps, macro_block_size=1)


def _mp4_cv2(frames, path, fps):
    import cv2
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open")
    try:
        for f in frames:
            vw.write(f[:, :, ::-1])          # cv2 wants BGR
    finally:
        vw.release()


def _gif_pil(frames, path, fps):
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=max(20, int(round(1000.0 / fps))), loop=0, optimize=True)


def write_video(frames, out_stem: str, fps: int = 10, stride: int = 1,
                max_side: int | None = 128, prefer: str | None = None):
    """Write `frames` to `<out_stem>.mp4` or `<out_stem>.gif`. Returns (path, backend).

    `prefer` forces a backend ("imageio" | "cv2" | "pil"); otherwise all are tried in order.
    Raises ValueError on empty input -- silently writing nothing would hide a broken rollout.
    """
    frames = _prep(frames, stride=stride, max_side=max_side)
    if not frames:
        raise ValueError("no frames to write")

    order = [("imageio", _mp4_imageio, ".mp4"),
             ("cv2", _mp4_cv2, ".mp4"),
             ("pil", _gif_pil, ".gif")]
    if prefer:
        order = [o for o in order if o[0] == prefer] or order

    last = None
    for name, fn, ext in order:
        path = out_stem + ext
        try:
            fn(frames, path, fps)
            return path, name
        except Exception as e:                # ImportError, codec missing, write failure
            last = f"{name}: {e}"
    raise RuntimeError(f"no video backend succeeded (last error -- {last})")
