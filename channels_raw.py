#!/usr/bin/env python3
"""The channel analysis with NO corrections applied -- the baseline the corrections are judged against.

WHY THIS EXISTS
---------------
`analyze_channels.py` reports channel statistics after two adjustments, and both deserve to be
seen alongside the unadjusted numbers rather than instead of them.

`decision_shares` normalises every decision to one unit of causal mass. It was introduced against
a hypothesised per-slot ||u_contrast|| difference: ordered bins put the average bin near the
centre, so a slot emitting EXTREME bins would carry more mass than one emitting central ones, and
the gripper is near-binary while the six delta channels are small nudges. That argument was later
tested and the effect is not present -- absolute mass per slot runs 0.1331 to 0.1537 against an
even 0.1429 (results.md P5c). The normalisation is applied uniformly to all seven slots, so it is
not gripper-specific, but if it corrects nothing then the absolute and share analyses must agree,
and any disagreement is the correction rather than the data.

The transition mask is a different matter, and it IS applied asymmetrically:

    grip      = tok_rows.reshape(n, n_sl)[:, args.gripper_slot]
    trans_dec = transition_mask(grip, ep_of, ts_of)
    trans     = np.repeat(trans_dec, n_sl)          # broadcast to ALL seven slots

Every `_trans` counter is therefore conditioned on whether THE GRIPPER changed at that timestep.
For the gripper that is the intended control. For dx it means "dx flipped at a timestep where the
gripper happened to move", which is not a control for anything -- dx changes on ~95% of steps
regardless. The `_trans` family is consequently NOT comparable across channels, and a table
mixing gripper `_trans` with dx `_trans` compares two different conditionings.

Fixing that requires a per-channel mask built from each channel's own emitted bins, which needs a
rerun of the GPU pass. This script does what can be done without one: report every statistic in
its unconditioned form, where all seven channels are treated identically.

WHAT IS AND IS NOT RECOVERABLE HERE
-----------------------------------
Recoverable from the saved npz: absolute (unnormalised) channel mass and profiles, and the
unmasked flip rates `flip` and `flip_active`. Sufficiency needs nothing -- it was never shared or
masked in the first place.

Not recoverable: per-channel transition masks. The per-decision emitted bins are not saved.

CPU only, seconds.

    python channels_raw.py --npz $B/CHANNELS/goal/layer_31_channels.npz \
                           --attr $B/ATTR/goal_k100/layer_31_attribution.npz
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from identify_features import adjusted_breadth
from mrvla.channels import DEFAULT_CHANNEL_NAMES, channel_profile
from mrvla.stats import rankdata_average


def rank_partial(y, x, c1, c2) -> float:
    """Rank-partial correlation of y and x controlling for c1 and c2, ties averaged."""
    m = np.isfinite(y) & np.isfinite(x) & np.isfinite(c1) & np.isfinite(c2)
    if m.sum() < 5:
        return float("nan")
    rk = lambda v: (lambda r: r - r.mean())(rankdata_average(v[m]))
    ry, rx = rk(y), rk(x)
    rc = np.stack([rk(c1), rk(c2)], axis=1)
    by, *_ = np.linalg.lstsq(rc, ry, rcond=None)
    bx, *_ = np.linalg.lstsq(rc, rx, rcond=None)
    ey, ex = ry - rc @ by, rx - rc @ bx
    den = np.sqrt((ex * ex).sum() * (ey * ey).sum())
    return float((ex * ey).sum() / den) if den > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, help="layer_NN_channels.npz")
    ap.add_argument("--attr", required=True, help="layer_NN_attribution.npz (for breadth)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    a = np.load(args.attr, allow_pickle=True)
    C_abs = z["C_slot_abs"].astype(np.float64)          # [S, G, F]
    C_shr = z["C_slot_share"].astype(np.float64)
    names = list(DEFAULT_CHANNEL_NAMES[:C_abs.shape[0]])
    S = C_abs.shape[0]

    adj = adjusted_breadth(a["PR"].astype(np.float64), a["magnitude"].astype(np.float64),
                           a["base_rate"].astype(np.float64), a["is_active"].astype(bool))
    mag = a["magnitude"].astype(np.float64)
    base = a["base_rate"].astype(np.float64)

    prof_abs, prof_shr = channel_profile(C_abs), channel_profile(C_shr)

    print(f"\n=== {os.path.basename(os.path.dirname(args.npz))} : all seven channels, "
          f"treated identically ===\n")
    tot = C_abs.sum()
    print(f"  {'channel':8s} {'mass share':>11s} {'vs even':>9s} "
          f"{'corr(breadth, conc)':>21s} {'':>3s} {'share form':>11s} {'delta':>8s}")
    rows = []
    for s in range(S):
        frac = C_abs[s].sum() / tot
        r_abs = rank_partial(prof_abs[s], adj, mag, base)
        r_shr = rank_partial(prof_shr[s], adj, mag, base)
        agree = "ok" if np.sign(r_abs) == np.sign(r_shr) else "SIGN"
        print(f"  {names[s]:8s} {frac:11.4f} {frac/(1/S):8.2f}x {r_abs:21.4f} {agree:>3s} "
              f"{r_shr:11.4f} {r_shr - r_abs:+8.4f}")
        rows.append({"channel": names[s], "mass_share": float(frac),
                     "corr_absolute": r_abs, "corr_share": r_shr,
                     "delta": float(r_shr - r_abs)})

    d = np.array([abs(r["delta"]) for r in rows])
    signflip = sum(1 for r in rows if np.sign(r["corr_absolute"]) != np.sign(r["corr_share"]))
    print(f"\n  largest |share - absolute| across channels: {d.max():.4f}   "
          f"sign disagreements: {signflip}/{S}")
    print("  If the normalisation corrects nothing, these two columns agree and the")
    print("  conclusion does not depend on which is reported.")

    if "sufficiency_recon" in z:
        sr, sf = z["sufficiency_recon"], z["sufficiency_features_only"]
        print(f"\n  sufficiency (never shared, never masked -- identical treatment already):")
        print(f"    {'channel':8s} {'features+bias':>14s} {'features alone':>15s}")
        for s in range(S):
            print(f"    {names[s]:8s} {sr[s]:14.3f} {sf[s]:15.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"channels": rows, "max_abs_delta": float(d.max()),
                       "sign_disagreements": int(signflip)}, f, indent=2)
        print(f"\n[raw] wrote {args.out}")


if __name__ == "__main__":
    main()
