# Pre-registration — single-feature ablation, LIBERO-goal

Written **before** the run. The point of recording it here is that each feature carries a
specific, falsifiable prediction about *where* damage should land, not merely that damage
happens. A single-feature ablation with a named target task is testable in a way the top-N
coalition was not.

- suite: `libero_goal` (10 tasks), layer 31, SAE `ACT_ACTION_SAE/goal/sae`, F=2048, k=100
- attribution: `ATTR/goal_k100/layer_31_attribution.npz`
- 20 episodes/task, paired: every condition replays the **same** init states as baseline
- each feature ablated **on its own** (one condition per feature), never as a coalition

## Features and predictions

| Feature | Role (from frames) | Fires strongly on | Prediction |
|---|---|---|---|
| 1167 | specialist | task 8 | damage concentrated on task 8, ~none elsewhere |
| 1140 | specialist | tasks 2, 9 | damage on tasks 2 and 9, ~none elsewhere |
| 1134 | specialist | tasks 4, 8 | damage on tasks 4 and 8, ~none elsewhere |
| 1235 | general | — | damage spread across many tasks, or none anywhere |
| 1628 | general | — | damage spread across many tasks, or none anywhere |
| 1999 | general | — | damage spread across many tasks, or none anywhere |

The generality claim is about the **shape** of the damage, not its size: specialists should
produce narrow, task-locked damage (low damage participation ratio) and generals broad,
diffuse damage (high PR). A specialist that damages its named task and nothing else is a
positive result even if the drop is only a few points, because the *location* was predicted
in advance.

## What counts as a hit / miss

- **hit**: the named task drops and unnamed tasks do not
- **miss**: no drop on the named task
- **informative miss**: a drop appears on tasks that were *not* named — the feature is not
  the specialist the frames suggested, and the frame-based labelling needs revisiting

## Independent check available before any rollout

`manifest.json` records `per_task_profile` per condition (the column sums of `C[:, idx]`).
That is the attribution-derived prediction of where each feature does its causal work, so the
frame-based claims above ("1167 fires strongly on task 8") can be verified against the
attribution data directly. If `per_task_profile` disagrees with the table above, resolve that
*before* interpreting rollouts.

## Power note

At 20 paired episodes on the named task, only a fairly large per-task drop will register
(McNemar needs roughly 6+ discordant pairs in one direction). A null on a single feature is
therefore weak evidence. If a specialist shows a suggestive but non-significant drop on its
named task, the cheap follow-up is to re-run **only** that (feature, task) pair at higher
episode count rather than raising episodes everywhere.

## Context

Runs after the top-5 coalition ablation, which was a null: baseline 78.9% (matching published
OpenVLA libero-goal ~79.2%), general 75.6%, specialist 79.4%, random 78.3%. The `firing`
condition (top-5 by raw firing rate) collapsed to 2.2%, but is not magnitude-matched to the
others and is likely removing a large share of the residual norm rather than action-specific
structure.
