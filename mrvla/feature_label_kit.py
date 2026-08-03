"""Hand-labelling kit for SAE features: general vs memorized.

Implements the three-stage visual protocol of arXiv:2603.19183 Sec 3.3.2
(Activation Viewer -> Feature Search -> labelling criteria) as an offline
tool, then fits a logistic regression on the labels (Eq. 11) so the
classifier can be refit on OUR model instead of inheriting coefficients.

WHY THE METRICS ARE HIDDEN
--------------------------
The paper's Stage-3 criteria instruct the annotator to consult "the global
per-feature metrics (mean onset count, mean active activation, episode
coverage)" when assigning a label, and the classifier is then fit on those
same metrics, reporting 100% LOO-CV accuracy.  That is partially circular:
labelling by the covariates guarantees the covariates separate the labels.

This kit therefore renders cards that show only the VISUAL evidence -
activation traces and the corresponding camera frames - and writes the four
computed statistics to a separate file that is joined only after labelling
is complete.  The resulting LOO-CV accuracy is then an honest estimate of
how well the four statistics recover a visually-defined concept.

Caveat, stated plainly: this de-circularises the *computed numbers*, not the
underlying visual patterns.  Burstiness is visible in a trace, and burstiness
is what mean onset count measures.  Stage 1 of the paper's own protocol
requires looking at exactly that.  The claim is only that the annotator does
not read off the classifier's input features.

WHAT THE ANNOTATOR JUDGES
-------------------------
For each feature card, the question is: *when this feature spikes, is the
same recognisable thing happening on screen, across different scenes?*

    general    bursts align with a semantically consistent sensorimotor event
               (a grasp, a placement, an object entering view) and that event
               recurs across visually DIVERSE episodes
    memorized  activation is sustained rather than event-locked, and/or the
               top episodes cluster into one or two visual scenes
    unclear    ambiguous -> excluded from the fit, as the paper also does

Subcommands
-----------
cards       render feature cards + labelling sheet + hidden metrics
fit         fit Eq. (11) on completed labels; report coefficients + LOO-CV
agreement   Cohen's kappa between two annotators' label files

Usage
-----
python mrvla/feature_label_kit.py cards \
    --codes-dir      E:/libero_goal_demos/codes_v4 \
    --generality-dir E:/libero_goal_demos/generality_v4 \
    --layer 8 --out-dir E:/libero_goal_demos/labeling/layer_08 \
    --task-suite libero_goal --n-features 60

# ... label in labeling/layer_08/label_sheet.html, export CSV ...

python mrvla/feature_label_kit.py fit \
    --labels  E:/libero_goal_demos/labeling/layer_08/labels_annotatorA.csv \
    --metrics E:/libero_goal_demos/labeling/layer_08/hidden_metrics.json \
    --out-dir E:/libero_goal_demos/labeling/layer_08

python mrvla/feature_label_kit.py agreement --labels A.csv B.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from mrvla.generality_classifier import BETA_LIBERO, sigmoid
except ImportError:  # direct execution
    from generality_classifier import BETA_LIBERO, sigmoid


METRIC_KEYS = ("mean_onsets", "coverage", "mean_act_mag", "rel_run_length")
# order must match Eq. (11): b0 + b1*obar + b2*c + b3*abar + b4*lbar_r
BETA_KEYS = ("intercept", "mean_onsets", "coverage",
             "mean_act_magnitude", "rel_run_length")


# ---------------------------------------------------------------------------
# Episode -> demo mapping
# ---------------------------------------------------------------------------
def build_episode_index(task_suite_name: str,
                        max_tasks: int | None = None,
                        max_demos_per_task: int | None = None) -> list[dict]:
    """Reconstruct the episode -> (task, demo file) map used at collection time.

    ``collect_libero_demos`` assigns ``global_episode`` by iterating tasks in
    order and, within each task, demo keys sorted numerically.  We replay that
    iteration so episode ids line up with the codes arrays.  Requires the
    libero package and the demo HDF5 files to be present.
    """
    import h5py
    from libero.libero import benchmark, get_libero_path

    try:
        from mrvla.libero_demos import _find_demo_file
    except ImportError:
        from libero_demos import _find_demo_file

    suite = benchmark.get_benchmark_dict()[task_suite_name]()
    n_tasks = suite.n_tasks if max_tasks is None else min(suite.n_tasks, max_tasks)
    datasets_root = get_libero_path("datasets")

    index, global_episode = [], 0
    for task_id in range(n_tasks):
        task = suite.get_task(task_id)
        demo_path = _find_demo_file(datasets_root, task_suite_name, task)
        with h5py.File(demo_path, "r") as f:
            demo_keys = sorted(f["data"].keys(),
                               key=lambda k: int(k.split("_")[-1]))
        if max_demos_per_task is not None:
            demo_keys = demo_keys[:max_demos_per_task]
        for demo_key in demo_keys:
            index.append({"episode": global_episode, "task_id": task_id,
                          "task": task.language, "demo_path": demo_path,
                          "demo_key": demo_key})
            global_episode += 1
    return index


class FrameSource:
    """Loads agentview frames for (episode, timestep) pairs."""

    def __init__(self, episode_index: list[dict] | None):
        self._by_ep = {r["episode"]: r for r in (episode_index or [])}

    @property
    def available(self) -> bool:
        return bool(self._by_ep)

    def get(self, episode: int, timesteps) -> list | None:
        rec = self._by_ep.get(int(episode))
        if rec is None:
            return None
        import h5py
        try:
            from mrvla.libero_demos import _resolve_image_key
        except ImportError:
            from libero_demos import _resolve_image_key
        with h5py.File(rec["demo_path"], "r") as f:
            obs = f["data"][f"{rec['demo_key']}/obs"]
            frames = obs[_resolve_image_key(obs, None)]
            n = frames.shape[0]
            return [np.asarray(frames[min(int(t), n - 1)]) for t in timesteps]

    def task_of(self, episode: int) -> str:
        rec = self._by_ep.get(int(episode))
        return rec["task"] if rec else ""


# ---------------------------------------------------------------------------
# Feature selection and per-feature evidence
# ---------------------------------------------------------------------------
def select_features(prob: np.ndarray, is_active: np.ndarray, n: int,
                    strategy: str = "stratified", seed: int = 0,
                    n_bins: int = 10, top_frac: float = 0.4) -> np.ndarray:
    """Choose which features to label.

    Uniform random sampling is useless here: with ~99.5% of features
    memorized, 60 random draws yield almost no positives, and the paper's
    balanced 15-general/15-memorized set was CURATED (they searched for
    general candidates), not sampled.  We mirror that in two parts:

      1. Reserve ``top_frac`` of the budget for the highest-P features -- the
         general CANDIDATES to adjudicate.  Many will still be memorized
         (borderline negatives near the boundary), which is exactly what
         sharpens the fitted decision surface; the true generals, if any
         exist in this layer, are guaranteed to be shown rather than left to
         a bin lottery.
      2. Fill the rest by binning the remaining features by P(general) VALUE
         (equal-width) and drawing from every bin, so clear memorized
         negatives across the whole score range are represented.  Value bins
         rather than rank bins matter: with 99.5% of mass near zero, rank
         binning starves the tail.

    NOTE P is used only to PRIORITISE and SPREAD the sample.  It supplies no
    labels, and the annotator never sees it -- the reserved top features are
    candidates to be judged, not pre-labelled positives.
    """
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(is_active)
    if len(idx) <= n:
        return np.sort(idx)
    if strategy == "random":
        return np.sort(rng.choice(idx, size=n, replace=False))
    if strategy != "stratified":
        raise ValueError(f"unknown strategy {strategy!r}")

    p_all = prob[idx]
    order_desc = idx[np.argsort(p_all)[::-1]]

    # 1. reserve the top-P candidates
    n_top = int(np.floor(np.clip(top_frac, 0.0, 1.0) * n))
    top = order_desc[:n_top]

    # 2. stratify the remaining (lower-P) pool by value bins
    rest = order_desc[n_top:]
    n_rem = n - len(top)
    p = prob[rest]
    lo, hi = float(p.min()), float(p.max()) if len(p) else (0.0, 0.0)
    if len(rest) <= n_rem:
        rem_pick = rest
    elif hi <= lo:
        rem_pick = rng.choice(rest, size=n_rem, replace=False)
    else:
        edges = np.linspace(lo, hi, n_bins + 1)
        bin_id = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
        groups = [rest[bin_id == b] for b in range(n_bins)]
        groups = [g for g in groups if len(g)]
        per = max(1, n_rem // len(groups))
        rem_pick = np.unique(np.concatenate(
            [rng.choice(g, size=min(per, len(g)), replace=False)
             for g in groups]))
        if len(rem_pick) < n_rem:               # top up from the remainder
            leftover = np.setdiff1d(rest, rem_pick)
            if len(leftover):
                rem_pick = np.unique(np.concatenate([
                    rem_pick,
                    rng.choice(leftover, size=min(n_rem - len(rem_pick),
                                                  len(leftover)),
                               replace=False)]))

    picked = np.unique(np.concatenate([top, rem_pick]))
    if len(picked) > n:                          # trim the stratified excess
        drop = rng.choice(np.setdiff1d(picked, top),
                          size=len(picked) - n, replace=False)
        picked = np.setdiff1d(picked, drop)
    return np.sort(picked)


def episode_peaks(zj: np.ndarray, episode: np.ndarray):
    """Per-episode peak activation. Returns (episode_ids, peaks)."""
    ep_ids = np.unique(episode)
    pos = np.searchsorted(ep_ids, episode)
    peaks = np.zeros(len(ep_ids), dtype=np.float64)
    np.maximum.at(peaks, pos, zj)
    return ep_ids, peaks


def top_episodes(zj: np.ndarray, episode: np.ndarray, k: int = 10):
    """The k episodes with the highest peak activation, descending."""
    ep_ids, peaks = episode_peaks(zj, episode)
    order = np.argsort(peaks)[::-1][:k]
    return ep_ids[order], peaks[order]


def episode_trace(zj: np.ndarray, episode: np.ndarray, timestep: np.ndarray,
                  ep_id: int):
    """(timesteps, activations) for one episode, ordered in time."""
    mask = episode == ep_id
    order = np.argsort(timestep[mask], kind="stable")
    return timestep[mask][order], zj[mask][order]


# ---------------------------------------------------------------------------
# Card rendering
# ---------------------------------------------------------------------------
def render_card(feat_id: int, zj: np.ndarray, episode: np.ndarray,
                timestep: np.ndarray, frames: FrameSource, out_path: str,
                k_episodes: int = 8, n_context: int = 6) -> None:
    """Render one feature card.

    Panels:
      A  activation traces over the top-k episodes, time-normalised
         (burst-locked vs sustained -- paper Stage 1)
      B  the camera frame at each top episode's PEAK activation
         (same event across diverse scenes? -- paper Stage 2)
      C  frames spanning the peak of the single top episode (temporal context)
      D  relative peak drop-off across the top episodes, normalised to the
         maximum.  The paper's Stage 2 uses drop-off shape as a diagnostic;
         normalising hides the absolute scale, which is what mean activation
         magnitude measures.

    No computed statistic is printed anywhere on the card.
    """
    ep_top, peaks = top_episodes(zj, episode, k_episodes)
    has_frames = frames.available

    n_rows = 4 if has_frames else 2
    fig = plt.figure(figsize=(14, 3.1 * n_rows))
    gs = fig.add_gridspec(n_rows, 1, hspace=0.55)

    # --- Panel A: traces ---
    axA = fig.add_subplot(gs[0])
    cmap = plt.get_cmap("viridis")
    for i, ep in enumerate(ep_top):
        ts, vals = episode_trace(zj, episode, timestep, ep)
        x = np.linspace(0, 1, len(vals)) if len(vals) > 1 else np.array([0.0])
        axA.plot(x, vals, lw=1.3, alpha=0.85,
                 color=cmap(i / max(len(ep_top) - 1, 1)),
                 label=f"ep{int(ep)}")
    axA.set_xlabel("normalised time within episode")
    axA.set_ylabel("activation")
    axA.set_title(f"A. Activation over the top {len(ep_top)} episodes  "
                  f"— bursty and event-locked, or sustained?", fontsize=10)
    axA.legend(fontsize=6, ncol=min(len(ep_top), 8), loc="upper right")

    # --- Panel D: normalised drop-off ---
    axD = fig.add_subplot(gs[1])
    rel = peaks / max(peaks.max(), 1e-12)
    axD.bar(range(len(rel)), rel, color="#4c72b0")
    axD.set_xticks(range(len(rel)))
    axD.set_xticklabels([f"ep{int(e)}" for e in ep_top], fontsize=7,
                        rotation=45, ha="right")
    axD.set_ylabel("peak / max peak")
    axD.set_ylim(0, 1.05)
    axD.set_title("D. Relative peak across top episodes — a steep drop-off "
                  "after one or two episodes suggests memorization",
                  fontsize=10)

    if has_frames:
        # --- Panel B: peak frame per top episode ---
        axB = fig.add_subplot(gs[2])
        axB.axis("off")
        axB.set_title("B. Frame at each episode's PEAK activation — is the "
                      "same event happening, across different scenes?",
                      fontsize=10)
        n_show = min(len(ep_top), 8)
        for i in range(n_show):
            ep = ep_top[i]
            ts, vals = episode_trace(zj, episode, timestep, ep)
            t_peak = ts[int(np.argmax(vals))]
            imgs = frames.get(ep, [t_peak])
            sub = axB.inset_axes([i / n_show, 0.0, 1 / n_show * 0.94, 0.88])
            sub.axis("off")
            if imgs:
                sub.imshow(imgs[0])
            sub.set_title(f"ep{int(ep)}  t={int(t_peak)}\n"
                          f"{frames.task_of(ep)[:26]}", fontsize=6)

        # --- Panel C: temporal context on the top episode ---
        axC = fig.add_subplot(gs[3])
        axC.axis("off")
        ep0 = ep_top[0]
        ts0, vals0 = episode_trace(zj, episode, timestep, ep0)
        i_peak = int(np.argmax(vals0))
        lo = max(0, i_peak - n_context // 2)
        sel = list(range(lo, min(len(ts0), lo + n_context)))
        axC.set_title(f"C. Episode {int(ep0)} around its peak — does the "
                      f"activation switch on when a visible event begins?",
                      fontsize=10)
        imgs = frames.get(ep0, [ts0[i] for i in sel])
        for i, s in enumerate(sel):
            sub = axC.inset_axes([i / len(sel), 0.0, 1 / len(sel) * 0.94, 0.88])
            sub.axis("off")
            if imgs:
                sub.imshow(imgs[i])
            mark = " *" if s == i_peak else ""
            sub.set_title(f"t={int(ts0[s])}{mark}", fontsize=6)

    fig.suptitle(f"Feature {feat_id}", fontsize=13, y=0.995)
    fig.savefig(out_path, dpi=95, bbox_inches="tight")
    plt.close(fig)


LABEL_SHEET_CSS = """
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;background:#f6f7f9;color:#111}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:12px 20px;z-index:10;
 box-shadow:0 1px 4px rgba(0,0,0,.06)}
h1{font-size:16px;margin:0 0 4px}
.sub{font-size:12px;color:#555}
button{font-size:13px;padding:7px 14px;border-radius:5px;border:1px solid #3b6ea5;background:#3b6ea5;
 color:#fff;cursor:pointer;margin-right:8px}
button.sec{background:#fff;color:#3b6ea5}
.card{background:#fff;margin:18px 20px;padding:14px;border-radius:7px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
.card img{width:100%;height:auto;border:1px solid #e3e3e3;border-radius:4px}
.ctrl{margin-top:10px;display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.ctrl label{font-size:14px;cursor:pointer}
.ctrl input[type=radio]{margin-right:5px}
.notes{width:min(560px,90%);padding:5px;font-size:13px;border:1px solid #ccc;border-radius:4px}
.done{border-left:5px solid #4a8a5c}
.guide{background:#eef3f8;border-left:4px solid #5b86ad;padding:10px 14px;margin:14px 20px;
 font-size:13px;border-radius:5px;line-height:1.5}
.count{font-weight:bold}
"""

LABEL_SHEET_JS = """
function collect(){
  const rows=[["feature_id","label","confidence","notes"]];
  document.querySelectorAll('.card').forEach(c=>{
    const id=c.dataset.fid;
    const lab=c.querySelector('input[name="lab_'+id+'"]:checked');
    const conf=c.querySelector('input[name="conf_'+id+'"]:checked');
    const notes=c.querySelector('.notes').value.replace(/"/g,"'");
    if(lab) rows.push([id,lab.value,conf?conf.value:"",notes]);
  });
  return rows.map(r=>r.map(x=>'"'+x+'"').join(",")).join("\\n");
}
function download(){
  const blob=new Blob([collect()],{type:"text/csv"});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob);
  a.download="labels_"+(document.getElementById("who").value||"annotator")+".csv";
  a.click();
}
function save(){localStorage.setItem("featlabels",collect());upd();}
function restore(){
  const d=localStorage.getItem("featlabels"); if(!d)return;
  d.split("\\n").slice(1).forEach(line=>{
    const m=line.match(/"([^"]*)","([^"]*)","([^"]*)","([^"]*)"/); if(!m)return;
    const [_,id,lab,conf,notes]=m;
    const r=document.querySelector('input[name="lab_'+id+'"][value="'+lab+'"]'); if(r)r.checked=true;
    if(conf){const c=document.querySelector('input[name="conf_'+id+'"][value="'+conf+'"]');if(c)c.checked=true;}
    const c=document.querySelector('.card[data-fid="'+id+'"] .notes'); if(c)c.value=notes;
  });
  upd();
}
function upd(){
  let n=0;
  document.querySelectorAll('.card').forEach(c=>{
    const done=c.querySelector('input[type=radio]:checked')!==null;
    c.classList.toggle('done',done); if(done)n++;
  });
  document.getElementById("n").textContent=n;
}
document.addEventListener("DOMContentLoaded",()=>{
  restore();
  document.addEventListener("change",e=>{if(e.target.type==="radio"){save();}});
});
"""


def write_label_sheet(feat_ids, card_names, out_path: str, layer: int) -> None:
    """Self-contained HTML labelling sheet with CSV export and autosave."""
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>Feature labelling — layer {layer:02d}</title>",
        f"<style>{LABEL_SHEET_CSS}</style>",
        f"<script>{LABEL_SHEET_JS}</script></head><body>",
        "<header>",
        f"<h1>SAE feature labelling — layer {layer:02d}</h1>",
        "<div class='sub'>Labelled: <span class='count' id='n'>0</span> / "
        f"{len(feat_ids)} &nbsp;·&nbsp; progress autosaves in this browser</div>",
        "<div style='margin-top:8px'>",
        "Annotator: <input id='who' class='notes' style='width:150px' "
        "placeholder='your name'> ",
        "<button onclick='download()'>Download CSV</button>",
        "<button class='sec' onclick='save()'>Save progress</button>",
        "</div></header>",
        "<div class='guide'><b>The question for each feature:</b> when it "
        "spikes, is the same recognisable thing happening on screen — across "
        "<i>different</i> scenes?<br>"
        "<b>general</b> — bursts lock to a consistent sensorimotor event "
        "(grasp, placement, object appearing) and recur across visually "
        "diverse episodes.<br>"
        "<b>memorized</b> — activation is sustained rather than event-locked, "
        "and/or the top episodes are all the same one or two scenes "
        "(steep drop-off in panel D).<br>"
        "<b>unclear</b> — ambiguous; excluded from the fit, as the paper also "
        "does. Use it freely; forced guesses add noise.</div>",
    ]
    for fid, card in zip(feat_ids, card_names):
        parts.append(
            f'<div class="card" data-fid="{fid}">'
            f'<b>Feature {fid}</b><br><img src="cards/{card}" loading="lazy">'
            f'<div class="ctrl"><span>'
            f'<label><input type="radio" name="lab_{fid}" value="general">general</label> '
            f'<label><input type="radio" name="lab_{fid}" value="memorized">memorized</label> '
            f'<label><input type="radio" name="lab_{fid}" value="unclear">unclear</label>'
            f'</span><span style="color:#777;font-size:13px">confidence:'
            f'<label><input type="radio" name="conf_{fid}" value="high">high</label>'
            f'<label><input type="radio" name="conf_{fid}" value="low">low</label>'
            f'</span>'
            f'<input class="notes" placeholder="notes (what event does it fire on?)">'
            f'</div></div>')
    parts.append("</body></html>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


# ---------------------------------------------------------------------------
# Logistic regression (Eq. 11) with ridge, fit by IRLS
# ---------------------------------------------------------------------------
def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                 n_iter: int = 100, tol: float = 1e-9) -> np.ndarray:
    """Ridge-penalised logistic MLE via Newton-Raphson (IRLS).

    X is [n, p] WITHOUT an intercept column (added here); y in {0, 1}.
    The intercept is not penalised, matching the usual convention.  A penalty
    is required rather than optional: hand-labelled sets of this size are
    typically linearly separable, and the unpenalised MLE then diverges.

    Returns coefficients [1 + p] ordered (intercept, *columns of X).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    A = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(p + 1)
    pen = np.full(p + 1, l2)
    pen[0] = 0.0
    for _ in range(n_iter):
        eta = A @ w
        mu = sigmoid(eta)
        s = np.clip(mu * (1 - mu), 1e-9, None)
        grad = A.T @ (y - mu) - pen * w
        H = (A * s[:, None]).T @ A + np.diag(pen)
        step = np.linalg.solve(H + 1e-10 * np.eye(p + 1), grad)
        w = w + step
        if np.max(np.abs(step)) < tol:
            break
    return w


def loo_cv_accuracy(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> float:
    """Leave-one-out cross-validated accuracy, as reported in the paper."""
    n = len(y)
    correct = 0
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        w = fit_logistic(X[m], y[m], l2=l2)
        p = sigmoid(np.array([1.0, *X[i]]) @ w)
        correct += int((p >= 0.5) == bool(y[i]))
    return correct / max(n, 1)


def single_metric_auc(values: np.ndarray, labels: np.ndarray) -> float:
    """AUC of one metric for predicting the label (1=general, 0=memorized).

    AUC = P(a random general scores above a random memorized) on this metric.
    0.5 = the metric does not separate the two label groups at all; >0.7 =
    clear separation in the paper's expected direction; <0.5 = separation in
    the OPPOSITE direction (e.g. labelled generals have LOWER onset counts).
    Computed from mean ranks (Mann-Whitney), tie-corrected.
    """
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    pos, neg = (labels == 1), (labels == 0)
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    vals, inv, counts = np.unique(values, return_inverse=True,
                                  return_counts=True)
    mean_rank = np.bincount(inv, weights=ranks) / counts
    ranks = mean_rank[inv]                       # average ties
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) /
                 (n_pos * n_neg))


def cohens_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two annotators over the same items."""
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    M = np.zeros((k, k))
    for x, y in zip(a, b):
        M[idx[x], idx[y]] += 1
    n = M.sum()
    if n == 0:
        return 0.0
    po = np.trace(M) / n
    pe = float((M.sum(0) / n) @ (M.sum(1) / n))
    return 0.0 if abs(1 - pe) < 1e-12 else (po - pe) / (1 - pe)


def read_labels(path: str) -> dict[int, str]:
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lab = (row.get("label") or "").strip().lower()
            if lab in ("general", "memorized"):
                out[int(row["feature_id"])] = lab
    return out


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
def cmd_cards(args) -> None:
    codes = np.load(os.path.join(args.codes_dir, f"layer_{args.layer:02d}.npz"))
    z = codes["z"].astype(np.float32)
    episode = codes["episode"].astype(np.int64)
    timestep = codes["timestep"].astype(np.int64)

    gen_path = os.path.join(args.generality_dir,
                            f"layer_{args.layer:02d}_generality.npz")
    g = np.load(gen_path)
    prob = g["prob_general"].astype(np.float64)
    is_active = g["is_active"].astype(bool) if "is_active" in g \
        else np.ones(z.shape[1], dtype=bool)

    feats = select_features(prob, is_active, args.n_features,
                            strategy=args.strategy, seed=args.seed)
    print(f"[cards] layer {args.layer:02d}: selected {len(feats)} of "
          f"{int(is_active.sum())} active features "
          f"(strategy={args.strategy})", flush=True)

    # episode -> demo mapping for frames
    episode_index = None
    if args.episode_index and os.path.exists(args.episode_index):
        episode_index = json.load(open(args.episode_index))
        print(f"[cards] loaded episode index ({len(episode_index)} episodes)")
    elif args.task_suite:
        try:
            episode_index = build_episode_index(
                args.task_suite, args.max_tasks, args.max_demos_per_task)
            if args.episode_index:
                json.dump(episode_index, open(args.episode_index, "w"))
            print(f"[cards] built episode index ({len(episode_index)} episodes)")
        except Exception as e:                       # libero/HDF5 unavailable
            print(f"[cards] WARNING: no frames ({type(e).__name__}: {e}).\n"
                  f"        Cards will show traces only, which is NOT enough "
                  f"for the paper's protocol — you need the images to judge "
                  f"whether the same event recurs across scenes.", flush=True)
    frames = FrameSource(episode_index)

    cards_dir = os.path.join(args.out_dir, "cards")
    os.makedirs(cards_dir, exist_ok=True)
    names = []
    for n, fid in enumerate(feats, 1):
        name = f"feat_{int(fid):04d}.png"
        render_card(int(fid), z[:, fid], episode, timestep, frames,
                    os.path.join(cards_dir, name),
                    k_episodes=args.top_episodes)
        names.append(name)
        if n % 10 == 0 or n == len(feats):
            print(f"  rendered {n}/{len(feats)}", flush=True)

    sheet = os.path.join(args.out_dir, "label_sheet.html")
    write_label_sheet([int(f) for f in feats], names, sheet, args.layer)

    with open(os.path.join(args.out_dir, "labels_template.csv"), "w",
              newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["feature_id", "label", "confidence", "notes"])
        for fid in feats:
            wtr.writerow([int(fid), "", "", ""])

    # hidden until labelling is done
    hidden = {"layer": args.layer, "features": [int(f) for f in feats],
              "metrics": {k: [float(g[k][f]) for f in feats]
                          for k in METRIC_KEYS},
              "prob_general_paper_beta": [float(prob[f]) for f in feats]}
    with open(os.path.join(args.out_dir, "hidden_metrics.json"), "w") as f:
        json.dump(hidden, f, indent=2)

    print(f"\n[cards] label here: {sheet}")
    print(f"[cards] metrics withheld -> "
          f"{os.path.join(args.out_dir, 'hidden_metrics.json')} "
          f"(do not open before labelling)", flush=True)


def cmd_fit(args) -> None:
    labels = read_labels(args.labels)
    hidden = json.load(open(args.metrics))
    feats = hidden["features"]
    pos = {f: i for i, f in enumerate(feats)}

    use = [f for f in labels if f in pos]
    if len(use) < 8:
        raise SystemExit(f"only {len(use)} usable labels; need more")
    X = np.array([[hidden["metrics"][k][pos[f]] for k in METRIC_KEYS]
                  for f in use], dtype=np.float64)
    y = np.array([1.0 if labels[f] == "general" else 0.0 for f in use])

    n_gen = int(y.sum())
    print(f"[fit] {len(use)} labelled features "
          f"({n_gen} general, {len(use) - n_gen} memorized)")
    if n_gen == 0 or n_gen == len(use):
        raise SystemExit("labels are single-class; cannot fit")

    w = fit_logistic(X, y, l2=args.l2)
    acc = loo_cv_accuracy(X, y, l2=args.l2)
    ours = dict(zip(BETA_KEYS, w))

    print(f"\n[fit] refit coefficients (ridge l2={args.l2}), Eq. (11) order:")
    print(f"  {'term':<20} {'ours':>9} {'paper (LIBERO)':>16}")
    for k in BETA_KEYS:
        print(f"  {k:<20} {ours[k]:>9.3f} {BETA_LIBERO[k]:>16.2f}")
    print(f"\n[fit] LOO-CV accuracy: {acc:.1%}  "
          f"(paper reports 100% on 30 labels, but its Stage-3 protocol "
          f"consults these same metrics when labelling)")

    p_ours = sigmoid(np.hstack([np.ones((len(X), 1)), X]) @ w)
    p_paper = sigmoid(
        BETA_LIBERO["intercept"]
        + BETA_LIBERO["mean_onsets"] * X[:, 0]
        + BETA_LIBERO["coverage"] * X[:, 1]
        + BETA_LIBERO["mean_act_magnitude"] * X[:, 2]
        + BETA_LIBERO["rel_run_length"] * X[:, 3])
    agree = float(((p_ours >= 0.5) == (p_paper >= 0.5)).mean())
    acc_paper = float(((p_paper >= 0.5) == (y == 1)).mean())
    print(f"[fit] paper-beta accuracy on our labels: {acc_paper:.1%}")
    print(f"[fit] decision agreement, ours vs paper beta: {agree:.1%}")

    out = {"n_labelled": len(use), "n_general": n_gen,
           "coefficients": ours, "paper_coefficients": BETA_LIBERO,
           "loo_cv_accuracy": acc, "paper_beta_accuracy": acc_paper,
           "decision_agreement": agree, "l2": args.l2,
           "features": use,
           "labels": [labels[f] for f in use]}
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        p = os.path.join(args.out_dir, "refit_classifier.json")
        json.dump(out, open(p, "w"), indent=2)
        print(f"[fit] wrote {p}")


def cmd_diagnose(args) -> None:
    """Is the labelling aligned with the metrics, and in which direction?

    For each of the four classifier metrics, report the mean over the
    general-labelled vs memorized-labelled features and the single-metric AUC.
    This separates two explanations for a flat refit: noisy labels (no metric
    separates, all AUC ~0.5) versus a layer where generality is simply not
    expressed through these metrics (labels may still be internally consistent
    but the paper's statistics do not capture what the annotator saw).
    """
    labels = read_labels(args.labels)
    hidden = json.load(open(args.metrics))
    feats = hidden["features"]
    pos = {f: i for i, f in enumerate(feats)}
    use = [f for f in labels if f in pos]
    y = np.array([1 if labels[f] == "general" else 0 for f in use])
    n_gen = int(y.sum())
    print(f"[diagnose] {len(use)} labelled  ({n_gen} general, "
          f"{len(use) - n_gen} memorized)")
    if n_gen == 0 or n_gen == len(use):
        raise SystemExit("labels are single-class; nothing to separate")

    print(f"\n  {'metric':<16} {'general':>9} {'memorized':>10} {'AUC':>6}  "
          f"direction")
    expected_sign = {"mean_onsets": +1, "coverage": +1, "mean_act_mag": +1,
                     "rel_run_length": -1}
    flat = []
    for k in METRIC_KEYS:
        v = np.array([hidden["metrics"][k][pos[f]] for f in use])
        auc = single_metric_auc(v, y)
        gm, mm = v[y == 1].mean(), v[y == 0].mean()
        # is separation in the paper's expected direction?
        want = expected_sign[k]
        ok = "as paper expects" if (auc - 0.5) * want > 0.05 else \
             ("OPPOSITE to paper" if (auc - 0.5) * want < -0.05 else
              "no separation")
        if abs(auc - 0.5) < 0.1:
            flat.append(k)
        print(f"  {k:<16} {gm:>9.3f} {mm:>10.3f} {auc:>6.2f}  {ok}")

    print()
    if len(flat) == len(METRIC_KEYS):
        print("  VERDICT: no metric separates your labels (all AUC ~0.5).")
        print("    Either the labels are inconsistent, OR this layer does not")
        print("    express generality through these four metrics. Check a")
        print("    layer with known bursty candidates (e.g. layer 31) to tell")
        print("    these apart: if the metrics separate there, the labels are")
        print("    fine and this layer is simply weak; if not, revisit the")
        print("    labelling protocol.")
    elif "mean_onsets" in flat:
        print("  NOTE: onset count (the paper's strongest predictor) does not")
        print("    separate your labels on this layer. If coverage still does,")
        print("    your generals are broad but not bursty here -- consistent")
        print("    with a layer whose top candidates had onset ~1.")
    else:
        print("  Metrics do separate your labels; a flat logistic refit is")
        print("    then likely collinearity + ridge shrinkage. Re-run `fit`")
        print("    with a smaller --l2 (e.g. 0.1) and read LOO-CV, not the")
        print("    individual coefficients.")


def cmd_agreement(args) -> None:
    a, b = (read_labels(p) for p in args.labels)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("no features labelled by both annotators")
    la, lb = [a[f] for f in shared], [b[f] for f in shared]
    agree = float(np.mean([x == y for x, y in zip(la, lb)]))
    k = cohens_kappa(la, lb)
    print(f"[agreement] {len(shared)} shared features")
    print(f"  raw agreement : {agree:.1%}")
    print(f"  Cohen's kappa : {k:.3f}")
    band = ("poor" if k < 0.2 else "fair" if k < 0.4 else "moderate"
            if k < 0.6 else "substantial" if k < 0.8 else "almost perfect")
    print(f"  interpretation: {band}")
    if k < 0.4:
        print("  NOTE: low agreement is itself a finding — it indicates the "
              "general/memorized distinction is not reliably annotatable at "
              "this resolution (Phase 2-D in EXPERIMENT_PLAN.md).")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cards", help="render feature cards + labelling sheet")
    c.add_argument("--codes-dir", required=True)
    c.add_argument("--generality-dir", required=True)
    c.add_argument("--layer", type=int, required=True)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--n-features", type=int, default=60)
    c.add_argument("--strategy", default="stratified",
                   choices=("stratified", "random"))
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--top-episodes", type=int, default=8)
    c.add_argument("--task-suite", default="libero_goal",
                   help="LIBERO suite used at collection time (for frames)")
    c.add_argument("--max-tasks", type=int, default=None)
    c.add_argument("--max-demos-per-task", type=int, default=None)
    c.add_argument("--episode-index", default=None,
                   help="JSON cache of the episode->demo map")
    c.set_defaults(func=cmd_cards)

    f = sub.add_parser("fit", help="fit Eq. (11) on completed labels")
    f.add_argument("--labels", required=True)
    f.add_argument("--metrics", required=True)
    f.add_argument("--out-dir", default=None)
    f.add_argument("--l2", type=float, default=1.0)
    f.set_defaults(func=cmd_fit)

    d = sub.add_parser("diagnose",
                       help="per-metric separability of your labels (AUC)")
    d.add_argument("--labels", required=True)
    d.add_argument("--metrics", required=True)
    d.set_defaults(func=cmd_diagnose)

    a = sub.add_parser("agreement", help="Cohen's kappa between annotators")
    a.add_argument("--labels", nargs=2, required=True)
    a.set_defaults(func=cmd_agreement)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
