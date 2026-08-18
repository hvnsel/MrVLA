"""Is the A x B null real, or two noisy measurements failing to correlate?

Path B's boundary condition is `corr(breadth, recurrence) = -0.127` -- general is not
recurrent. The referee response is that both quantities are noisy (SAE dictionaries are only
~60% seed-reproducible), and two unreliable measures cannot correlate strongly no matter what
is true underneath. Without an answer the null is "could not measure", which analysis
commitment #4 forbids conflating with "not there".

The tool is the classical attenuation correction. With test-retest reliabilities r_xx, r_yy,

    r_obs = r_true * sqrt(r_xx * r_yy)                        (attenuation)
    |r_obs| <= sqrt(r_xx * r_yy) =: CEILING                   (what any measurement could show)

MIND THE DIRECTION -- it is the opposite of what it first looks like:

  * For the PATH A positive (+0.493) the correction is free. Attenuation only ever shrinks an
    observed correlation, so r_obs / sqrt(r_xx) is a LOWER BOUND on the truth even with the
    other reliability unknown. The result survives, strengthened.

  * For the A x B NULL it is not free, and plugging r_yy = 1 does NOT rescue it. That
    substitution yields a lower bound on |r_true| too -- the wrong direction entirely, since
    defending a null needs an UPPER bound. What defends the null is a high CEILING: if the
    measurements could have shown |r| up to 0.85 and returned 0.127, the dissociation is
    real; if the ceiling is itself near 0.15, nothing was ever measurable. The ceiling depends
    on r_yy, so THE NULL CANNOT BE DEFENDED WITHOUT AN ESTIMATE OF RECURRENCE RELIABILITY.

So when r_yy is unknown this script does not pretend otherwise. It inverts the question and
reports the BREAKEVEN reliability: how reliable would recurrence have to be for the observed
value to bound the truth below a threshold you name. That converts an unanswerable question
into a concrete, cheap measurement to go and make -- q_cross recomputed on disjoint halves of
the probe frames, which needs no new rollouts and no retraining.

Reliability inputs:
  * breadth   -- Spearman-Brown corrected split-half from `split_half_breadth.py`, a genuine
    test-retest across disjoint task halves. The right input.
  * recurrence -- pass `--rel-b` only if a real test-retest estimate exists. Do NOT substitute
    the SAE seed-match q_seed: that is a matching quality between dictionaries, not the
    retest reliability of the per-feature recurrence score, and using it overstates the case.

Usage
-----
python reliability_ceiling.py --split-half $B/ATTR/split_half_breadth.json \
                              --join $B/RECURRENCE_ACTION/join_pathA_pathB.json \
                              --partial 0.493
python reliability_ceiling.py --rel-a 0.72 --corr -0.127 --rel-b 0.55
"""

from __future__ import annotations

import argparse
import json
import math


def disattenuate(r_obs: float, rel_a: float, rel_b: float) -> dict:
    """Attenuation-corrected correlation and the ceiling, with both reliabilities known."""
    if not (0 < rel_a <= 1) or not (0 < rel_b <= 1):
        return {"error": "reliabilities must lie in (0, 1]"}
    ceiling = math.sqrt(rel_a * rel_b)
    return {"r_observed": r_obs, "reliability_a": rel_a, "reliability_b": rel_b,
            "ceiling": ceiling, "r_corrected": r_obs / ceiling,
            "saturated": abs(r_obs) > ceiling}


def lower_bound_one_reliability(r_obs: float, rel_a: float) -> dict:
    """|r_true| >= |r_obs| / sqrt(rel_a), valid when the other reliability is unknown.

    Sound for STRENGTHENING a positive result. Useless for defending a null, which needs an
    upper bound -- see `breakeven_reliability`.
    """
    if not 0 < rel_a <= 1:
        return {"error": "reliability must lie in (0, 1]"}
    return {"r_observed": r_obs, "reliability_a": rel_a,
            "max_possible_ceiling": math.sqrt(rel_a),
            "abs_r_true_lower_bound": abs(r_obs) / math.sqrt(rel_a)}


def breakeven_reliability(r_obs: float, rel_a: float, threshold: float = 0.30) -> dict:
    """How reliable must the OTHER measure be for the null to hold?

    Requiring |r_true| = |r_obs| / sqrt(rel_a * rel_b) <= threshold gives
    rel_b >= r_obs^2 / (rel_a * threshold^2). Also reports the reliability at which the
    measurement is fully saturated (|r_true| = 1), below which the observed correlation is
    consistent with ANY underlying relationship and says nothing at all.
    """
    if not 0 < rel_a <= 1 or threshold <= 0:
        return {"error": "need 0 < rel_a <= 1 and threshold > 0"}
    need = r_obs ** 2 / (rel_a * threshold ** 2)
    saturate = r_obs ** 2 / rel_a
    return {"threshold": threshold,
            "rel_b_needed": need,
            "rel_b_needed_feasible": need <= 1.0,
            "rel_b_saturating": saturate}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-half", default=None,
                    help="split_half_breadth.py json (reliability of breadth)")
    ap.add_argument("--suite", default=None, help="suite key in the split-half json")
    ap.add_argument("--which", default="adjusted", choices=["adjusted", "raw_PR"],
                    help="which breadth ranking's reliability (default adjusted, the ranking "
                         "features are actually selected on)")
    ap.add_argument("--rel-a", type=float, default=None, help="breadth reliability")
    ap.add_argument("--rel-b", type=float, default=None,
                    help="recurrence reliability, ONLY if a test-retest estimate exists")
    ap.add_argument("--join", default=None, help="join_pathA_pathB.py json")
    ap.add_argument("--corr", type=float, default=None,
                    help="observed corr(breadth, recurrence)")
    ap.add_argument("--partial", type=float, default=None, help="Path A partial|both")
    ap.add_argument("--threshold", type=float, default=0.30,
                    help="|r_true| below this counts as 'no relationship' (default 0.30)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rel_a = args.rel_a
    if rel_a is None and args.split_half:
        with open(args.split_half) as f:
            sh = json.load(f)
        suites = sh.get("suites", sh)
        key = args.suite or next(iter(suites))
        rel_a = suites[key][args.which].get("sb_median")
        print(f"[rel] breadth reliability from {args.split_half} [{key}/{args.which}]: {rel_a}")
    if rel_a is None:
        raise SystemExit("need --rel-a or --split-half (with a usable sb_median)")

    r_ab = args.corr
    if r_ab is None and args.join:
        with open(args.join) as f:
            j = json.load(f)
        r_ab = j.get("corr_breadth_qcross")
        print(f"[rel] observed corr(breadth, recurrence) from {args.join}: {r_ab}")

    out: dict = {"reliability_breadth": rel_a, "reliability_recurrence": args.rel_b,
                 "threshold": args.threshold}

    if r_ab is not None:
        r_ab = float(r_ab)
        print("\n=== the A x B null: corr(breadth, recurrence) ===")
        print(f"  observed                       : {r_ab:+.3f}")
        print(f"  breadth reliability            : {rel_a:.3f}")
        if args.rel_b is not None:
            d = disattenuate(r_ab, float(rel_a), float(args.rel_b))
            out["a_x_b"] = d
            if "error" in d:
                print(f"  {d['error']}")
            else:
                print(f"  recurrence reliability         : {d['reliability_b']:.3f}")
                print(f"  measurement ceiling            : +-{d['ceiling']:.3f}"
                      "   <- the largest |r| these measurements could ever show")
                print(f"  attenuation-corrected r_true   : {d['r_corrected']:+.3f}")
                if d["saturated"]:
                    print("  WARNING |r_obs| exceeds the ceiling: a reliability estimate is "
                          "too low to be consistent with the observed correlation.")
                elif abs(d["r_corrected"]) <= args.threshold:
                    print(f"  READING: corrected |r| stays under {args.threshold:.2f} while "
                          f"the ceiling was {d['ceiling']:.2f} -- the dissociation is REAL, "
                          "not a measurement failure.")
                else:
                    print(f"  READING: corrected |r| = {abs(d['r_corrected']):.3f} exceeds "
                          f"{args.threshold:.2f} -- once noise is accounted for this is a "
                          "relationship, not a null. Report it as such.")
        else:
            lb = lower_bound_one_reliability(r_ab, float(rel_a))
            be = breakeven_reliability(r_ab, float(rel_a), args.threshold)
            out["a_x_b_unknown_rel_b"] = {**lb, **be}
            print(f"  recurrence reliability         : UNKNOWN")
            print(f"  ceiling is at most             : +-{lb['max_possible_ceiling']:.3f} "
                  "(and lower for any real reliability)")
            print(f"  |r_true| >= {lb['abs_r_true_lower_bound']:.3f}  -- a LOWER bound, which "
                  "argues AGAINST a null, not for it")
            print(f"  BREAKEVEN: recurrence reliability must exceed {be['rel_b_needed']:.3f} "
                  f"for |r_true| <= {args.threshold:.2f}"
                  + ("" if be["rel_b_needed_feasible"] else "  (> 1: unattainable)"))
            print(f"  Below reliability {be['rel_b_saturating']:.3f} the observed value is "
                  "consistent with any underlying relationship whatsoever.")
            print("  ACTION: the null is not defensible until recurrence reliability is "
                  "measured. Cheapest estimate: recompute q_cross on disjoint halves of the "
                  "probe frames and correlate the two rankings (no rollouts, no retraining).")

    if args.partial is not None:
        p = float(args.partial)
        lb = lower_bound_one_reliability(p, float(rel_a))
        out["path_a_partial"] = lb
        print("\n=== the Path A positive: partial | both ===")
        print(f"  observed                       : {p:+.3f}")
        print(f"  breadth reliability            : {rel_a:.3f}")
        print(f"  |r_true| >= {lb['abs_r_true_lower_bound']:.3f}"
              "   <- attenuation only ever weakens a correlation, so this is a floor and needs "
              "no assumption about the other measure's reliability")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[rel] wrote {args.out}")


if __name__ == "__main__":
    main()
