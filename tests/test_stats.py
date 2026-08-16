"""Tests for mrvla.stats -- the small-sample machinery the ablation conclusions rest on.

The power bound is the load-bearing piece: it is what licenses "we can exclude damage larger
than X" from a null run, so it is pinned against the closed-form worst case, against its own
inverse, and for monotonicity in n.

Run directly:
    python tests/test_stats.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.stats import (  # noqa: E402
    mcnemar_exact_p, mcnemar_p, mde_paired, norm_cdf, norm_ppf, paired_diff_ci,
    required_pairs, wilson_interval,
)


def test_normal_helpers_round_trip():
    for p in (0.001, 0.025, 0.1, 0.5, 0.8, 0.975, 0.999):
        assert abs(norm_cdf(norm_ppf(p)) - p) < 1e-6
    assert abs(norm_ppf(0.975) - 1.959964) < 1e-4
    assert abs(norm_ppf(0.8) - 0.841621) < 1e-4
    assert abs(norm_cdf(0.0) - 0.5) < 1e-12


def test_wilson_is_asymmetric_at_the_boundary():
    """The reason for using Wilson at all: a normal-approx interval at k = n gives zero
    width, which would claim a 20/20 task is known perfectly."""
    lo, hi = wilson_interval(20, 20)
    assert hi <= 1.0 and lo < 1.0
    assert lo < 0.9          # 20/20 does not pin the rate above 90%
    lo, hi = wilson_interval(0, 20)
    assert lo >= 0.0 and hi > 0.0


def test_mcnemar_uses_only_discordant_pairs():
    """Concordant pairs cancel; a symmetric split is never significant however large."""
    assert math.isnan(mcnemar_p(0, 0))
    assert mcnemar_p(50, 50) > 0.9
    assert mcnemar_p(0, 12) < 0.01           # 12-0 one direction is decisive
    assert mcnemar_p(5, 12) > 0.05           # 12-5 is not, at that count


def test_exact_mcnemar_matches_hand_computed_binomial():
    # two-sided exact p for (b01, b10) = (0, 5) is 2 * P(X = 0), X ~ Bin(5, 0.5) = 2/32
    assert abs(mcnemar_exact_p(0, 5) - 2 / 32) < 1e-12
    # (1, 4): 2 * (C(5,0) + C(5,1))/32 = 12/32
    assert abs(mcnemar_exact_p(1, 4) - 12 / 32) < 1e-12
    assert mcnemar_exact_p(3, 3) == 1.0
    assert math.isnan(mcnemar_exact_p(0, 0))
    # the CONTINUITY-CORRECTED chi2 is conservative at small discordant counts: it reports a
    # larger p than the exact test and so throws away real power. That is why the per-task
    # table (m often < 10) is scored exactly.
    for pair in [(0, 4), (0, 6), (0, 8), (1, 7), (2, 8)]:
        assert mcnemar_exact_p(*pair) <= mcnemar_p(*pair) + 1e-12
    # both agree well once the discordant count is large
    assert abs(mcnemar_exact_p(20, 40) - mcnemar_p(20, 40)) < 0.02


def test_paired_diff_ci_sign_and_width():
    d, lo, hi = paired_diff_ci(b01=2, b10=22, n=200)
    assert abs(d - 0.10) < 1e-12             # (22-2)/200, positive = damage
    assert lo > 0 and hi > d                 # a real effect excludes zero
    d, lo, hi = paired_diff_ci(b01=12, b10=14, n=200)
    assert lo < 0 < hi                       # a near-tie does not
    # no discordant pairs -> zero difference and zero width
    assert paired_diff_ci(0, 0, 100) == (0.0, 0.0, 0.0)


def test_mde_matches_the_closed_form_worst_case():
    """MDE ~= (z_a + z_b) * sqrt(disc_rate / n) when the variance is taken at p = 0.5. The
    iterated version solves for the true p, so it must be at or below that bound."""
    n, disc = 200, 0.20
    bound = (norm_ppf(0.975) + norm_ppf(0.80)) * math.sqrt(disc / n)
    mde = mde_paired(n, disc)
    assert mde <= bound + 1e-12
    assert mde > 0.5 * bound                 # and not wildly below it either


def test_mde_shrinks_with_n_and_required_pairs_inverts_it():
    assert mde_paired(800, 0.2) < mde_paired(200, 0.2) < mde_paired(50, 0.2)
    # the run's actual geometry: one task (20 episodes) resolves far less than 200 pooled.
    # The ratio is below the sqrt(10) a fixed-variance formula would give, because solving
    # for p rather than pinning it at 0.5 shrinks the variance term at large effects.
    assert 2.0 < mde_paired(20, 0.2) / mde_paired(200, 0.2) < math.sqrt(10)
    # required_pairs is the p = 0.5 inverse, so feeding it the (smaller) iterated MDE asks
    # for at least the n we started from
    n, disc = 200, 0.2
    assert required_pairs(mde_paired(n, disc), disc) >= n - 1e-6


def test_mde_guards_bad_inputs():
    assert math.isnan(mde_paired(0, 0.2))
    assert math.isnan(mde_paired(100, 0.0))
    assert math.isnan(required_pairs(0.0, 0.2))


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all stats tests passed")
