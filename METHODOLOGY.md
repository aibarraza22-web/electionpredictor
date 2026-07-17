# Methodology

**Model family:** chamber-specific ridge regressions over vintage-safe
features, predicting the Democratic two-party margin (`app/model.py`,
coefficients stored as versioned data in `model_versions`).

Two fits per chamber:

* **full** — fundamentals + time-decayed polling average (21-day half-life,
  partisan polls down-weighted); applied to races with polls.
* **fundamentals** — poll columns excluded, fit on all historical races;
  applied to unpolled races so their weights are estimated honestly rather
  than extrapolated from the polled-race intercept.

Features (`app/features.py`): seat prior margin (most recent same-seat
result, clipped), prior availability flag, seat-holder party, president-party
environment sign, its midterm interaction, poll average, and poll
availability. Every feature for cycle *t* uses only information available
before that election; missing inputs are flagged, never imputed.

**Uncertainty:** each fit's training-residual standard deviation
(polled/unpolled pools), plus added variance for seats without history.
Margins map to probabilities with a normal CDF; ratings are labels over
probabilities, never substitutes for them.

**Control simulation** (`app/simulation.py`): 25,000 seeded draws in margin
space with a shared 3.5-point national shock plus race-specific noise,
preserving correlated errors. Senate control includes the explicit
tie-break assumption and the count of Democratic-caucus seats not up
(derived from ingested term data).

Ranked-choice (AK/ME) and runoff mechanics are flagged per race and remain
registered challengers until genuine out-of-sample evidence supports
promotion.
