# Methodology

**Model family:** chamber-specific ridge regressions over vintage-safe
features, predicting the Democratic two-party margin (`app/model.py`,
coefficients stored as versioned data in `model_versions`).

Three fits per chamber, routed by what data actually exists for a race so the
model never extrapolates through a feature absent at prediction time:

* **full** — fundamentals + time-decayed race-polling average (21-day
  half-life, partisan polls down-weighted); applied to races with polls.
* **fundamentals** — race-poll columns excluded, generic-ballot columns kept;
  applied to unpolled races when national generic-ballot polling exists for
  the cycle *and* the champion spec uses it (see below).
* **core** — seat history, seat-holder party, and president-party environment
  only; applied when neither race polls nor usable national polling exist.

**Champion / challenger discipline:** every pipeline run re-evaluates the
champion spec against challenger specs (currently per-state partial-pooled
residual offsets, and a generic-ballot variant) and five naive baselines
under the identical expanding-window protocol. The raw generic-ballot
average is *excluded* from the champion because it degraded held-out
accuracy in both chambers when first tested (research claim N-001); it is
automatically re-tested every run and will be promoted only on evidence.
Per-race predictions from every model are stored, and
`/api/races/{id}/models` exposes where they disagree.

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
