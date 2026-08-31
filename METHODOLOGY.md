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

**Expert-ratings overlay (`app/ratings.py`, research claim R-001):** the
published margin is the fitted model blended with a reading of the seat's
published expert consensus:

```
implied(seat) = level + slope x (consensus(seat) - mean consensus)
published     = (1 - w) x model(seat) + w x implied(seat)
```

`consensus` is the unweighted mean of each rater's *latest* signed call on
the shared Safe/Likely/Lean/Tilt/Tossup ladder (Democratic positive).
`level` is the model's own mean prediction across the cycle's rated seats, so
the national environment still comes entirely from the fitted model and no
outcome is ever consulted. `slope` is fitted in within-cycle deviation form on
strictly earlier cycles — 3.9 margin points per rating step for the House,
6.1 for the Senate, stable in every held-out cycle. `w` is chosen by held-out
log loss per chamber and separately for polled and unpolled seats, capped at
0.75 so polls and seat history always retain weight.

Walk-forward over 2016–2024, scored on the seats it applies to, this is the
largest accuracy gain in the model's history. House, 490 held-out rated
seats: Brier 0.1642 → 0.1325, log loss 0.5251 → 0.4345, winner accuracy
0.7755 → 0.8449, margin MAE 9.69 → 6.00. Senate, 66 seats: Brier 0.0871 →
0.0763, log loss 0.3594 → 0.3388, winner accuracy 0.8788 → 0.8939 — but
margin MAE 11.34 → 11.99, so the Senate buys better win probabilities at the
cost of slightly worse point margins on a small sample; the weight is chosen
on log loss, which is the metric the chamber champions are also picked on.
The overlay recomputes this scoreboard on every run and publishes it at
`/api/data-health`; nothing here is hand-entered. It matters most where the
model was previously blind — only 43 of 470 2026 races carry any polling. Rated-seat sigma is re-estimated from
the blended walk-forward residuals (the pooled sigma, fitted over uncontested
blowouts too, badly over-covered competitive races) and inflated 1.45x for the
gap between the final-vintage ratings it is fitted on and the ~2-months-out
ratings it is applied to.

Two things it deliberately does not do. It is **not** a model feature: added
to the ridge as a `(rating_consensus, has_rating)` pair it made rated seats
*worse*, because `has_rating` is a selection indicator and one global
coefficient turns that selection into a biased constant (claim R-002). And it
does not touch House seats every rater calls safe, which sit outside the
population its slope was fitted on; where the model calls such a seat
competitive anyway, the run reports the disagreement
(`competitive_but_consensus_safe`) rather than splitting the difference.

**Campaign-development layer (`app/campaign.py`):** every champion snapshot
also freezes a structural baseline, polling contribution, decisive win-size
probabilities, stage/opponent-relative FEC context, candidate observations,
and a source-backed event ledger. Model 2026.18 applies a provisional bounded
overlay using finance capacity, candidate-quality asymmetry, and explicitly
eligible events. Ordinary effects are capped at three margin points and only
exceptional events can expand the cap to six. Recent polls absorb most
pre-existing campaign information, thin or stale inputs are discounted, and
active adjustments add uncertainty. The project's prequential test still
rejects raw receipts disparity, so the overlay is reported separately from
the validated ridge model and is not described as a proven accuracy gain.

**Release gates (`app/gates.py`, claim R-003):** before any snapshot is
frozen, every competitive race must carry data grade A or B, a new model
version must move at least 75% of comparable competitive races against the
outgoing champion, and the ratings feed must have delivered current-cycle
coverage. A failure refuses the publish and leaves the previous forecast
live. This exists because model 2026.18 shipped a campaign layer whose
candidate-profile and campaign-event feeds were never configured in
production; it ran finance-only and moved the toplines by 0.1 point in the
House and 1.4 in the Senate, and nothing noticed.

**Uncertainty:** each fit's training-residual standard deviation
(polled/unpolled pools), plus added variance for seats without history and
for any active provisional campaign overlay, and the overlay's own
rated-seat residual sigma where it applies.
Margins map to probabilities with a normal CDF; ratings are labels over
probabilities, never substitutes for them.

**Control simulation** (`app/simulation.py`): 25,000 seeded draws in margin
space with an empirically estimated shared national shock plus race-specific noise,
preserving correlated errors. Senate control includes the explicit
tie-break assumption and the count of Democratic-caucus seats not up
(derived from ingested term data).

Ranked-choice (AK/ME) and runoff mechanics are flagged per race and remain
registered challengers until genuine out-of-sample evidence supports
promotion.

Snapshots use UTC timestamps rather than one date per day. A content
fingerprint prevents unchanged inputs from creating duplicate snapshots.
