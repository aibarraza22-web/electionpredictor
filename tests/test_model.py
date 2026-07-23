from random import Random

from app.features import FeatureRow
from app.model import MarginModel, ridge_fit


def _row(chamber, cycle, x, y, poll_count=0, has_prior=True):
    return FeatureRow(seat_key=f"{chamber}-XX-01", cycle=cycle, chamber=chamber,
                      state="XX", district=None, x=x, actual_margin=y,
                      poll_count=poll_count, has_prior=has_prior,
                      detail={"as_of": f"{cycle}-11-08"})


def test_ridge_recovers_linear_weights():
    rng = Random(7)
    true = [2.0, 0.8, 1.5]
    xs, ys = [], []
    for _ in range(400):
        x = [1.0, rng.uniform(-30, 30), rng.choice([-1.0, 1.0])]
        xs.append(x)
        ys.append(sum(w * v for w, v in zip(true, x)) + rng.gauss(0, 0.5))
    fitted = ridge_fit(xs, ys, l2=1e-6)
    assert all(abs(f - t) < 0.15 for f, t in zip(fitted, true))


def _training(rng, n=300):
    rows = []
    for i in range(n):
        prior = rng.uniform(-40, 40)
        polled = i % 2 == 0
        poll = prior + rng.gauss(0, 3) if polled else 0.0
        x = [1.0, prior, 1.0, 1.0 if prior > 0 else -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
             poll, 1.0 if polled else 0.0]
        y = prior * 0.9 + rng.gauss(0, 5)
        rows.append(_row("house", 2014 + 2 * (i % 5), x, y, poll_count=3 if polled else 0))
    return rows


# Feature vector layout (see app.features.FEATURE_NAMES):
# [intercept, prior_margin, has_prior, prior_winner, state_lean, has_state_lean,
#  environment, midterm_environment, generic_ballot, has_generic_ballot,
#  poll_average, has_polls]
def test_two_tier_routing_and_bounds():
    model = MarginModel().fit(_training(Random(3)))
    polled = _row("house", 2026, [1.0, 20.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 22.0, 1.0], None, poll_count=4)
    unpolled = _row("house", 2026, [1.0, 20.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], None, poll_count=0)
    p_polled, p_unpolled = model.predict(polled), model.predict(unpolled)
    # no generic ballot in the vector and use_generic_ballot=False -> core tier
    assert p_polled.model == "full" and p_unpolled.model == "core"
    for p in (p_polled, p_unpolled):
        assert 0.005 <= p.dem_probability <= 0.995
        assert p.mean > 5  # strong D prior must yield a D-leaning margin
    # a seat with no history gets extra variance
    no_prior = _row("house", 2026, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], None,
                    poll_count=0, has_prior=False)
    assert model.predict(no_prior).sigma > p_unpolled.sigma


def test_forecast_payload_shape():
    model = MarginModel().fit(_training(Random(5)))
    row = _row("house", 2026, [1.0, -12.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], None)
    payload = model.forecast_payload(row, "2026-house-XX-01")
    assert payload["low95"] < payload["low80"] < payload["margin"] < payload["high80"] < payload["high95"]
    assert payload["rating"]
    assert "prior_margin" in payload["components"]


def test_serialization_round_trip():
    model = MarginModel().fit(_training(Random(9)))
    restored = MarginModel.from_json(model.to_json())
    row = _row("house", 2026, [1.0, 8.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], None)
    assert abs(restored.predict(row).mean - model.predict(row).mean) < 1e-9
