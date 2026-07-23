# Simulation

Control simulations run in **margin space**: each race's snapshot margin
carries its own sigma (recovered from the stored 80% interval), decomposed
into a shared national shock plus idiosyncratic noise, so race errors stay
correlated the way real polling/fundamentals misses are.

The national-shock size is **computed, not guessed**:
`backtest.national_error_sigma` takes the standard deviation of out-of-sample
walk-forward cycle-level mean prediction error (real historical House
cycles ranged from -10.6 to +10.1 points) and `forecast.py` passes the
result into `simulate_control`. An earlier hardcoded 3.5-point constant
badly understated this (real value: ~5.5 for House), which let independent
per-seat noise wash out via the law of large numbers and turned a modest
seat-level lean into false chamber-level certainty (95.9% House control
against an actual near-even 218R/212D chamber) — see research claim N-002.

Official pipeline runs use 25,000 seeded draws and are stored in
`control_snapshots`; the API serves the stored result rather than
recomputing. Output includes seat distributions, means/medians, 80/95%
intervals, control probabilities, tipping-point race, and an explicit Senate
tie-break assumption plus the ingested count of Democratic-caucus seats not
up. `/api/scenarios` runs a smaller, clearly-labelled 2,000-draw variant
with a user-supplied national shift.

Future Alaska/Maine adapters must execute full ranked-choice transfer
rounds; Georgia must model runoffs separately.
