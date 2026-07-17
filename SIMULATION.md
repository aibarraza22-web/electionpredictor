# Simulation

Control simulations run in **margin space**: each race's snapshot margin
carries its own sigma (recovered from the stored 80% interval), decomposed
into a shared national shock (σ = 3.5 points) plus idiosyncratic noise, so
race errors stay correlated the way real polling/fundamentals misses are.

Official pipeline runs use 25,000 seeded draws and are stored in
`control_snapshots`; the API serves the stored result rather than
recomputing. Output includes seat distributions, means/medians, 80/95%
intervals, control probabilities, tipping-point race, and an explicit Senate
tie-break assumption plus the ingested count of Democratic-caucus seats not
up. `/api/scenarios` runs a smaller, clearly-labelled 2,000-draw variant
with a user-supplied national shift.

Future Alaska/Maine adapters must execute full ranked-choice transfer
rounds; Georgia must model runoffs separately.
