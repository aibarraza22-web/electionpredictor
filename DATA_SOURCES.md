# Data sources and ingestion

Every raw record carries `source`, `url`, `license`, `retrieved_at`,
`available_at`, and a SHA-256 payload hash in `raw_sources`; normalized rows
are inserted idempotently. Do not scrape prohibited sources or fabricate
missing values — absent inputs are flagged and widen uncertainty instead.

## Configured adapters

* **`fte_polls`** — FiveThirtyEight pollster-ratings `raw_polls.csv`
  (CC-BY-4.0, github.com/fivethirtyeight/data): 1998–2022 House/Senate
  general-election polls with field dates, plus the certified margins of
  every polled race (all Senate races; competitive House districts).
* **`legislators`** — @unitedstates `congress-legislators` (CC0): current
  members, districts, parties, and Senate term classes. Drives the 2026 race
  universe, including class-3 special elections for appointed seats, and the
  count of Democratic-caucus seats not up.
* **`medsl`** — MIT Election Data + Science Lab constituency returns
  (CC0, Harvard Dataverse: House 1976–2024 doi:10.7910/DVN/IG0UN2, Senate
  1976–2024 doi:10.7910/DVN/PEJ5QU): authoritative full-coverage district
  margins — the single highest-value source for raising House races from
  data grade D (no seat history) to C.
  * **Senate**: fetched live from Dataverse every pipeline run (no gating).
  * **House**: this file is guestbook-gated by Dataverse — even an
    authenticated `DATAVERSE_API_KEY` request was confirmed insufficient to
    satisfy it programmatically. Rather than depend on that fragile path for
    data that is final once certified (no reason to re-fetch daily), the
    adapter ships a **bundled vintage snapshot**,
    `data/vintage/medsl_us_house_1976_2024.tab` (10,863 seat-cycle rows,
    1976–2024), downloaded once through Dataverse's web UI after a
    maintainer satisfied the guestbook, and verified byte-for-byte against
    the production parser. This is the default, always-used path for House
    results — see `app/ingest/medsl.py` for full provenance and the
    (optional, best-effort) live-fetch fallback via `DATAVERSE_API_KEY`.
    **Refresh**: after a new House cycle is certified (next: ~Nov 2026),
    re-download the file from
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2
    and replace `data/vintage/medsl_us_house_1976_2024.tab`.
* **`fec`** — official FEC API candidate totals for 2026 (public domain;
  requires `FEC_API_KEY`). Displayed per race; **not** a model input until a
  finance term passes vintage-correct backtesting.
* **`polls_feed`** — live 2026 polling: any CSV in the 538 raw-polls schema,
  configured with `POLLS_FEED_URL`.
* **`scripts/import_csv.py`** — certified results from state election
  authorities (`cycle,chamber,state,district,dem_votes,rep_votes[,special]`),
  e.g. to load 2024 results before an aggregate release covers them.

## Source precedence

When multiple sources report the same seat-cycle, feature construction
prefers `official-results-csv` > `medsl-constituency-returns` >
`fivethirtyeight-raw-polls` (see `SOURCE_PRIORITY` in `app/features.py`).

## Known coverage limits (reported by /api/data-health)

* Without the MEDSL/official adapters, House seat priors exist only for
  districts polled in the prior cycle; affected forecasts carry wider
  intervals and lower quality grades.
* Incumbency reflects the current seat holder; announced retirements need a
  candidate-status source and are otherwise not marked as open seats.
