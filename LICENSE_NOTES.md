# License notes

Application code aside, ingested datasets retain their own licenses and
terms; provenance (source, URL, license, retrieval time, payload hash) is
recorded per ingest run in `raw_sources`.

* **FiveThirtyEight data** (polls + polled-race outcomes): CC-BY-4.0 —
  attribution required to FiveThirtyEight/ABC News.
  https://github.com/fivethirtyeight/data
* **MIT Election Data + Science Lab constituency returns**: CC0 1.0 public
  domain dedication; cite MEDSL per their citation guidance.
* **@unitedstates congress-legislators**: CC0 1.0 public domain dedication.
* **FEC API data**: U.S. federal government work, public domain; respect API
  terms and rate limits (api.open.fec.gov).
* Any polling feed configured via `POLLS_FEED_URL` and any official-results
  CSVs you import carry their publishers' terms — verify permission before
  ingestion or redistribution.
