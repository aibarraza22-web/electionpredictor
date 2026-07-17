# Data sources and ingestion

Production adapters should preferentially use official state election authorities, FEC filings, Census, BLS/BEA, and licensed polling/rating sources. Every raw record requires `retrieved_at`, `available_at`, source, license/terms metadata, and immutable payload hash. Do not scrape prohibited sources or fabricate missing values. This checkout has no live data sources configured; its dashboard is synthetic demo data.
