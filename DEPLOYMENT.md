# Deployment

Run `docker compose up --build`; `/api/data-health` is the health endpoint. Use managed PostgreSQL, migrations, encrypted backups, structured logs and monitoring in production. Configure source credentials only in a secret manager. Scheduled workflow is an example; production refreshes must validate availability timestamps, preserve raw payloads, write a data version, then run a versioned forecast. Retraining is triggered only by completed results, validated methodology, or correction.
