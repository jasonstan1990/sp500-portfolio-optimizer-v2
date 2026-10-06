# Optional automatic monthly updates

The repository includes `.github/workflows/monthly_update.yml`.

- Runs automatically on the 5th of each month at 07:20 UTC.
- Can also be started manually from **GitHub → Actions → Monthly S&P 500 Snapshot Update → Run workflow**.
- Runs the unit tests first.
- Runs `update_data.py`.
- The updater writes to a staged SQLite database, checks integrity, and only then replaces `cache.db`.
- If Yahoo Finance, Wikipedia, validation, tests, or publication fails, the workflow fails and the existing `cache.db` is not committed.
- When successful, only `.portfolio_cache/cache.db` is committed automatically.

For private repositories, ensure GitHub Actions is enabled. If repository policy blocks workflow pushes, use the local `UPDATE_MONTHLY.bat` fallback.
