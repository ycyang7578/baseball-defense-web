"""Shared config: the project's single source of truth for the PostgreSQL connection string.

The DSN string used to be hardcoded separately in 11+ files (src/, scripts/, api/main.py),
so changing one environment (new password, new host) meant remembering to change it
everywhere. Now everything imports it from here instead.

When deploying to the cloud, set the BASEBALL_DSN environment variable (e.g. the
connection string Neon gives you) to override this without touching the code;
if it's not set locally, it defaults to connecting to the local postgres instance.
"""
import os

DSN: str = os.environ.get(
    "BASEBALL_DSN",
    "host=localhost dbname=baseball user=postgres password=postgres",
)
