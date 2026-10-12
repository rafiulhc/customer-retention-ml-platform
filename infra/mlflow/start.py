"""Start the MLflow server from separate settings (used on AWS; docker compose passes a full command).

The database password comes from Secrets Manager and may contain characters such as '@' or '/', which
break a connection URL unless encoded. Building the URL here, then replacing this process with
`mlflow server` (so it receives signals directly), keeps the secret out of the task definition.
"""

import os
from urllib.parse import quote

env = os.environ
database = (
    f"postgresql://{quote(env['DB_USER'], safe='')}:{quote(env['DB_PASSWORD'], safe='')}"
    f"@{env['DB_HOST']}:{env.get('DB_PORT', '5432')}/{env['DB_NAME']}"
)
os.execvp(
    "mlflow",
    [
        "mlflow",
        "server",
        "--host",
        "0.0.0.0",
        "--port",
        "5000",
        "--backend-store-uri",
        database,
        "--artifacts-destination",
        env["ARTIFACTS_DESTINATION"],
        "--serve-artifacts",
        "--workers",
        env.get("MLFLOW_WORKERS", "2"),
        "--allowed-hosts",
        env["ALLOWED_HOSTS"],
    ],
)
