"""The single place that talks to MLflow: where the server is, and how a run is opened.

Training, tuning and the pipeline call `configure()` once, then open runs with `start_run()`,
so every run lands in the same experiment and carries its code + data lineage.

Check the connection:  uv run python -m retention.tracking
"""

from __future__ import annotations

import os
import sys
import urllib.request
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from retention import lineage as lin
from retention.data.download import PROJECT_ROOT

EXPERIMENT = "customer-retention"
DEFAULT_TRACKING_URI = "http://localhost:5001"  # the docker compose stack
ENV_FILE = PROJECT_ROOT / ".env"


class TrackingServerUnavailable(RuntimeError):
    """The tracking server didn't answer its health check."""


def check_server(uri: str, timeout: float = 3.0) -> None:
    """Fail fast with a clear message instead of MLflow's minutes of silent retries.

    Only HTTP servers are checked; local stores (sqlite:///, file paths) need no server.
    """
    if not uri.startswith(("http://", "https://")):
        return
    try:
        with urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=timeout) as response:
            if response.status == 200:
                return
    except OSError as error:
        raise TrackingServerUnavailable(
            f"MLflow server at {uri} is not reachable ({error}). Start it with `docker compose up -d`."
        ) from error
    raise TrackingServerUnavailable(f"MLflow server at {uri} is unhealthy (HTTP {response.status}).")


def configure(
    tracking_uri: str | None = None, experiment: str = EXPERIMENT, env_file: Path | None = ENV_FILE
) -> str:
    """Point MLflow at the tracking server and select the experiment. Returns the URI used.

    Precedence: the `tracking_uri` argument, then the real environment, then `.env`, then the
    local default. Real environment variables win over `.env`, so CI and AWS can override it.
    """
    if env_file is not None:
        load_dotenv(env_file, override=False)
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    check_server(uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri


@contextmanager
def start_run(
    run_name: str,
    *,
    nested: bool = False,
    tags: Mapping[str, str] | None = None,
    manifest_path: Path | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Open an MLflow run stamped with lineage. Extra `tags` are added (and win on clashes)."""
    lineage_tags = lin.lineage(manifest_path or lin.MANIFEST_PATH)  # looked up per call: tests can swap it
    with mlflow.start_run(run_name=run_name, nested=nested, tags={**lineage_tags, **(tags or {})}) as run:
        yield run


def main() -> int:
    uri = configure()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    print(f"tracking URI : {uri}")
    print(f"experiment   : {EXPERIMENT} (id {experiment.experiment_id if experiment else '?'})")
    for key, value in lin.lineage().items():
        print(f"{key:<24} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
