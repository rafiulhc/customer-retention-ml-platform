# MLflow tracking server: the official image plus the drivers it needs for our stores.
#   psycopg2 -> PostgreSQL backend store (runs, params, metrics, model registry)
#   boto3    -> S3-compatible artifact store (SeaweedFS locally, S3 on AWS)
FROM ghcr.io/mlflow/mlflow:v3.17.0

RUN pip install --no-cache-dir "psycopg2-binary==2.9.13" "boto3==1.43.104"
# On AWS the server starts from separate settings (see start.py); docker compose overrides the command.
COPY start.py /opt/mlflow/start.py
CMD ["python", "/opt/mlflow/start.py"]
